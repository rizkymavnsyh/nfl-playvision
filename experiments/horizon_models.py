"""Validation-selected horizon specialists; never use the test split for selection.

Run after `src.playvision.prepare()` has built data/processed/all_samples.csv.gz.
This is an experiment, not a replacement for the frozen verified model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'horizon_v2'
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / 'data/processed/all_samples.csv.gz', low_memory=False)
features = json.loads((ROOT / 'data/processed/features.json').read_text())
numeric = [c for c in features if c not in {'officialPosition', 'horizon'}]
prep = ColumnTransformer([
    ('num', SimpleImputer(strategy='median', add_indicator=True), numeric),
    ('pos', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['officialPosition']),
])
train = df[df.split == 'train'].copy()
valid = df[df.split == 'validation'].copy()
test = df[df.split == 'test'].copy()
assert set(train.gameId).isdisjoint(valid.gameId)
assert set(train.gameId).isdisjoint(test.gameId)
assert set(valid.gameId).isdisjoint(test.gameId)
Xtr = prep.fit_transform(train[features])
Xva = prep.transform(valid[features])
Xte = prep.transform(test[features])

def metrics(frame, dx, dy):
    e = np.hypot(frame.true_dx.to_numpy() - dx, frame.true_dy.to_numpy() - dy)
    return {'n': int(len(e)), 'mean_error_yards': float(e.mean()),
            'within_0_5_yards_pct': float(100 * np.mean(e <= .5)),
            'within_1_yard_pct': float(100 * np.mean(e <= 1))}

output = {'method': 'One LightGBM residual regressor per horizon and coordinate; train/validation split by game.',
          'selection': 'Early stopping and baseline blend chosen on validation only.',
          'caveat': 'The 2021 test games were already scored by the original project; this is a repeated benchmark, not a new blind test.',
          'horizons': {}}
models = {}
for h in [.5, 1.]:
    ti = np.flatnonzero(train.horizon.to_numpy() == h)
    vi = np.flatnonzero(valid.horizon.to_numpy() == h)
    si = np.flatnonzero(test.horizon.to_numpy() == h)
    vt = valid.iloc[vi]
    st = test.iloc[si]
    fitted = {}
    vr = []
    sr = []
    for axis in ['x', 'y']:
        model = LGBMRegressor(n_estimators=900, learning_rate=.05,
                              num_leaves=31, min_child_samples=80,
                              colsample_bytree=.85, subsample=.85,
                              reg_lambda=2., random_state=42, n_jobs=4,
                              verbosity=-1)
        model.fit(Xtr[ti], train.iloc[ti]['residual_' + axis],
                  eval_set=[(Xva[vi], vt['residual_' + axis])],
                  eval_metric='l2', callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
        fitted[axis] = model
        vr.append(model.predict(Xva[vi]))
        sr.append(model.predict(Xte[si]))
    options = []
    for base in ['cv', 'ca']:
        for weight in np.linspace(0, 1, 21):
            dx = (1-weight)*vt[base+'_dx'].to_numpy() + weight*(vt.cv_dx.to_numpy()+vr[0])
            dy = (1-weight)*vt[base+'_dy'].to_numpy() + weight*(vt.cv_dy.to_numpy()+vr[1])
            options.append((metrics(vt, dx, dy)['mean_error_yards'], base, float(weight)))
    val_error, base, weight = min(options)
    dx = (1-weight)*st[base+'_dx'].to_numpy() + weight*(st.cv_dx.to_numpy()+sr[0])
    dy = (1-weight)*st[base+'_dy'].to_numpy() + weight*(st.cv_dy.to_numpy()+sr[1])
    output['horizons'][str(h)] = {'validation_mean_error_yards': val_error,
                                  'baseline': base, 'model_weight': weight,
                                  'test': metrics(st, dx, dy),
                                  'test_by_game': {str(g): metrics(st[st.gameId == g], dx[st.gameId.to_numpy()==g], dy[st.gameId.to_numpy()==g]) for g in sorted(st.gameId.unique())},
                                  'best_iterations': {a: int(m.best_iteration_) for a,m in fitted.items()}}
    models[str(h)] = fitted
    print(h, output['horizons'][str(h)], flush=True)
joblib.dump({'preprocessor': prep, 'features': features, 'models': models,
             'choices': {h: {'baseline': v['baseline'], 'weight': v['model_weight']}
                         for h,v in output['horizons'].items()}}, OUT / 'model.joblib', compress=3)
(OUT / 'metrics.json').write_text(json.dumps(output, indent=2))

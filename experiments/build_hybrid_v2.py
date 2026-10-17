"""Combine the validation-winning 0.5s specialist with the original 1s model."""
from pathlib import Path
import json,sys
import joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from playvision import predict,point_metrics

old=joblib.load(ROOT/'models/verified/model.joblib')
new=joblib.load(ROOT/'experiments/horizon_v2/model.joblib')
v2=json.loads((ROOT/'experiments/horizon_v2/metrics.json').read_text())
old_selection=json.loads((ROOT/'models/verified/selection.json').read_text())
old_val=min(x['mean_distance_yards'] for x in old_selection if x['horizon']==.5)
assert v2['horizons']['0.5']['validation_mean_error_yards'] < old_val
bundle={'kind':'horizon_hybrid_v2','legacy':old,'specialist':new}
out=ROOT/'models/hybrid_v2';out.mkdir(exist_ok=True)
joblib.dump(bundle,out/'model.joblib',compress=3)
df=pd.read_csv(ROOT/'data/processed/all_samples.csv.gz',low_memory=False)
test=df[df.split.eq('test')].copy()
dx,dy=predict(bundle,test)
report=json.loads((ROOT/'models/verified/metrics.json').read_text())
report['model_version']='hybrid_v2: 0.5s specialist; original 1.0s model'
report['selection_rule']='Validation mean distance only; original 1.0s model retained because specialist validation error increased.'
report['evaluation_caveat']='The same two 2021 test games were already inspected in v1; this is a repeated benchmark, not a fresh blind test.'
report['validation_mean_error_yards']={'0.5_v1':old_val,'0.5_v2':v2['horizons']['0.5']['validation_mean_error_yards'],
    '1.0_v1':min(x['mean_distance_yards'] for x in old_selection if x['horizon']==1.),
    '1.0_v2':v2['horizons']['1.0']['validation_mean_error_yards']}
for h in [.5,1.]:
    mask=test.horizon.eq(h).to_numpy();s=test.loc[mask]
    report['horizons'][str(h)]['selected_model']=point_metrics(s,dx[mask],dy[mask])
    report['horizons'][str(h)]['choice']={'model':'specialist_v2' if h==.5 else 'original_v1'}
    report['horizons'][str(h)]['relative_mean_error_reduction_vs_CV_pct']=100*(1-report['horizons'][str(h)]['selected_model']['mean_distance_yards']/report['horizons'][str(h)]['constant_velocity']['mean_distance_yards'])
    print(h,report['horizons'][str(h)]['selected_model'],flush=True)
(out/'metrics.json').write_text(json.dumps(report,indent=2))

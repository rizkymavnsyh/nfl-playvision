"""External-era check on the complete 2017 KC-NE game, without tuning."""
from pathlib import Path
import json,sys
import joblib,pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from legacy2017 import load_legacy
from playvision import build_samples,predict,point_metrics

raw,plays,players=load_legacy()
samples,_,_=build_samples(raw,plays,players)
bundle=joblib.load(ROOT/'models/hybrid_v2/model.joblib')
legacy=bundle['legacy']
result={'source':'One 2017 KC-NE game; never used to fit or choose v2.', 'horizons':{}}
for h in [.5,1.]:
    s=samples[samples.horizon.eq(h)].copy()
    new_x,new_y=predict(bundle,s)
    old_x,old_y=predict(legacy,s)
    after=s.frameId.ge(s.throw_frame).to_numpy()
    result['horizons'][str(h)]={
        'all': {'v1':point_metrics(s,old_x,old_y), 'v2':point_metrics(s,new_x,new_y)},
        'post_release': {'v1':point_metrics(s.loc[after],old_x[after],old_y[after]),
                         'v2':point_metrics(s.loc[after],new_x[after],new_y[after])}}
    print(h,result['horizons'][str(h)],flush=True)
(ROOT/'reports/hybrid_v2_external2017.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

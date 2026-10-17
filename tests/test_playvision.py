import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,pytest,joblib
from playvision import ROOT,build_samples,predict,point_metrics

def linear_tracking():
    rows=[]
    for frame in range(1,46):
        for pid,team,role,x,y in [(1,'A','WR',20+.4*frame,15+.1*frame),(2,'B','CB',23+.3*frame,17+.05*frame),(-1,'football','BALL',21+.2*frame,20)]:
            rows.append({'gameId':1,'playId':10,'nflId':pid if pid!=-1 else np.nan,'frameId':frame,
              'x':x,'y':y,'team':team,'playDirection':'right','possessionTeam':'A','officialPosition':role,
              'displayName':str(pid),'event':{6:'ball_snap',20:'pass_forward',35:'pass_arrived'}.get(frame,'')})
    return pd.DataFrame(rows)

def test_constant_velocity_is_exact_on_linear_motion():
    d,f,_=build_samples(linear_tracking(),stride=1)
    m=point_metrics(d,d.cv_dx,d.cv_dy)
    assert m['mean_distance_yards']<1e-10
    assert m['within_1_yard_pct']==100

def test_future_coordinates_cannot_change_current_features():
    raw=linear_tracking();a,f,_=build_samples(raw,stride=1)
    changed=raw.copy();changed.loc[changed.frameId.gt(20),['x','y']]+=999
    b,_,_=build_samples(changed,stride=1)
    key=['gameId','playId','actor_id','frameId','horizon']
    a=a[a.frameId.le(20)].sort_values(key).reset_index(drop=True)
    b=b[b.frameId.le(20)].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[f],b[f])
    assert not np.allclose(a.true_dx,b.true_dx)

def test_predictor_allowlist_has_no_future_or_identifiers():
    _,f,_=build_samples(linear_tracking())
    assert not set(f)&{'gameId','playId','actor_id','nflId','throw_frame','arrival_frame','future_frame','true_dx','true_dy','residual_x','residual_y'}

def test_missing_frames_do_not_create_wrong_time_horizons():
    d=linear_tracking();d=d[~(d.nflId.eq(1)&d.frameId.eq(15))]
    sample,_,_=build_samples(d,stride=1)
    assert ((sample.future_frame-sample.frameId)==sample.horizon*10).all()

def test_left_right_rotation_preserves_physics():
    raw=linear_tracking();a,_,_=build_samples(raw,stride=1)
    raw['x']=120-raw.x;raw['y']=160/3-raw.y;raw['playDirection']='left'
    b,_,_=build_samples(raw,stride=1)
    np.testing.assert_allclose(a.cv_dx,b.cv_dx,atol=1e-10)
    np.testing.assert_allclose(a.true_dx,b.true_dx,atol=1e-10)
    np.testing.assert_allclose(a.true_dy,b.true_dy,atol=1e-10)

def test_frozen_model_and_game_splits():
    folder=ROOT/'models/verified'
    if not (folder/'frozen.json').exists():pytest.skip('Not trained yet')
    f=json.loads((folder/'frozen.json').read_text())
    assert hashlib.sha256((folder/'model.joblib').read_bytes()).hexdigest()==f['model_sha256']
    train,val,test=[set(f['split_games'][k]) for k in ['train','validation','test']]
    assert not train&val and not train&test and not val&test

def test_model_inference_needs_no_future_labels():
    folder=ROOT/'models/verified'
    if not (folder/'frozen.json').exists():pytest.skip('Not trained yet')
    b=joblib.load(folder/'model.joblib');x=pd.read_csv(ROOT/'examples/forecast_inputs.csv')
    dx,dy=predict(b,x);changed=x.copy();changed['true_dx']=999;changed['true_dy']=-999
    dx2,dy2=predict(b,changed);np.testing.assert_allclose(dx,dx2);np.testing.assert_allclose(dy,dy2)
    assert np.isfinite(dx).all() and np.isfinite(dy).all()

def test_inference_works_from_an_observed_prefix_only():
    from playvision import build_inputs
    raw=linear_tracking();full,f,_=build_inputs(raw)
    for cutoff in [15,25]:
        prefix,_,_=build_inputs(raw[raw.frameId.le(cutoff)])
        keys=['actor_id','frameId','horizon']
        a=full[full.frameId.le(cutoff)].sort_values(keys).reset_index(drop=True)
        b=prefix.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[f],b[f])

def test_kaggle_adapter_ignores_future_landing_and_target_role():
    from import_kaggle2026 import convert_pair
    raw=linear_tracking()
    raw=raw[raw.nflId.notna()&raw.frameId.le(20)].copy()
    inp=raw.rename(columns={'gameId':'game_id','playId':'play_id','nflId':'nfl_id','frameId':'frame_id','officialPosition':'player_position','playDirection':'play_direction'})
    inp['player_side']=np.where(inp.team.eq('A'),'Offense','Defense')
    inp['ball_land_x']=999;inp['ball_land_y']=999;inp['player_role']='Targeted Receiver'
    out=pd.DataFrame([{'game_id':1,'play_id':10,'nfl_id':1,'frame_id':i,'x':28+.4*i,'y':17+.1*i} for i in range(1,11)])
    converted=convert_pair(inp,out)
    assert 'ball_land_x' not in converted and 'player_role' not in converted
    assert converted.frameId.max()==30
    sample,features,_=build_samples(converted)
    assert len(sample)>0 and 'ball_land_x' not in features

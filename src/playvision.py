"""Short-horizon tracking forecasts. Features depend only on frames <= anchor."""
from pathlib import Path
import argparse,json,urllib.request,hashlib,time
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from flaml import AutoML

ROOT=Path(__file__).resolve().parents[1]
FPS=10;WIDTH=160/3
ROLES=['WR','TE','RB','FB','CB','FS','SS','DB','S','SAF']
KEY=['gameId','playId','actor_id'];ROWKEY=KEY+['frameId']

def dump(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def download():
    plan=json.loads((ROOT/'data/split_plan.json').read_text());commit=(ROOT/'data/source_commit.txt').read_text().strip()
    base=f'https://raw.githubusercontent.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data/{commit}/data/'
    names=['games.csv','players.csv','plays.csv']+[f'tracking_{g}.csv' for g in plan]
    rows=[];raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    for name in names:
        p=raw/name;url=base+('tracking/' if name.startswith('tracking_') else '')+name
        if not p.exists():
            for retry in range(3):
                try:
                    with urllib.request.urlopen(url,timeout=120) as r:b=r.read()
                    temp=p.with_suffix('.tmp');temp.write_bytes(b);temp.replace(p);break
                except Exception:
                    if retry==2:raise
                    time.sleep(2)
        rows.append({'file':name,'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        print('Ready',name,flush=True)
    dump(ROOT/'data/source_manifest.json',{'commit':commit,'files':rows})

def normalize(raw,plays=None,players=None):
    d=raw.copy()
    aliases={'game_id':'gameId','play_id':'playId','nfl_id':'nflId','frame_id':'frameId','display_name':'displayName','play_direction':'playDirection','player_position':'officialPosition'}
    d=d.rename(columns={k:v for k,v in aliases.items() if k in d and v not in d})
    required={'gameId','playId','nflId','frameId','x','y','team','playDirection'}
    if required-set(d):raise ValueError(f'Missing tracking columns: {required-set(d)}')
    d['actor_id']=d.nflId.fillna(-1).astype('int64');d['frameId']=d.frameId.astype('int64')
    if d.duplicated(ROWKEY).any():raise ValueError('Duplicate actor-frame records')
    if plays is not None:
        cols=[c for c in ['possessionTeam','defensiveTeam','playDescription'] if c not in d and c in plays]
        d=d.merge(plays[['gameId','playId']+cols],on=['gameId','playId'],how='left',validate='many_to_one')
    if players is not None:
        cols=[c for c in ['officialPosition','displayName'] if c not in d]
        d=d.merge(players[['nflId']+cols],on='nflId',how='left',validate='many_to_one')
    if 'officialPosition' not in d:d['officialPosition']='UNK'
    if 'displayName' not in d:d['displayName']=d.actor_id.astype(str)
    if 'possessionTeam' not in d:raise ValueError('possessionTeam is required; join plays.csv')
    if 'event' not in d:d['event']=''
    d['is_offense']=d.team.eq(d.possessionTeam).astype(int)
    d['flip']=np.where(d.playDirection.eq('left'),-1.,1.)
    d['cx']=np.where(d.flip.eq(-1),120-d.x,d.x);d['cy']=np.where(d.flip.eq(-1),WIDTH-d.y,d.y)
    return d.sort_values(ROWKEY).reset_index(drop=True)

def causal_features(d):
    d=d.sort_values(ROWKEY).copy();features=['cx','cy','is_offense','officialPosition']
    gb=d.groupby(KEY,sort=False)
    for lag in [1,2,3,5,6,9]:
        gap=d.frameId-gb.frameId.shift(lag)
        for axis in ['cx','cy']:
            col=f'{axis}_lag{lag}';d[col]=gb[axis].shift(lag).where(gap.eq(lag))
            delta=f'{axis}_delta{lag}';d[delta]=d[axis]-d[col];features.append(delta)
    d['vx']=d.cx_delta3/.3;d['vy']=d.cy_delta3/.3
    d['ax']=((d.cx-d.cx_lag3)-(d.cx_lag3-d.cx_lag6))/.09
    d['ay']=((d.cy-d.cy_lag3)-(d.cy_lag3-d.cy_lag6))/.09
    d['speed']=np.hypot(d.vx,d.vy)
    d['turn_cross']=(d.cx_lag3-d.cx_lag6)*d.cy_delta3-(d.cy_lag3-d.cy_lag6)*d.cx_delta3
    features+=['vx','vy','ax','ay','speed','turn_cross']
    # Contemporaneous spatial context, computed before applying skill-player filter.
    social=['opp_dx','opp_dy','opp_dist','opp_dvx','opp_dvy','closing_speed','mate_dist','ball_dx','ball_dy']
    d[social]=np.nan
    xy_all=d[['cx','cy']].to_numpy();v_all=np.nan_to_num(d[['vx','vy']].to_numpy())
    codes=pd.factorize(d.team)[0];actor_ids=d.actor_id.to_numpy()
    values=np.full((len(d),len(social)),np.nan)
    for ids in d.groupby(['gameId','playId','frameId'],sort=False).indices.values():
        ids=np.asarray(ids);ai=ids[actor_ids[ids]!=-1]
        if len(ai)<2:continue
        xy=xy_all[ai];v=v_all[ai]
        delta=xy[None,:,:]-xy[:,None,:];dist=np.linalg.norm(delta,axis=2)
        opp=codes[ai,None]!=codes[ai][None,:]
        if not opp.any():continue
        oi=np.argmin(np.where(opp,dist,np.inf),axis=1);rows=np.arange(len(ai))
        rel=delta[rows,oi];dv=v[oi]-v;od=dist[rows,oi]
        close=-np.sum(rel*dv,axis=1)/np.maximum(od,1e-6)
        mate=np.where(~opp,dist,np.inf);np.fill_diagonal(mate,np.inf)
        md=np.min(mate,axis=1);bi=ids[actor_ids[ids]==-1]
        bd=xy_all[bi[0]]-xy if len(bi) else np.full_like(xy,np.nan)
        values[ai]=np.column_stack([rel,od,dv,close,md,bd])
    d[social]=values
    features+=social
    return d,features

def air_limits(d):
    events=d[['gameId','playId','frameId','event']].drop_duplicates()
    start=events[events.event.isin(['pass_forward','pass_shovel'])].groupby(['gameId','playId']).frameId.min().rename('throw_frame')
    snap=events[events.event.isin(['ball_snap','autoevent_ballsnap'])].groupby(['gameId','playId']).frameId.min().rename('snap_frame')
    end=events[events.event.isin(['pass_outcome_caught','pass_outcome_incomplete','pass_outcome_interception','pass_arrived'])].groupby(['gameId','playId']).frameId.min().rename('arrival_frame')
    return pd.concat([start,end,snap],axis=1).reset_index()

def build_samples(raw,plays=None,players=None,stride=2):
    normalized=normalize(raw,plays,players);d,features=causal_features(normalized)
    d=d.merge(air_limits(d),on=['gameId','playId'],how='left',validate='many_to_one')
    d['pass_released']=d.frameId.ge(d.throw_frame).astype(int)
    d['time_since_snap']=(d.frameId-d.snap_frame)/FPS
    features+=['pass_released','time_since_snap']
    gb=d.groupby(KEY,sort=False);samples=[]
    for horizon in [5,10]:
        target_x=gb.cx.shift(-horizon);target_y=gb.cy.shift(-horizon);future_frame=gb.frameId.shift(-horizon)
        ok=d.actor_id.ne(-1)&d.officialPosition.isin(ROLES)&d.throw_frame.notna()&d.frameId.ge(d.snap_frame.fillna(0))
        ok &= (d.arrival_frame.isna()|d.frameId.le(d.arrival_frame))
        ok &= (d.frameId%stride).eq(0)&(future_frame-d.frameId).eq(horizon)
        ok &= d[['cx_lag9','cy_lag9','cx_lag6','cy_lag6','vx','vy','ax','ay']].notna().all(axis=1)
        s=d.loc[ok,ROWKEY+['nflId','displayName','team','playDirection','possessionTeam','x','y','flip','throw_frame']+features].copy()
        s['horizon']=horizon/FPS;s['cv_dx']=s.vx*s.horizon;s['cv_dy']=s.vy*s.horizon
        s['ca_dx']=s.cv_dx+.5*s.ax.clip(-12,12)*s.horizon**2
        s['ca_dy']=s.cv_dy+.5*s.ay.clip(-12,12)*s.horizon**2
        s['true_dx']=target_x[ok].to_numpy()-s.cx;s['true_dy']=target_y[ok].to_numpy()-s.cy
        s['residual_x']=s.true_dx-s.cv_dx;s['residual_y']=s.true_dy-s.cv_dy
        s['future_frame']=future_frame[ok].to_numpy().astype(int)
        samples.append(s)
    result=pd.concat(samples,ignore_index=True)
    return result,features+['horizon','cv_dx','cv_dy','ca_dx','ca_dy'],normalized

def build_inputs(raw,plays=None,players=None,stride=1):
    """Feature-only inference: no future coordinates or future-frame existence needed."""
    normalized=normalize(raw,plays,players);d,features=causal_features(normalized)
    d=d.merge(air_limits(d),on=['gameId','playId'],how='left',validate='many_to_one')
    d['pass_released']=d.frameId.ge(d.throw_frame).astype(int)
    d['time_since_snap']=(d.frameId-d.snap_frame)/FPS
    features+=['pass_released','time_since_snap'];rows=[]
    ok=d.actor_id.ne(-1)&d.officialPosition.isin(ROLES)&d.frameId.ge(d.snap_frame.fillna(0))
    ok &= (d.frameId%stride).eq(0)&d[['cx_lag9','cy_lag9','cx_lag6','cy_lag6','vx','vy','ax','ay']].notna().all(axis=1)
    for horizon in [.5,1.]:
        s=d.loc[ok,ROWKEY+['displayName','team','x','y','flip']+features].copy()
        s['horizon']=horizon;s['cv_dx']=s.vx*horizon;s['cv_dy']=s.vy*horizon
        s['ca_dx']=s.cv_dx+.5*s.ax.clip(-12,12)*horizon**2
        s['ca_dy']=s.cv_dy+.5*s.ay.clip(-12,12)*horizon**2
        rows.append(s)
    return pd.concat(rows,ignore_index=True),features+['horizon','cv_dx','cv_dy','ca_dx','ca_dy'],normalized

def prepare():
    plan=json.loads((ROOT/'data/split_plan.json').read_text())
    plays=pd.read_csv(ROOT/'data/raw/plays.csv');players=pd.read_csv(ROOT/'data/raw/players.csv')
    all_samples=[];features=None;counts={}
    cache=ROOT/'data/processed';cache.mkdir(exist_ok=True)
    for gid,split in plan.items():
        path=cache/f'samples_{gid}.csv.gz'
        if path.exists() and (cache/'features.json').exists():
            s=pd.read_csv(path,low_memory=False);features=json.loads((cache/'features.json').read_text())
        else:
            raw=pd.read_csv(ROOT/f'data/raw/tracking_{gid}.csv',low_memory=False)
            s,features,_=build_samples(raw,plays,players);s['split']=split
            temp=path.with_name(path.stem+'.tmp.csv.gz')
            s.to_csv(temp,index=False);temp.replace(path);dump(cache/'features.json',features)
        all_samples.append(s);counts[gid]={'split':split,'samples':len(s),'plays':int(s.playId.nunique())}
        print('Prepared',gid,counts[gid],flush=True)
    combined=pd.concat(all_samples,ignore_index=True)
    temp=cache/'all_samples.tmp.csv.gz'
    combined.to_csv(temp,index=False);temp.replace(cache/'all_samples.csv.gz');dump(cache/'audit.json',counts)
    return combined,features

def point_metrics(frame,dx,dy):
    err=np.hypot(frame.true_dx.to_numpy()-dx,frame.true_dy.to_numpy()-dy)
    return {'n':len(err),'mean_distance_yards':float(err.mean()),'median_distance_yards':float(np.median(err)),
      'p90_distance_yards':float(np.quantile(err,.9)),'coordinate_RMSE_yards':float(np.sqrt(np.mean(err**2)/2)),
      'within_0_5_yards_pct':float(100*np.mean(err<=.5)),'within_1_yard_pct':float(100*np.mean(err<=1)),
      'within_2_yards_pct':float(100*np.mean(err<=2))}

def predict(bundle,frame):
    if bundle.get('kind') == 'horizon_hybrid_v2':
        dx=np.empty(len(frame));dy=np.empty(len(frame))
        short=frame.horizon.eq(.5).to_numpy();long=frame.horizon.eq(1.).to_numpy()
        if short.any():
            new=bundle['specialist'];rows=frame.loc[short]
            X=new['preprocessor'].transform(rows[new['features']])
            choice=new['choices']['0.5'];w=choice['weight'];base=choice['baseline']
            dx[short]=(1-w)*rows[base+'_dx'].to_numpy()+w*(rows.cv_dx.to_numpy()+new['models']['0.5']['x'].predict(X))
            dy[short]=(1-w)*rows[base+'_dy'].to_numpy()+w*(rows.cv_dy.to_numpy()+new['models']['0.5']['y'].predict(X))
        if long.any():dx[long],dy[long]=predict(bundle['legacy'],frame.loc[long])
        if not (short|long).all():raise ValueError('Only 0.5 and 1.0 second horizons are supported')
        return dx,dy
    X=bundle['preprocessor'].transform(frame[bundle['features']])
    rx=np.asarray(bundle['x_model'].predict(X));ry=np.asarray(bundle['y_model'].predict(X))
    dx=np.empty(len(frame));dy=np.empty(len(frame))
    for h,choice in bundle['choices'].items():
        mask=frame.horizon.eq(float(h)).to_numpy();w=choice['weight'];base=choice['baseline']
        dx[mask]=(1-w)*frame.loc[mask,base+'_dx']+w*(frame.loc[mask,'cv_dx']+rx[mask])
        dy[mask]=(1-w)*frame.loc[mask,base+'_dy']+w*(frame.loc[mask,'cv_dy']+ry[mask])
    return dx,dy

def train(budget=60,max_train=100000,dataset_dir=None,run_dir=None):
    folder=Path(run_dir or ROOT/'models/verified');folder.mkdir(parents=True,exist_ok=True)
    if (folder/'frozen.json').exists():raise FileExistsError('Model is already frozen; choose a fresh project/run copy')
    path=Path(dataset_dir or ROOT/'data/processed')/'all_samples.csv.gz'
    if not path.exists() and dataset_dir is None:df,features=prepare()
    else:df=pd.read_csv(path,low_memory=False);features=json.loads((path.parent/'features.json').read_text())
    parts={s:df[df.split.eq(s)].copy() for s in ['train','validation','test']}
    sets=[set(p.gameId) for p in parts.values()]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    tr=parts['train'];va=parts['validation']
    if len(tr)>max_train:tr=tr.sample(max_train,random_state=42)
    numeric=[c for c in features if c!='officialPosition']
    prep=ColumnTransformer([('numeric',SimpleImputer(strategy='median',add_indicator=True),numeric),
                            ('position',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['officialPosition'])])
    X=prep.fit_transform(tr[features]);V=prep.transform(va[features]);models={};search={}
    for axis in ['x','y']:
        print('Training residual axis',axis,flush=True);m=AutoML()
        m.fit(X_train=X,y_train=tr['residual_'+axis].to_numpy(),X_val=V,y_val=va['residual_'+axis].to_numpy(),
          task='regression',metric='rmse',time_budget=budget,estimator_list=['lgbm','extra_tree'],n_jobs=2,
          sample=False,ensemble=False,retrain_full=False,eval_method='holdout',seed=42,verbose=1,
          log_file_name=str(folder/f'automl_{axis}.log'))
        models[axis]=m.model;search[axis]={'learner':m.best_estimator,'config':m.best_config}
    rx=models['x'].predict(V);ry=models['y'].predict(V);choices={};selection=[]
    for h in [.5,1.]:
        mask=va.horizon.eq(h).to_numpy();s=va[mask]
        options=[]
        for base in ['cv','ca']:
            for w in [0,.25,.5,.75,1]:
                dx=(1-w)*s[base+'_dx'].to_numpy()+w*(s.cv_dx.to_numpy()+rx[mask])
                dy=(1-w)*s[base+'_dy'].to_numpy()+w*(s.cv_dy.to_numpy()+ry[mask])
                met=point_metrics(s,dx,dy);options.append({'horizon':h,'baseline':base,'weight':w,**met})
        best=min(options,key=lambda z:z['mean_distance_yards']);choices[str(h)]={'baseline':best['baseline'],'weight':best['weight']};selection+=options
    b={'preprocessor':prep,'x_model':models['x'],'y_model':models['y'],'features':features,'choices':choices,'search':search,'fps':FPS}
    joblib.dump(b,folder/'model.joblib',compress=3);dump(folder/'selection.json',selection)
    dump(folder/'frozen.json',{'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
      'model_sha256':hashlib.sha256((folder/'model.joblib').read_bytes()).hexdigest(),
      'train_sample_n':len(tr),'validation_n':len(va),'test_not_scored_yet':True,
      'split_games':{k:sorted(set(v.gameId.astype(int))) for k,v in parts.items()},'budget_per_axis':budget,'dataset_dir':str(path.parent)})
    print('Model frozen; test not scored',flush=True)

def evaluate(run_dir=None,dataset_dir=None):
    folder=Path(run_dir or ROOT/'models/verified')
    if (folder/'metrics.json').exists():raise FileExistsError('Test already scored')
    b=joblib.load(folder/'model.joblib');df=pd.read_csv(Path(dataset_dir or ROOT/'data/processed')/'all_samples.csv.gz',low_memory=False)
    test=df[df.split.eq('test')].copy();dx,dy=predict(b,test)
    results=test.copy();results['pred_dx']=dx;results['pred_dy']=dy
    results['pred_x']=results.x+results.flip*dx;results['pred_y']=results.y+results.flip*dy
    results['true_x']=results.x+results.flip*results.true_dx;results['true_y']=results.y+results.flip*results.true_dy
    results['error_yards']=np.hypot(results.true_dx-dx,results.true_dy-dy)
    results.to_csv((folder/'test_predictions.csv.gz') if run_dir else (ROOT/'reports/test_predictions.csv.gz'),index=False)
    report={'source':'Real 2021 regional-event NFL tracking; not 2026 Kaggle data' if dataset_dir is None else 'User-imported tracking; see dataset provenance','test_games':sorted(set(test.gameId.astype(int))),
      'test_plays':int(test[['gameId','playId']].drop_duplicates().shape[0]),'horizons':{}}
    for h in [.5,1.]:
        mask=test.horizon.eq(h).to_numpy();s=test[mask]
        model=point_metrics(s,dx[mask],dy[mask]);cv=point_metrics(s,s.cv_dx,s.cv_dy);ca=point_metrics(s,s.ca_dx,s.ca_dy)
        moving=(mask & test.speed.gt(1).to_numpy())
        report['horizons'][str(h)]={'selected_model':model,'constant_velocity':cv,'constant_acceleration':ca,
          'relative_mean_error_reduction_vs_CV_pct':100*(1-model['mean_distance_yards']/cv['mean_distance_yards']),
          'moving_players':point_metrics(test[moving],dx[moving],dy[moving]),'choice':b['choices'][str(h)]}
    subgroup=[]
    for (h,role),s in results.groupby(['horizon','officialPosition']):
        subgroup.append({'horizon':h,'position':role,**point_metrics(s,s.pred_dx,s.pred_dy)})
    pd.DataFrame(subgroup).to_csv((folder/'position_metrics.csv') if run_dir else (ROOT/'reports/position_metrics.csv'),index=False)
    dump(folder/'metrics.json',report);print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['download','prepare','train','evaluate']);p.add_argument('--budget',type=int,default=60);p.add_argument('--max-train',type=int,default=100000);p.add_argument('--dataset-dir');p.add_argument('--run-dir');a=p.parse_args()
    if a.command=='download':download()
    elif a.command=='prepare':prepare()
    elif a.command=='train':train(a.budget,a.max_train,a.dataset_dir,a.run_dir)
    else:evaluate(a.run_dir,a.dataset_dir)

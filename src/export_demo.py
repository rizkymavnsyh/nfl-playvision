"""Build a fully offline, interactive replay from real tracking coordinates."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd,joblib
from plotly.offline import get_plotlyjs
from playvision import ROOT,build_inputs,predict,causal_features
from legacy2017 import load_legacy

def scene(raw,plays,players,bundle,label,source,focus=None):
    inputs,_,norm=build_inputs(raw,plays,players,stride=1)
    if inputs.empty:raise ValueError('No usable history in scene')
    dx,dy=predict(bundle,inputs);inputs['pred_x']=inputs.x+inputs.flip*dx;inputs['pred_y']=inputs.y+inputs.flip*dy
    causal,_=causal_features(norm)
    ball_snap=norm.loc[norm.event.eq('ball_snap'),'frameId'];start=int(ball_snap.min())-4 if len(ball_snap) else int(norm.frameId.min())
    causal=causal[causal.frameId.ge(start)]
    roster={}
    for pid,g in causal.groupby('actor_id'):
        row=g.iloc[0];name='Football' if pid==-1 else str(row.displayName)
        jersey=row.get('jerseyNumber','')
        roster[str(pid)]={'name':name,'position':str(row.officialPosition),'offense':bool(row.is_offense),
                          'jersey':'' if pd.isna(jersey) else str(int(jersey)),'team':str(row.team)}
    forecasts={}
    for row in inputs.itertuples():
        forecasts.setdefault(int(row.frameId),{}).setdefault(str(float(row.horizon)),{})[str(row.actor_id)]=[round(row.pred_x,4),round(row.pred_y,4),round(row.x+row.flip*row.cv_dx,4),round(row.y+row.flip*row.cv_dy,4)]
    frames=[]
    for f,g in causal.groupby('frameId',sort=True):
        rows=[]
        for r in g.itertuples():
            vx=0 if pd.isna(r.vx) else r.flip*r.vx;vy=0 if pd.isna(r.vy) else r.flip*r.vy
            rows.append([str(r.actor_id),round(r.x,3),round(r.y,3),round(float(vx),3),round(float(vy),3)])
        events=sorted(set(g.event.dropna().astype(str))-{''})
        frames.append({'frame':int(f),'players':rows,'events':events,'forecasts':forecasts.get(int(f),{})})
    receivers=[pid for pid,r in roster.items() if r['offense'] and r['position'] in ['WR','TE','RB','FB']]
    default=next((pid for pid in receivers if focus and focus.lower() in roster[pid]['name'].lower()),receivers[0] if receivers else None)
    return {'label':label,'source':source,'roster':roster,'frames':frames,'receivers':receivers,'defaultReceiver':default}

def build_demo(model_path=None,metrics_path=None):
    bundle=joblib.load(model_path or ROOT/'models/verified/model.joblib')
    plays=pd.read_csv(ROOT/'data/raw/plays.csv');players=pd.read_csv(ROOT/'data/raw/players.csv')
    plan=json.loads((ROOT/'data/split_plan.json').read_text());scenes=[]
    for gid,split in sorted(plan.items(),key=lambda item:({'test':0,'validation':1,'train':2}[item[1]],item[0])):
        game=pd.read_csv(ROOT/f'data/raw/tracking_{gid}.csv',low_memory=False)
        choices=plays[plays.gameId.eq(int(gid))&plays.passResult.isin(['C','I'])].sort_values('playId')
        # Fixed play-ID ordering keeps scene selection independent of model error.
        limit={'train':1,'validation':2,'test':3}[split]
        added=0
        for play in choices.itertuples():
            raw=game[game.playId.eq(play.playId)]
            label=f'2021 · {split.upper()} · {play.possessionTeam} vs {play.defensiveTeam} · Play {play.playId}'
            scope={'train':'Training game; in-sample illustration, excluded from test score',
                   'validation':'Validation game; used for model choice, excluded from test score',
                   'test':'Held-out 2021 test game; sequence ends shortly after throw'}[split]
            try:scenes.append(scene(raw,plays,players,bundle,label,scope))
            except ValueError as exc:
                if str(exc)!='No usable history in scene':raise
                continue
            added+=1
            if added>=limit:break
    legacy,lp,lplayers=load_legacy()
    for pid,name in [(938,'Harris'),(2756,'Hill'),(3725,'Hunt')]:
        raw=legacy[legacy.playId.eq(pid)]
        if raw.empty:continue
        scenes.append(scene(raw,lp,lplayers,bundle,f'2017 · KC vs NE · {name} · Play {pid}',
                     'Full replay from official NFL repository; external-era demonstration, not the 2021 scored test',name))
    metrics=json.loads(Path(metrics_path or ROOT/'models/verified/metrics.json').read_text())
    assets=ROOT/'assets';assets.mkdir(exist_ok=True)
    payload={'scenes':scenes,'metrics':metrics,'fps':10}
    (assets/'replay_data.json').write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8')
    template=(ROOT/'src/demo_template.html').read_text(encoding='utf-8')
    html=template.replace('__PLOTLY_JS__',get_plotlyjs()).replace('__DATA_JSON__',json.dumps(payload,separators=(',',':')).replace('</','<\\/'))
    (ROOT/'PlayVision_Offline.html').write_text(html,encoding='utf-8')
    print('Offline replay ready:',len(scenes),'scenes;',sum(len(x['frames']) for x in scenes),'frames',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model-path');p.add_argument('--metrics-path')
    args=p.parse_args();build_demo(args.model_path,args.metrics_path)

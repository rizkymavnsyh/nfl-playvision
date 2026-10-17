"""Import locally downloaded competition files. No login/token or rule acceptance bypass."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from playvision import build_samples,dump

def convert_pair(inp,out):
    aliases={'game_id':'gameId','play_id':'playId','nfl_id':'nflId','frame_id':'frameId','play_direction':'playDirection','player_position':'officialPosition','player_display_name':'displayName'}
    inp=inp.rename(columns=aliases).copy();out=out.rename(columns=aliases).copy()
    keys=['gameId','playId','nflId'];playkeys=['gameId','playId']
    if 'player_side' not in inp:raise ValueError('player_side required to identify current teams')
    inp['team']=np.where(inp.player_side.str.lower().eq('offense'),'Offense','Defense')
    inp['possessionTeam']='Offense'
    if 'displayName' not in inp:inp['displayName']=inp.nflId.astype(str)
    meta=inp.drop_duplicates(keys,keep='last')[keys+['team','possessionTeam','officialPosition','playDirection','displayName']]
    last=inp.groupby(playkeys).frameId.max().rename('input_end').reset_index()
    out=out.merge(last,on=playkeys,validate='many_to_one');out['frameId']=out.frameId+out.input_end
    out=out.merge(meta,on=keys,validate='many_to_one')
    if out.team.isna().any():raise ValueError('Output actors missing input metadata')
    inp=inp.merge(last,on=playkeys,validate='many_to_one')
    # The competition defines the input endpoint at release. Snap time is UNKNOWN.
    inp['event']=np.where(inp.frameId.eq(inp.input_end),'pass_forward','')
    out['event']=''
    keep=keys+['frameId','x','y','team','possessionTeam','officialPosition','playDirection','displayName','event']
    return pd.concat([inp[keep],out[keep]],ignore_index=True)

def prepare_imported(data_dir,output_dir,max_files=4,max_games=12):
    data_dir=Path(data_dir);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    files=sorted(data_dir.rglob('input_*.csv'))[:max_files]
    if not files:raise FileNotFoundError('Expected train/input_2023_wXX.csv and matching output files')
    eligible=[]
    for p in files:eligible+=pd.read_csv(p,usecols=['game_id']).game_id.unique().tolist()
    ids=sorted(set(eligible))
    if len(ids)<6:raise ValueError('Need >=6 games for disjoint train/validation/test; provide more files')
    ids=[ids[i] for i in np.linspace(0,len(ids)-1,min(max_games,len(ids))).astype(int)]
    n=len(ids);train_n=max(2,int(n*.67));val_n=max(1,int(n*.16));plan={}
    for i,g in enumerate(ids):plan[g]='train' if i<train_n else 'validation' if i<train_n+val_n else 'test'
    all_samples=[];features=None
    for p in files:
        q=p.with_name(p.name.replace('input_','output_',1))
        if not q.exists():raise FileNotFoundError(q)
        inp=pd.read_csv(p,low_memory=False);inp=inp[inp.game_id.isin(ids)]
        out=pd.read_csv(q,low_memory=False);out=out[out.game_id.isin(ids)]
        for gid,g in inp.groupby('game_id'):
            track=convert_pair(g,out[out.game_id.eq(gid)])
            sample,features,_=build_samples(track,stride=2);sample['split']=plan[gid];all_samples.append(sample)
            print('Imported game',gid,'samples',len(sample),flush=True)
    combined=pd.concat(all_samples,ignore_index=True)
    for split in ['train','validation','test']:
        if combined[combined.split.eq(split)].empty:raise ValueError(f'No usable samples in {split}')
    combined.to_csv(output_dir/'all_samples.csv.gz',index=False);dump(output_dir/'features.json',features)
    dump(output_dir/'provenance.json',{'source':'User-downloaded NFL Big Data Bowl 2026 input/output pairs',
      'note':'Rolling next-position task; not the official submission task. Input end is release; snap unknown.',
      'discarded_future_fields':['ball_land_x','ball_land_y','num_frames_output','player_role','player_to_predict'],
      'ball_tracking':'not present in these input/output files; ball features missing, no fabrication',
      'split_plan':plan,'files':[str(p.name) for p in files]})
    return combined

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--max-files',type=int,default=4);p.add_argument('--max-games',type=int,default=12);a=p.parse_args()
    prepare_imported(a.data_dir,a.output_dir,a.max_files,a.max_games)

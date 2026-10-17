"""Adapter for the single complete game published by NFL Football Operations."""
from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
from playvision import ROOT
FILES=['games.csv','players.csv','plays.csv','tracking_gameId_2017090700.csv']

def download_legacy():
    for name in FILES:
        p=ROOT/'data/raw'/('official2017_'+name)
        if not p.exists():
            with urllib.request.urlopen('https://raw.githubusercontent.com/nfl-football-ops/Big-Data-Bowl/master/Data/'+name,timeout=120) as r:p.write_bytes(r.read())

def load_legacy():
    raw=ROOT/'data/raw'
    track=pd.read_csv(raw/'official2017_tracking_gameId_2017090700.csv',low_memory=False).rename(columns={'frame.id':'frameId'})
    players=pd.read_csv(raw/'official2017_players.csv');players['displayName']=players.FirstName+' '+players.LastName
    players=players.rename(columns={'PositionAbbr':'officialPosition'})
    plays=pd.read_csv(raw/'official2017_plays.csv');game=pd.read_csv(raw/'official2017_games.csv').query('gameId == 2017090700').iloc[0]
    track['team']=track.team.replace({'home':game.homeTeamAbbr,'away':game.visitorTeamAbbr,'ball':'football'})
    t=track.merge(players[['nflId','officialPosition']],on='nflId',how='left',validate='many_to_one')
    directions={}
    for pid,g in t.groupby('playId'):
        snap=g.loc[g.event.eq('ball_snap'),'frameId']
        if snap.empty:continue
        at=g[g.frameId.eq(snap.min())];ball=at[at.team.eq('football')];qb=at[at.officialPosition.eq('QB')]
        if ball.empty or len(qb)!=1:continue
        delta=float(ball.x.iloc[0]-qb.x.iloc[0])
        if abs(delta)<.1:continue
        directions[pid]='right' if delta>0 else 'left'
    track['playDirection']=track.playId.map(directions)
    track=track.dropna(subset=['playDirection'])
    return track,plays,players

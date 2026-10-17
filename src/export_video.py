"""Render a shareable animation directly from real coordinates (not broadcast video)."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from playvision import ROOT

def export_video():
    scene=json.loads((ROOT/'assets/replay_data.json').read_text())['scenes'][3]
    roster=scene['roster'];focus=scene['defaultReceiver'];frames=scene['frames']
    fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#080e19')
    ax=fig.add_axes([.035,.19,.71,.64],facecolor='#0c352a')
    ax.set(xlim=(0,120),ylim=(0,160/3));ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_color('#36594b')
    ax.axvspan(0,10,color='#163e45');ax.axvspan(110,120,color='#163e45')
    for x in range(10,111,5):ax.axvline(x,color='#668f79' if x%10==0 else '#294d3c',lw=.7)
    for x in range(11,110):
        for y in [23.6,29.73]:ax.plot([x,x],[y-.3,y+.3],color='#436d56',lw=.65)
    ax.text(60,26.66,'PLAYVISION',ha='center',va='center',color='white',alpha=.07,fontsize=27,weight='bold')
    fig.text(.04,.94,'NFL PlayVision',color='#edf4ff',fontsize=27,weight='bold')
    fig.text(.04,.89,'Receiver–defender replay  /  +0.5 second motion forecast',color='#9aaec8',fontsize=12)
    fig.text(.78,.94,'REAL NFL TRACKING',color='#39e0ce',fontsize=10,weight='bold')
    fig.text(.78,.89,'2017 · Kansas City vs New England',color='#9aaec8',fontsize=9)
    panel=fig.add_axes([.78,.21,.19,.58],facecolor='#111b2c');panel.set_xticks([]);panel.set_yticks([])
    for spine in panel.spines.values():spine.set_color('#25334a')
    panel.text(.08,.9,'RECEIVER FOCUS',color='#9aaec8',fontsize=9,weight='bold',transform=panel.transAxes)
    panel.text(.08,.81,roster[focus]['name'],color='white',fontsize=17,weight='bold',transform=panel.transAxes)
    gap_text=panel.text(.08,.62,'',color='#39e0ce',fontsize=31,weight='bold',transform=panel.transAxes)
    panel.text(.08,.55,'YARDS TO NEAREST DB',color='#9aaec8',fontsize=8,transform=panel.transAxes)
    error_text=panel.text(.08,.36,'',color='#edf4ff',fontsize=13,transform=panel.transAxes)
    panel.text(.08,.28,'Model → actual future error',color='#9aaec8',fontsize=8,transform=panel.transAxes)
    clock=panel.text(.08,.1,'',color='#9aaec8',fontsize=10,transform=panel.transAxes)
    fig.text(.04,.125,'● Offense    ● Defense    ◇ Forecast    × Actual future',color='#9aaec8',fontsize=10)
    fig.text(.04,.068,'Illustrative external-era replay. Model trained on separate 2021 tracking; this is not broadcast footage.',color='#7f93af',fontsize=9)
    dots=ax.scatter([],[],s=72,zorder=3);ball=ax.scatter([],[],s=25,c='white',zorder=7)
    halo=ax.scatter([],[],s=220,facecolors='none',edgecolors='#39e0ce',linewidths=2,zorder=5)
    line,=ax.plot([],[],color='#39e0ce',lw=1.3,ls=':',alpha=.8)
    trail,=ax.plot([],[],color='#62a6ff',lw=2.5,alpha=.8)
    ghost,=ax.plot([],[],color='#39e0ce',lw=1.8,ls='--',marker='D',markevery=[1],ms=7)
    truth=ax.scatter([],[],s=60,c='#91f6a8',marker='x',zorder=8)
    dest=ROOT/'demo';dest.mkdir(exist_ok=True)
    writer=FFMpegWriter(fps=10,bitrate=1600,codec='libx264',extra_args=['-pix_fmt','yuv420p'])
    with writer.saving(fig,str(dest/'PlayVision_Replay.mp4'),100):
        for i,f in enumerate(frames):
            rows=[p for p in f['players'] if p[0]!='-1'];br=[p for p in f['players'] if p[0]=='-1']
            dots.set_offsets(np.array([[p[1],p[2]] for p in rows]));dots.set_color(['#62a6ff' if roster[p[0]]['offense'] else '#ffb454' for p in rows])
            ball.set_offsets(np.array([[p[1],p[2]] for p in br]).reshape(-1,2))
            p=next((p for p in rows if p[0]==focus),None)
            if p:
                halo.set_offsets([[p[1],p[2]]])
                db=[q for q in rows if not roster[q[0]]['offense'] and roster[q[0]]['position'] in ['CB','FS','SS','DB']]
                nearest=min(db,key=lambda q:(q[1]-p[1])**2+(q[2]-p[2])**2) if db else None
                if nearest:gap_text.set_text(f'{np.hypot(nearest[1]-p[1],nearest[2]-p[2]):.2f}');line.set_data([p[1],nearest[1]],[p[2],nearest[2]])
                history=[next((q for q in t['players'] if q[0]==focus),None) for t in frames[max(0,i-8):i+1]];history=[q for q in history if q]
                trail.set_data([q[1] for q in history],[q[2] for q in history])
                pred=f['forecasts'].get('0.5',{}).get(focus)
                future=next((t for t in frames if t['frame']==f['frame']+5),None)
                actual=next((q for q in future['players'] if q[0]==focus),None) if future else None
                if pred:ghost.set_data([p[1],pred[0]],[p[2],pred[1]])
                else:ghost.set_data([],[])
                if actual:truth.set_offsets([[actual[1],actual[2]]])
                else:truth.set_offsets(np.empty((0,2)))
                error_text.set_text(f'{np.hypot(pred[0]-actual[1],pred[1]-actual[2]):.2f} yd' if pred and actual else '—')
            clock.set_text(f"Frame {f['frame']}  /  {(f['frame']-frames[0]['frame'])/10:.1f}s")
            writer.grab_frame()
    plt.close(fig);print('Video exported:',dest/'PlayVision_Replay.mp4',flush=True)

if __name__=='__main__':export_video()

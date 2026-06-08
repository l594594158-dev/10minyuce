#!/usr/bin/env python3
"""4版本 × 2年全量数据 = 按月分解"""
import json, numpy as np
from datetime import datetime, timezone

DATA="/root/.openclaw/btc_1m_2year.json"; H=10; W=300
with open(DATA) as f: raw=json.load(f)
ts=np.array([r[0] for r in raw],dtype=np.int64)
c=np.array([float(r[4]) for r in raw]); v=np.array([float(r[5]) for r in raw])
h=np.array([float(r[2]) for r in raw]); l=np.array([float(r[3]) for r in raw])
N=len(c)

print(f"数据: {N:,}根, {datetime.fromtimestamp(ts[0]/1000,tz=timezone.utc)} → {datetime.fromtimestamp(ts[-1]/1000,tz=timezone.utc)}")

atr1=np.zeros(N)
for i in range(1,N): atr1[i]=h[i]-l[i]
atr20=np.zeros(N)
for i in range(20,N): atr20[i]=atr1[i-19:i+1].mean()
z20=np.full(N,np.nan)
for i in range(W,N):
    if atr20[i]>0: z20[i]=(c[i]-c[i-19:i+1].mean())/atr20[i]

rsi5=np.full(N,50.0); rsi30=np.full(N,50.0)
for i in range(10,N):
    for arr,p in [(rsi5,5),(rsi30,30)]:
        if i<p+1: continue
        d=np.diff(c[i-p:i+1]); g=np.clip(d,0,None).mean(); l2=np.clip(-d,0,None).mean()
        arr[i]=100 if l2<1e-10 else 100-100/(1+g/l2)

vsma=np.zeros(N)
for i in range(N): vsma[i]=v[max(0,i-19):i+1].mean()
cons_d=np.zeros(N,dtype=int)
for i in range(1,N):
    if c[i]<c[i-1]: cons_d[i]=cons_d[i-1]+1

M=N-W-H
sz=slice(W,N-H)
zW=np.array(z20[sz]); r5W=np.array(rsi5[sz]); r30W=np.array(rsi30[sz])
vW=np.array(v[sz]); vsW=np.array(vsma[sz]); cdW=np.array(cons_d[sz])
zOK=~np.isnan(zW)

target=np.zeros(M,dtype=np.int8)
for i in range(M): target[i]=1 if c[i+W+H]>c[i+W] else -1

month_labels=[]
for i in range(M):
    dt=datetime.fromtimestamp(ts[i+W]/1000,tz=timezone.utc)
    month_labels.append(f"{dt.year}-{dt.month:02d}")

base_up=(zW<-2.0)&(r30W<25)&zOK; base_dn=(zW>2.0)&(r30W>75)&zOK

strats=[
    ("V1.2x+R5<18+连4", base_up&(vW>vsW*1.2)&(r5W<18)&(cdW>=4), base_dn&(vW>vsW*1.2)&(r5W>82)),
    ("V2.0x+R5<22", base_up&(vW>vsW*2.0)&(r5W<22), base_dn&(vW>vsW*2.0)&(r5W>78)),
    ("V2.2x+R5<18", base_up&(vW>vsW*2.2)&(r5W<18), base_dn&(vW>vsW*2.2)&(r5W>82)),
    ("V1.2x+R5<18+连3", base_up&(vW>vsW*1.2)&(r5W<18)&(cdW>=3), base_dn&(vW>vsW*1.2)&(r5W>82)),
]

for sname, up, dn in strats:
    print(f"\n{'='*115}")
    print(f"📊 {sname}  —  2年全量回测")
    print(f"{'='*115}")
    
    # 按月统计：按月标签收集信号
    months=sorted(set(month_labels))
    
    mon_data={}
    all_pred=[]; all_tgt=[]
    for m in months:
        idx=[i for i,ml in enumerate(month_labels) if ml==m]
        
        # 多
        mi_u=[i for i in idx if up[i]]
        wins_u=sum(1 for i in mi_u if target[i]==1)
        los_u=len(mi_u)-wins_u
        # 空
        mi_d=[i for i in idx if dn[i]]
        wins_d=sum(1 for i in mi_d if target[i]==-1)
        los_d=len(mi_d)-wins_d
        
        tot=len(mi_u)+len(mi_d)
        tot_w=wins_u+wins_d
        acc=tot_w/tot*100 if tot>0 else 0
        pnl=(wins_u+wins_d)*4-(los_u+los_d)*5
        
        mon_data[m]={'u_signals':len(mi_u),'u_acc':wins_u/len(mi_u)*100 if mi_u else 0,'u_pnl':wins_u*4-los_u*5,
                      'd_signals':len(mi_d),'d_acc':wins_d/len(mi_d)*100 if mi_d else 0,'d_pnl':wins_d*4-los_d*5,
                      'tot':tot,'acc':acc,'pnl':pnl}
        
        for i in mi_u: all_pred.append(1); all_tgt.append(target[i])
        for i in mi_d: all_pred.append(-1); all_tgt.append(target[i])
    
    # 连中连挂
    mw=0; ml=0; cw=0; cl=0
    for p,t in zip(all_pred,all_tgt):
        correct=(p==t)
        if correct: cw+=1; cl=0; mw=max(mw,cw)
        else: cl+=1; cw=0; ml=max(ml,cl)
    
    print(f"{'月份':<9} {'多信号':>5} {'多胜率':>7} {'多盈':>6}  {'空信号':>5} {'空胜率':>7} {'空盈':>6}  {'总信':>5} {'总胜':>7} {'合计U':>8}")
    print(f"{'-'*115}")
    
    total_u=0; total_sig=0; total_win=0
    for m in months:
        md=mon_data[m]
        tb="⚠️" if md['acc']<50 else "  "
        print(f"{tb}{m:<9} {md['u_signals']:>5} {md['u_acc']:>6.1f}% {md['u_pnl']:>5}U {md['d_signals']:>5} {md['d_acc']:>6.1f}% {md['d_pnl']:>5}U  {md['tot']:>5} {md['acc']:>6.1f}% {md['pnl']:>+8}U")
        total_u+=md['pnl']; total_sig+=md['tot']; total_win+=md['u_signals']*md['u_acc']/100+md['d_signals']*md['d_acc']/100
    
    tacc=total_win/total_sig*100 if total_sig else 0
    print(f"{'-'*115}")
    print(f"{'合计':<9} {sum(md['u_signals'] for md in mon_data.values()):>5} {'':>7} {'':>6} {sum(md['d_signals'] for md in mon_data.values()):>5} {'':>7} {'':>6}  {total_sig:>5} {tacc:>6.1f}% {total_u:>+8}U")
    print(f"🏆 最大连中: {mw}次 | 💀 最大连挂: {ml}次 | 胜率<50%月数: {sum(1 for md in mon_data.values() if md['acc']<50)}")
    print(f"📊 日均: {total_sig/730:.1f}次 | 每笔EV: {total_u/total_sig if total_sig else 0:+.3f}U")

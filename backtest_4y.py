#!/usr/bin/env python3
"""4策略 × 4年全量回测"""
import json, numpy as np
from datetime import datetime, timezone

DATA="/root/.openclaw/btc_1m_4year.json"; H=10; W=300
with open(DATA) as f: raw=json.load(f)
ts=np.array([r[0] for r in raw],dtype=np.int64)
c=np.array([float(r[4]) for r in raw]); v=np.array([float(r[5]) for r in raw])
h=np.array([float(r[2]) for r in raw]); l=np.array([float(r[3]) for r in raw])
N=len(c)
print(f"数据: {N:,}根, {datetime.fromtimestamp(ts[0]/1000,tz=timezone.utc)} → {datetime.fromtimestamp(ts[-1]/1000,tz=timezone.utc)}")

# 预计算
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

target=np.zeros(M,dtype=np.int8)
for i in range(M): target[i]=1 if c[i+W+H]>c[i+W] else -1

month_labels=[]
for i in range(M):
    dt=datetime.fromtimestamp(int(ts[i+W])/1000,tz=timezone.utc)
    month_labels.append(f"{dt.year}-{dt.month:02d}")

zOK = ~np.isnan(zW)
base_up=(zW<-2.0)&(r30W<25)&zOK
base_dn=(zW>2.0)&(r30W>75)&zOK

strategies = [
    ("V2.0x+R5<22", base_up & (vW>vsW*2.0) & (r5W<22), base_dn & (vW>vsW*2.0) & (r5W>78)),
    ("V2.2x+R5<18", base_up & (vW>vsW*2.2) & (r5W<18), base_dn & (vW>vsW*2.2) & (r5W>82)),
    ("V1.2x+R5<18+连4", base_up & (vW>vsW*1.2) & (r5W<18) & (cdW>=4), base_dn & (vW>vsW*1.2) & (r5W>82)),
    ("V1.2x+R5<18+连3", base_up & (vW>vsW*1.2) & (r5W<18) & (cdW>=3), base_dn & (vW>vsW*1.2) & (r5W>82)),
]

def compute_stats(up_mask, dn_mask):
    n_u = up_mask.sum()
    n_d = dn_mask.sum()
    total = n_u + n_d
    if total == 0:
        return {'acc':0,'pnl':0,'total':0,'n_u':0,'n_d':0,'u_acc':0,'d_acc':0,'mw':0,'ml':0}
    
    uc = (target[up_mask] == 1).sum()
    dc = (target[dn_mask] == -1).sum()
    total_wins = uc + dc
    acc = total_wins / total * 100
    pnl = total_wins * 4 - (total - total_wins) * 5
    u_acc = uc / n_u * 100 if n_u else 0
    d_acc = dc / n_d * 100 if n_d else 0
    
    # 按月统计连中连挂
    # 先构建信号序列 (按时间顺序)
    signals = []
    for i in range(M):
        if up_mask[i]: signals.append((i, 1))
        elif dn_mask[i]: signals.append((i, -1))
    
    mw = 0; ml = 0; cw = 0; cl = 0
    for idx, pred in signals:
        if pred == target[idx]:
            cw += 1; cl = 0
            mw = max(mw, cw)
        else:
            cl += 1; cw = 0
            ml = max(ml, cl)
    
    # 按月统计胜率
    months = sorted(set(month_labels))
    bad_months = 0
    for m in months:
        mi = [i for i, ml in enumerate(month_labels) if ml == m]
        mu = sum(1 for i in mi if up_mask[i])
        md = sum(1 for i in mi if dn_mask[i])
        mtot = mu + md
        if mtot == 0: continue
        w = sum(1 for i in mi if up_mask[i] and target[i] == 1) + sum(1 for i in mi if dn_mask[i] and target[i] == -1)
        if w / mtot * 100 < 50: bad_months += 1
    
    return {'acc':acc,'pnl':pnl,'total':total,'n_u':n_u,'n_d':n_d,'u_acc':u_acc,'d_acc':d_acc,'mw':mw,'ml':ml,'bad_months':bad_months}

# 打印按月
for sname, up, dn in strategies:
    stats = compute_stats(up, dn)
    print(f"\n{'='*110}")
    print(f"📊 {sname}")
    print(f"{'='*110}")
    print(f"{'月份':<9} {'多信':>5} {'多胜':>7} {'多盈':>6}  {'空信':>5} {'空胜':>7} {'空盈':>6}  {'总信':>5} {'总胜':>7} {'月盈':>8}")
    print(f"{'-'*110}")
    
    months = sorted(set(month_labels))
    total_u = 0; total_s = 0; total_w = 0; total_u_count = 0; total_d_count = 0
    for m in months:
        idx = [i for i, ml in enumerate(month_labels) if ml == m]
        mu = [i for i in idx if up[i]]
        md = [i for i in idx if dn[i]]
        wu = sum(1 for i in mu if target[i] == 1)
        wd = sum(1 for i in md if target[i] == -1)
        lu = len(mu) - wu; ld = len(md) - wd
        tot = len(mu) + len(md); tot_w = wu + wd
        acc_m = tot_w / tot * 100 if tot else 0
        pnl_m = (wu + wd) * 4 - (lu + ld) * 5
        tb = "⚠️" if (tot > 0 and acc_m < 50) else "  "
        print(f"{tb}{m:<9} {len(mu):>5} {wu/len(mu)*100 if mu else 0:>6.1f}% {(wu*4-lu*5):>5}U {len(md):>5} {wd/len(md)*100 if md else 0:>6.1f}% {(wd*4-ld*5):>5}U {tot:>5} {acc_m:>6.1f}% {pnl_m:>+8}U")
        total_u += pnl_m; total_s += tot; total_w += tot_w
        total_u_count += len(mu); total_d_count += len(md)
    
    print(f"{'-'*110}")
    print(f"{'合计':<9} {total_u_count:>5} {'':>7} {'':>6} {total_d_count:>5} {'':>7} {'':>6} {total_s:>5} {total_w/total_s*100 if total_s else 0:>6.1f}% {total_u:>+8}U")
    print(f"连中{stats['mw']} 连挂{stats['ml']} | <50%月: {stats['bad_months']} | 日均 {total_s/1460:.1f}次")

# 汇总
print(f"\n{'='*65}")
print(f"📊 4年汇总 (2022-06 → 2026-06, 1460天)")
print(f"{'='*65}")
print(f"{'策略':<22} {'胜率':>7} {'信号':>8} {'年利润':>8} {'连挂':>5} {'<50%月':>8}")
for sname, up, dn in strategies:
    s = compute_stats(up, dn)
    print(f"{sname:<22} {s['acc']:>6.1f}% {s['total']:>8,} {s['pnl']/4:>8.0f}U {s['ml']:>5} {s['bad_months']:>8}")
print(f"\n每日5U/仓, 对+4U/错-5U, 盈亏平衡55.6%")

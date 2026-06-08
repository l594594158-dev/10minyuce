#!/usr/bin/env python3
"""4个基底策略 × 多级过滤 = 最高胜率组合"""
import json, numpy as np

DATA="/root/.openclaw/btc_1m_1year.json"; H=10; W=300
with open(DATA) as f: raw=json.load(f)
o=np.array([float(r[1]) for r in raw]); h=np.array([float(r[2]) for r in raw])
l=np.array([float(r[3]) for r in raw]); c=np.array([float(r[4]) for r in raw])
v=np.array([float(r[5]) for r in raw]); N=len(c)

# 预计算
print("预计算...",end='',flush=True)
atr1=np.zeros(N)
for i in range(1,N): atr1[i]=h[i]-l[i]
atr20=np.zeros(N)
for i in range(20,N): atr20[i]=atr1[i-19:i+1].mean()
z20=np.full(N,np.nan)
for i in range(W,N):
    if atr20[i]>0: z20[i]=(c[i]-c[i-19:i+1].mean())/atr20[i]

rsi5=np.full(N,50.0); rsi14=np.full(N,50.0); rsi30=np.full(N,50.0)
for i in range(10,N):
    for arr,p in [(rsi5,5),(rsi14,14),(rsi30,30)]:
        if i<p+1: continue
        d=np.diff(c[i-p:i+1]); g=np.clip(d,0,None).mean(); l2=np.clip(-d,0,None).mean()
        arr[i]=100 if l2<1e-10 else 100-100/(1+g/l2)

vsma=np.zeros(N)
for i in range(N): vsma[i]=v[max(0,i-19):i+1].mean()

bb_l=np.zeros(N); bb_u=np.zeros(N)
for i in range(20,N):
    w=c[i-20:i]; mu=w.mean(); s=w.std(); bb_l[i]=mu-2*s; bb_u[i]=mu+2*s

roc3=np.zeros(N); roc5=np.zeros(N)
for i in range(3,N): roc3[i]=(c[i]-c[i-3])/c[i-3]*100
for i in range(5,N): roc5[i]=(c[i]-c[i-5])/c[i-5]*100

cons_d=np.zeros(N,dtype=int)
for i in range(1,N):
    if c[i]<c[i-1]: cons_d[i]=cons_d[i-1]+1

def ema(a,p):
    r=np.zeros_like(a); r[0]=a[0]; a2=2/(p+1)
    for i in range(1,len(a)): r[i]=a2*a[i]+(1-a2)*r[i-1]
    return r
e5=ema(c,5); e26=ema(c,26)
e12=ema(c,12); e26b=ema(c,26)
macd_line=e12-e26b; macd_sig=ema(macd_line,9)

vola20=np.zeros(N)
for i in range(20,N): vola20[i]=np.std(np.diff(c[i-19:i+1])/c[i-19:i])*100

ma10=np.zeros(N); ma50=np.zeros(N)
for i in range(10,N): ma10[i]=c[i-10:i].mean()
for i in range(50,N): ma50[i]=c[i-50:i].mean()
print("完成",flush=True)

# 目标
M=N-W-H
target=np.zeros(M,dtype=np.int8)
for i in range(M): target[i]=1 if c[i+W+H]>c[i+W] else -1

# 切片
sz=slice(W,N-H)
def s(arr): return np.array(arr[sz])
zW=s(z20); r5W=s(rsi5); r14W=s(rsi14); r30W=s(rsi30)
vW=s(v); vsW=s(vsma); bblW=s(bb_l); bbuW=s(bb_u)
rc3W=s(roc3); rc5W=s(roc5); cdW=s(cons_d)
e5W=s(e5); e26W=s(e26); mW=s(macd_line); msW=s(macd_sig)
vvW=s(vola20); cW=s(c); ma10W=s(ma10); ma50W=s(ma50)
zOK=~np.isnan(zW)

# ===== 4个基底 =====
bases=[
    ("B1:Z<-2.0+RSI30<25", (zW<-2.0)&(r30W<25), (zW>2.0)&(r30W>75)),
    ("B2:Z<-1.5+MACD金叉", (zW<-1.5)&(mW>msW), (zW>1.5)&(mW<msW)),
    ("B3:Z<-2.0+RSI30<35", (zW<-2.0)&(r30W<35), (zW>2.0)&(r30W>65)),
    ("B4:MA下偏0.5%+Z<-1.2", (cW<ma50W*0.995)&(zW<-1.2), (cW>ma50W*1.005)&(zW>1.2)),
]

for bn,bu,bd in bases:
    print(f"{bn}: 多{bu.sum():,} 空{bd.sum():,} 总{bu.sum()+bd.sum():,}")

# 过滤器池
filters_pool=[
    ("V1.0x", vW>vsW, vW>vsW),
    ("V1.2x", vW>vsW*1.2, vW>vsW*1.2),
    ("V1.5x", vW>vsW*1.5, vW>vsW*1.5),
    ("V2.0x", vW>vsW*2.0, vW>vsW*2.0),
    ("BB", cW<=bblW*1.01, cW>=bbuW*0.99),
    ("BBt", cW<=bblW*1.005, cW>=bbuW*0.995),
    ("R5<20", r5W<20, r5W>80),
    ("R5<15", r5W<15, r5W>85),
    ("R14<20", r14W<20, r14W>80),
    ("连2", cdW>=2, np.ones(M,dtype=bool)),
    ("连3", cdW>=3, np.ones(M,dtype=bool)),
    ("连4", cdW>=4, np.ones(M,dtype=bool)),
    ("连5", cdW>=5, np.ones(M,dtype=bool)),
    ("R3<-0.3", rc3W<-0.3, rc3W>0.3),
    ("R3<-0.5", rc3W<-0.5, rc3W>0.5),
    ("R5<-0.3", rc5W<-0.3, rc5W>0.3),
    ("R5<-0.5", rc5W<-0.5, rc5W>0.5),
    ("EMA", e5W>e26W, e5W<e26W),
    ("MACD", mW>msW, mW<msW),
    ("波>0.1", vvW>0.1, vvW>0.1),
    ("波>0.15", vvW>0.15, vvW>0.15),
]

print(f"\n搜索: {len(bases)}基底×{len(filters_pool)}过滤器...",flush=True)

results=[]; cnt=0

def eval2(name, base_up, base_dn, f_up, f_dn, ms=100):
    global cnt
    up=base_up&f_up; dn=base_dn&f_dn
    nu=up.sum(); nd=dn.sum()
    cnt+=1
    if nu+nd<ms: return False
    uc=(target[up]==1).sum(); dc=(target[dn]==-1).sum()
    acc=(uc+dc)/(nu+nd)*100
    results.append((acc,nu+nd,uc/nu*100 if nu>0 else 0,dc/nd*100 if nd>0 else 0,nu,nd,name))
    return True

# 对每个基底：单层 + 双层组合
for bn,bu,bd in bases:
    # 单层
    for fn,fu,fd in filters_pool:
        eval2(f"{bn}+{fn}", bu, bd, fu, fd, ms=100)

    # 双层
    for i,(fi,fiu,fid) in enumerate(filters_pool):
        for j,(fj,fju,fjd) in enumerate(filters_pool):
            if i>=j: continue
            eval2(f"{bn}+{fi}+{fj}", bu, bd, fiu&fju, fid&fjd, ms=50)

# 排序
results.sort(reverse=True)
top_all=[r for r in results if r[1]>=100]

print(f"\n{'='*110}")
print(f"📊 4基底×多层过滤 — {len(top_all)}个有效组合 (信号≥100)")
print(f"{'='*110}")
print(f"{'#':<3} {'策略':<58} {'信号':>8} {'胜率':>7} {'UP':>7} {'DN':>7} {'/day':>5}")
print(f"{'-'*110}")

accs=[]
for rank,r in enumerate(top_all):
    acc,tot,ua,da,ut,dt,nm=r
    if rank>60: break
    if rank<5 or acc>=56:
        ph=tot/365
        m="🏆" if acc>=58 else ("⭐" if acc>=57 else ("🔸" if acc>=56 else "  "))
        print(f"{m}{rank:<2} {nm[:57]:<58} {tot:>8,} {acc:>6.1f}% {ua:>6.1f}% {da:>6.1f}% {ph:>5.0f}")
    accs.append(acc)

print(f"\n{'='*110}")
print(f"📊 统计")
for th in [60,58,57,56]:
    n2=sum(1 for a in top_all if a[0]>=th)
    top_at=[r for r in top_all if r[0]>=th]
    if n2>0:
        best_acc=max(r[0] for r in top_at)
        biggest=max(top_at,key=lambda r: r[1])
        print(f"  ≥{th}%: {n2} | 最高{best_acc:.1f}% | 最多信号:{biggest[-1][:50]}({biggest[1]:,})")
    else: print(f"  ≥{th}%: 0")

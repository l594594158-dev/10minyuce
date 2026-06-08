#!/usr/bin/env python3
"""在最优策略上加多层过滤，筛选出胜率>57%的组合"""
import json, numpy as np

DATA="/root/.openclaw/btc_1m_1year.json"; H=10; W=300
with open(DATA) as f: raw=json.load(f)
o=np.array([float(r[1]) for r in raw]); h=np.array([float(r[2]) for r in raw])
l=np.array([float(r[3]) for r in raw]); c=np.array([float(r[4]) for r in raw])
v=np.array([float(r[5]) for r in raw]); N=len(c)

# 预计算
atr1=np.zeros(N)
for i in range(1,N): atr1[i]=h[i]-l[i]
atr20=np.zeros(N)
for i in range(20,N): atr20[i]=atr1[i-19:i+1].mean()
z20=np.full(N,np.nan)
for i in range(W,N):
    if atr20[i]>0: z20[i]=(c[i]-c[i-19:i+1].mean())/atr20[i]
rsi5=np.full(N,50.0); rsi14=np.full(N,50.0); rsi30=np.full(N,50.0)
for i in range(10,N):
    for arr,p,out in [(rsi5,5,'r5'),(rsi14,14,'r14'),(rsi30,30,'r30')]:
        if i<p+1: continue
        d=np.diff(c[i-p:i+1]); g=np.clip(d,0,None).mean(); l2=np.clip(-d,0,None).mean()
        arr[i]=100 if l2<1e-10 else 100-100/(1+g/l2)
vsma=np.zeros(N)
for i in range(N): vsma[i]=v[max(0,i-19):i+1].mean()
bb_l=np.zeros(N); bb_u=np.zeros(N)
for i in range(20,N):
    w=c[i-20:i]; mu=w.mean(); s=w.std()
    bb_l[i]=mu-2*s; bb_u[i]=mu+2*s
roc3=np.zeros(N); roc5=np.zeros(N); roc10=np.zeros(N)
for i in range(3,N): roc3[i]=(c[i]-c[i-3])/c[i-3]*100
for i in range(5,N): roc5[i]=(c[i]-c[i-5])/c[i-5]*100
for i in range(10,N): roc10[i]=(c[i]-c[i-10])/c[i-10]*100
cons_d=np.zeros(N,dtype=int)
for i in range(1,N):
    if c[i]<c[i-1]: cons_d[i]=cons_d[i-1]+1
def ema(a,p):
    r=np.zeros_like(a); r[0]=a[0]; alpha=2/(p+1)
    for i in range(1,len(a)): r[i]=alpha*a[i]+(1-alpha)*r[i-1]
    return r
e5=ema(c,5); e26=ema(c,26); e50=ema(c,50)
macd=e12=e26_2=None
e12=ema(c,12); e26_2=ema(c,26)
macd_line=e12-e26_2; macd_sig=ema(macd_line,9)
vola20=np.zeros(N)
for i in range(20,N): vola20[i]=np.std(np.diff(c[i-19:i+1])/c[i-19:i])*100

# 目标
M=N-W-H
target=np.zeros(M,dtype=np.int8)
for i in range(M): target[i]=1 if c[i+W+H]>c[i+W] else -1
bl=np.mean(target==1)*100
print(f"基准UP: {bl:.1f}% | 预测点数: {M:,}")

# 切片
sz=slice(W,N-H)
zW=np.array(z20[sz]); r5W=np.array(rsi5[sz]); r14W=np.array(rsi14[sz])
r30W=np.array(rsi30[sz]); vW=np.array(v[sz]); vsW=np.array(vsma[sz])
bb_lW=np.array(bb_l[sz]); bb_uW=np.array(bb_u[sz])
roc3W=np.array(roc3[sz]); roc5W=np.array(roc5[sz]); roc10W=np.array(roc10[sz])
cdW=np.array(cons_d[sz]); cW=np.array(c[sz])
e5W=np.array(e5[sz]); e26W=np.array(e26[sz]); e50W=np.array(e50[sz])
mW=np.array(macd_line[sz]); msW=np.array(macd_sig[sz])
vvW=np.array(vola20[sz]); zOK=~np.isnan(zW)

# 基础策略
BASE_NAME="Z<-2.0+RSI30<25"
base_up=(zW<-2.0)&(r30W<25)
base_dn=(zW>2.0)&(r30W>75)
base_up_n=base_up.sum(); base_dn_n=base_dn.sum()
print(f"基础: {BASE_NAME} 多:{base_up_n:,} 空:{base_dn_n:,} 总:{base_up_n+base_dn_n:,}")

# 评估
results=[]
def eval_filter(name, filter_up, filter_dn, ms=200):
    up=base_up&filter_up; dn=base_dn&filter_dn
    nu=up.sum(); nd=dn.sum()
    if nu+nd<ms: return False
    uc=(target[up]==1).sum(); dc=(target[dn]==-1).sum()
    acc=(uc+dc)/(nu+nd)*100
    results.append((acc,nu+nd,uc/nu*100 if nu>0 else 0,dc/nd*100 if nd>0 else 0,nu,nd,name))
    return True

# 基准无过滤
eval_filter(BASE_NAME, np.ones(M,dtype=bool), np.ones(M,dtype=bool))

print("\n单层过滤...",flush=True)
cnt=0

# 1. 量
for vm in [1.0,1.2,1.5,2.0]:
    f_up=f_dn=vW>vsW*vm
    eval_filter(f"+量>{vm}x", f_up, f_dn); cnt+=1

# 2. BB
eval_filter("+BB下轨", cW<=bb_lW*1.01, cW>=bb_uW*0.99); cnt+=1
eval_filter("+BB下轨更紧", cW<=bb_lW*1.005, cW>=bb_uW*0.995); cnt+=1

# 3. 多周期RSI
for nm,arr,th in [("RSI5",r5W,20),("RSI5",r5W,15),("RSI14",r14W,20),("RSI14",r14W,15)]:
    eval_filter(f"+{nm}<{th}", arr<th, arr>100-th); cnt+=1

# 4. 连跌
for nc in [2,3,4,5]:
    eval_filter(f"+连跌{nc}", cdW>=nc, np.ones(M,dtype=bool)); cnt+=1

# 5. ROC
for mt in [-0.1,-0.2,-0.3,-0.5]:
    for arr,nm in [(roc3W,'ROC3'),(roc5W,'ROC5')]:
        eval_filter(f"+{nm}<{mt}%", arr<mt, arr>-mt); cnt+=1

# 6. EMA
eval_filter("+EMA多头", e5W>e26W, e5W<e26W); cnt+=1
eval_filter("+EMA强多头", (e5W>e26W)&(e26W>e50W), (e5W<e26W)&(e26W<e50W)); cnt+=1

# 7. MACD
eval_filter("+MACD<0", mW<0, mW>0); cnt+=1
eval_filter("+MACD死叉", mW<msW, mW>msW); cnt+=1

# 8. 波动率
for vf in [0.05,0.08,0.1,0.12,0.15,0.2]:
    eval_filter(f"+波>{vf}%", vvW>vf, vvW>vf); cnt+=1

print(f"单层: {cnt}策略",flush=True)

print("双层组合...",flush=True)
cnt2=0
# 最优单层 + 第二层
# 选几个最优的单层叠加
filters=[
    ("量>1.2x", vW>vsW*1.2, vW>vsW*1.2),
    ("量>1.5x", vW>vsW*1.5, vW>vsW*1.5),
    ("量>2.0x", vW>vsW*2.0, vW>vsW*2.0),
    ("BB下轨", cW<=bb_lW*1.01, cW>=bb_uW*0.99),
    ("BB紧", cW<=bb_lW*1.005, cW>=bb_uW*0.995),
    ("RSI5<20", r5W<20, r5W>80),
    ("RSI5<15", r5W<15, r5W>85),
    ("连跌3", cdW>=3, np.ones(M,dtype=bool)),
    ("连跌4", cdW>=4, np.ones(M,dtype=bool)),
    ("ROC5<-0.3%", roc5W<-0.3, roc5W>0.3),
    ("ROC5<-0.5%", roc5W<-0.5, roc5W>0.5),
    ("EMA多头", e5W>e26W, e5W<e26W),
    ("EMA强多", (e5W>e26W)&(e26W>e50W), (e5W<e26W)&(e26W<e50W)),
    ("波>0.1%", vvW>0.1, vvW>0.1),
    ("波>0.15%", vvW>0.15, vvW>0.15),
]

for i,(n1,f1u,f1d) in enumerate(filters):
    for j,(n2,f2u,f2d) in enumerate(filters):
        if i>=j: continue
        eval_filter(f"+{n1}+{n2}", f1u&f2u, f1d&f2d, ms=100); cnt2+=1

print(f"双层: {cnt2}策略",flush=True)

print("三层...",flush=True)
cnt3=0
for i,(n1,f1u,f1d) in enumerate(filters[:8]):
    for j,(n2,f2u,f2d) in enumerate(filters[:8]):
        if i>=j: continue
        for k,(n3,f3u,f3d) in enumerate(filters[:8]):
            if j>=k: continue
            eval_filter(f"+{n1}+{n2}+{n3}", f1u&f2u&f3u, f1d&f2d&f3d, ms=50); cnt3+=1

print(f"三层: {cnt3}策略",flush=True)

# 排序
results.sort(reverse=True)
top=[r for r in results if r[1]>=50]  # 至少50信号

print(f"\n{'='*105}")
print(f"📊 {BASE_NAME} + 多层过滤 ({len(top)}个有效)")
print(f"{'='*105}")
print(f"{'#':<3} {'过滤组合':<55} {'信号':>8} {'胜率':>7} {'UP':>7} {'DN':>7} {'/hr':>5}")
print(f"{'-'*105}")

for rank,r in enumerate(top,1):
    acc,tot,ua,da,ut,dt,nm=r
    if rank>50: break
    ph=tot/(365*24)
    m="🏆" if acc>=57 else ("⭐" if acc>=56.5 else ("🔸" if acc>=56 else "  "))
    print(f"{m}{rank:<2} {nm[:54]:<55} {tot:>8,} {acc:>6.1f}% {ua:>6.1f}% {da:>6.1f}% {ph:>4.0f}")

accs=[r[0] for r in top]
print(f"\n{'='*105}")
print(f"📊 {len(top)}策略 | {min(accs):.1f}%~{max(accs):.1f}%")
for th in [57,56.5,56]:
    n2=sum(1 for a in accs if a>=th)
    if n2>0:
        best=[r for r in top if r[0]>=th][-1]
        print(f"  ≥{th}%: {n2}  最优:{best[-1][:60]}({best[0]:.1f}%,{best[1]:,}信号)")
    else: print(f"  ≥{th}%: 0")

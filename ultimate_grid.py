#!/usr/bin/env python3
"""终极网格搜索：基底参数×过滤参数 全组合"""
import json, numpy as np

DATA="/root/.openclaw/btc_1m_1year.json"; H=10; W=300
with open(DATA) as f: raw=json.load(f)
c=np.array([float(r[4]) for r in raw]); v=np.array([float(r[5]) for r in raw])
h=np.array([float(r[2]) for r in raw]); l=np.array([float(r[3]) for r in raw])
N=len(c)

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
cons_d=np.zeros(N,dtype=int)
for i in range(1,N):
    if c[i]<c[i-1]: cons_d[i]=cons_d[i-1]+1
roc5=np.zeros(N)
for i in range(5,N): roc5[i]=(c[i]-c[i-5])/c[i-5]*100

M=N-W-H
target=np.zeros(M,dtype=np.int8)
for i in range(M): target[i]=1 if c[i+W+H]>c[i+W] else -1

sz=slice(W,N-H)
def s(arr): return np.array(arr[sz])
zW=s(z20); r5W=s(rsi5); r14W=s(rsi14); r30W=s(rsi30)
vW=s(v); vsW=s(vsma); cdW=s(cons_d); rc5W=s(roc5)
zOK=~np.isnan(zW)

results=[]; cnt=0
def test(name, up, dn, ms=50):
    global cnt; cnt+=1
    nu=up.sum(); nd=dn.sum()
    if nu+nd<ms: return
    uc=(target[up]==1).sum(); dc=(target[dn]==-1).sum()
    acc=(uc+dc)/(nu+nd)*100
    wins=int((nu+nd)*acc/100); losses=nu+nd-wins
    pnl=wins*4-losses*5
    results.append((acc,nu+nd,uc/nu*100 if nu>0 else 0,dc/nd*100 if nd>0 else 0,nu,nd,pnl,name))
    return True

print("终极网格...",flush=True)

# ==== 基底参数 ====
for zt in [1.8,2.0,2.2,2.5]:
    for r30 in [18,20,22,25]:
        bn=f"B(Z<-{zt}+R30<{r30})"
        bu=(zW<-zt)&(r30W<r30)&zOK; bd=(zW>zt)&(r30W>100-r30)&zOK
        
        # 无过滤
        test(bn,bu,bd)

        # ==== 单层过滤 ====
        # 量
        for vm in [1.2,1.5,1.8,2.0,2.2,2.5]:
            vo=vW>vsW*vm
            test(f"{bn}+V{vm}x", bu&vo, bd&vo)
        
        # R5
        for r5t in [12,15,18,20,22]:
            test(f"{bn}+R5<{r5t}", bu&(r5W<r5t), bd&(r5W>100-r5t))
        
        # R14
        for r14t in [18,20,22,25]:
            test(f"{bn}+R14<{r14t}", bu&(r14W<r14t), bd&(r14W>100-r14t))
        
        # 连跌
        for nc in [2,3,4,5]:
            test(f"{bn}+连{nc}", bu&(cdW>=nc), bd)
        
        # ROC5
        for rt in [-0.3,-0.5,-0.8]:
            test(f"{bn}+R5<{rt}%", bu&(rc5W<rt), bd&(rc5W>-rt))
        
        # ==== 双层 ====
        for vm in [1.2,1.5,2.0,2.2]:
            vo=vW>vsW*vm
            for r5t in [15,18,20]:
                test(f"{bn}+V{vm}x+R5<{r5t}", bu&vo&(r5W<r5t), bd&vo&(r5W>100-r5t), ms=30)
            for nc in [2,3,4]:
                test(f"{bn}+V{vm}x+连{nc}", bu&vo&(cdW>=nc), bd&vo, ms=30)
            for rt in [-0.3,-0.5]:
                test(f"{bn}+V{vm}x+R5<{rt}%", bu&vo&(rc5W<rt), bd&vo&(rc5W>-rt), ms=30)
        
        # R5+连
        for r5t in [15,18,20]:
            for nc in [2,3,4]:
                test(f"{bn}+R5<{r5t}+连{nc}", bu&(r5W<r5t)&(cdW>=nc), bd&(r5W>100-r5t), ms=30)
        
        # R5+R14
        for r5t in [15,18,20]:
            for r14t in [18,20,25]:
                test(f"{bn}+R5<{r5t}+R14<{r14t}", bu&(r5W<r5t)&(r14W<r14t), bd&(r5W>100-r5t)&(r14W>100-r14t), ms=20)

print(f"共{cnt}个",flush=True)

results.sort(reverse=True)
# 取年利润最优
results_by_pnl=sorted(results, key=lambda r:r[7], reverse=True)

print(f"\n{'='*105}")
print(f"📊 终极网格 — 胜率Top20 + 年利润Top10")
print(f"{'='*105}")
print(f"{'#':<3} {'策略':<50} {'信号':>7} {'胜率':>7} {'/day':>5} {'年利润':>8}")
print(f"{'-'*105}")

print("\n🏆 胜率排名:")
for rank,r in enumerate(results[:20],1):
    acc,tot,ua,da,ut,dt,pnl,nm=r
    m="🔴" if acc>=60 else ("🟡" if acc>=59 else "  ")
    print(f"{m}{rank:<2} {nm[:49]:<50} {tot:>7,} {acc:>6.1f}% {tot/365:>5.1f} {pnl:>8,}U")

print("\n💰 年利润排名:")
for rank,r in enumerate(results_by_pnl[:10],1):
    acc,tot,ua,da,ut,dt,pnl,nm=r
    print(f"  {rank:<2} {nm[:49]:<50} {tot:>7,} {acc:>6.1f}% {tot/365:>5.1f} {pnl:>8,}U")

print(f"\n📊 共{len(results)}个策略")
print(f"  ≥60%: {sum(1 for r in results if r[0]>=60)}")
print(f"  ≥59%: {sum(1 for r in results if r[0]>=59)}")
print(f"  年利润>900: {sum(1 for r in results if r[7]>900)}")

# 确认4个版本在不在前10
old4=['V1.2x+R5<18+连4','V2.0x+R5<22','V2.2x+R5<18','V1.2x+R5<18+连3']
found=[r for r in results if any(o in r[-1] for o in old4)]
print(f"\n🔍 原4版本在Top中:")
for r in sorted(found,key=lambda x:-x[7])[:8]:
    acc,tot,_,_,_,_,pnl,nm=r
    print(f"  {nm[:60]}: {acc:.1f}% {pnl}U {tot}信号")

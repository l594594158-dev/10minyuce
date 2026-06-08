#!/usr/bin/env python3
"""精细调参：在B1基础上遍历更细粒度的过滤参数"""
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

roc3=np.zeros(N); roc5=np.zeros(N)
for i in range(3,N): roc3[i]=(c[i]-c[i-3])/c[i-3]*100
for i in range(5,N): roc5[i]=(c[i]-c[i-5])/c[i-5]*100

M=N-W-H
target=np.zeros(M,dtype=np.int8)
for i in range(M): target[i]=1 if c[i+W+H]>c[i+W] else -1

sz=slice(W,N-H)
def s(arr): return np.array(arr[sz])
zW=s(z20); r5W=s(rsi5); r14W=s(rsi14); r30W=s(rsi30)
vW=s(v); vsW=s(vsma); cdW=s(cons_d); rc3W=s(roc3); rc5W=s(roc5)
zOK=~np.isnan(zW)

base_up=(zW<-2.0)&(r30W<25); base_dn=(zW>2.0)&(r30W>75)

results=[]; cnt=0
def test(name, fu, fd, ms=100):
    global cnt; cnt+=1
    up=base_up&fu; dn=base_dn&fd
    nu=up.sum(); nd=dn.sum()
    if nu+nd<ms: return
    uc=(target[up]==1).sum(); dc=(target[dn]==-1).sum()
    acc=(uc+dc)/(nu+nd)*100
    results.append((acc,nu+nd,uc/nu*100 if nu>0 else 0,dc/nd*100 if nd>0 else 0,nu,nd,name))

# ===== 精细参数扫描 =====
print("精细扫描...",flush=True)

# 量阈值
for vm in [1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.5,3.0]:
    vo=vW>vsW*vm
    test(f"V{vm}x", vo, vo)

# R5阈值
for rt in [10,12,15,18,20,22,25]:
    test(f"R5<{rt}", r5W<rt, r5W>100-rt)

# R14阈值
for rt in [15,18,20,22,25,30]:
    test(f"R14<{rt}", r14W<rt, r14W>100-rt)

# 连跌
for nc in [2,3,4,5,6]:
    test(f"连{nc}", cdW>=nc, np.ones(M,dtype=bool))

# ROC
for mt in [-0.1,-0.2,-0.3,-0.4,-0.5,-0.8,-1.0]:
    test(f"R3<{mt}%", rc3W<mt, rc3W>-mt)
    test(f"R5<{mt}%", rc5W<mt, rc5W>-mt)

# 组合：V + R5
for vm in [1.2,1.5,1.8,2.0,2.2,2.5]:
    for rt in [15,18,20,22]:
        vo=vW>vsW*vm
        test(f"V{vm}x+R5<{rt}", vo&(r5W<rt), vo&(r5W>100-rt), ms=50)

# V + 连
for vm in [1.2,1.5,1.8,2.0,2.2]:
    for nc in [2,3,4,5]:
        vo=vW>vsW*vm
        test(f"V{vm}x+连{nc}", vo&(cdW>=nc), vo, ms=50)

# V + ROC
for vm in [1.2,1.5,2.0,2.5]:
    for mt in [-0.3,-0.5,-0.8]:
        vo=vW>vsW*vm
        test(f"V{vm}x+R5<{mt}%", vo&(rc5W<mt), vo&(rc5W>-mt), ms=50)

# V + R5 + 连
for vm in [1.2,1.5,2.0]:
    for rt in [15,18,20]:
        for nc in [2,3,4]:
            vo=vW>vsW*vm
            test(f"V{vm}x+R5<{rt}+连{nc}", vo&(r5W<rt)&(cdW>=nc), vo&(r5W>100-rt), ms=50)

# V + R5 + R14
for vm in [1.2,1.5,2.0]:
    for rt5 in [15,18,20]:
        for rt14 in [18,20,25]:
            vo=vW>vsW*vm
            test(f"V{vm}x+R5<{rt5}+R14<{rt14}", vo&(r5W<rt5)&(r14W<rt14), vo&(r5W>100-rt5)&(r14W>100-rt14), ms=30)

print(f"共{cnt}个策略",flush=True)

results.sort(reverse=True)
print(f"\n{'='*100}")
print(f"📊 精细调参 Top 40")
print(f"{'='*100}")
print(f"{'#':<3} {'策略':<45} {'信号':>8} {'胜率':>7} {'UP':>7} {'DN':>7} {'/day':>5} {'年利润':>8}")
print(f"{'-'*100}")

stake=5; win_p=4; lose_p=5
for rank,r in enumerate(results[:40],1):
    acc,tot,ua,da,ut,dt,nm=r
    wins=int(tot*acc/100); losses=tot-wins
    pnl=wins*win_p-losses*lose_p
    ph=tot/365
    m="🏆" if acc>=58.5 else ("⭐" if acc>=58 else ("🔸" if pnl>500 else "  "))
    print(f"{m}{rank:<2} {nm[:44]:<45} {tot:>8,} {acc:>6.1f}% {ua:>6.1f}% {da:>6.1f}% {ph:>5.1f} {pnl:>8,}U")

# 最优3
accs=[r[0] for r in results]
print(f"\n📊 {len(results)}策略 | {min(accs):.1f}%~{max(accs):.1f}%")
print(f"  ≥59%: {sum(1 for a in accs if a>=59)}")
print(f"  58.5-59%: {sum(1 for a in accs if 58.5<=a<59)}")
print(f"  58-58.5%: {sum(1 for a in accs if 58<=a<58.5)}")

# 年利润排名
profit_ranked=sorted([(r[0],r[1],r[-1],int(r[1]*r[0]/100)*4-int(r[1]*(1-r[0]/100))*5) for r in results], key=lambda x:-x[3])[:10]
print(f"\n💰 年利润最优:")
for acc,tot,nm,pnl in profit_ranked:
    print(f"  {nm[:50]}: {pnl:,}U ({acc:.1f}%, {tot}信号)")

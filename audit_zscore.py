#!/usr/bin/env python3
"""
深度审计：Z-score是否偷看了未来数据
逐步骤追踪一个具体蜡烛的数据流
"""
import json, numpy as np

DATA = "/root/.openclaw/btc_1m_1year.json"

with open(DATA) as f: raw = json.load(f)
ts1m=np.array([r[0] for r in raw], dtype=np.int64)
o1m=np.array([float(r[1]) for r in raw]); h1m=np.array([float(r[2]) for r in raw])
l1m=np.array([float(r[3]) for r in raw]); c1m=np.array([float(r[4]) for r in raw])
n=len(raw)

# ===== 构建5m蜡烛 =====
kl5_o,kl5_h,kl5_l,kl5_c,kl5_ts=[],[],[],[],[]
i=0
while i<n:
    st=ts1m[i]-(ts1m[i]%300000); ed=st+300000
    o=o1m[i]; hi=l1m[i]; lo=l1m[i]; j=i
    while j<n and ts1m[j]<ed: hi=max(hi,h1m[j]); lo=min(lo,l1m[j]); j+=1
    if j>i:
        kl5_o.append(o); kl5_h.append(hi); kl5_l.append(lo)
        kl5_c.append(c1m[j-1]); kl5_ts.append(st)
    i=j
kl5_o=np.array(kl5_o); kl5_h=np.array(kl5_h); kl5_l=np.array(kl5_l)
kl5_c=np.array(kl5_c); m=len(kl5_c)

# ===== 找到第4分钟索引 =====
kl5_idx4=np.full(m,-1,dtype=np.int64)
ci=0; cnt=0; last_idx=-1
for k in range(n):
    while ci<m and ts1m[k]>=kl5_ts[ci]+300000:
        if cnt>=4: kl5_idx4[ci]=last_idx
        ci+=1; cnt=0
    if ci<m and ts1m[k]>=kl5_ts[ci]:
        cnt+=1; last_idx=k
        if cnt==4: kl5_idx4[ci]=k

# ===== 选一个具体蜡烛做审计 =====
print("="*80)
print("🔍 Z-score 数据时序审计")
print("="*80)

# 找一个有Z分数的蜡烛
target_ci = None
for ci in range(200, m):
    if kl5_idx4[ci] > 3:
        target_ci = ci
        break

if target_ci is None:
    print("未找到有效蜡烛")
    exit()

tid = kl5_idx4[target_ci]
print(f"\n🎯 审计蜡烛: ci={target_ci}, 1m_idx4={tid}")
print(f"   5m蜡烛时间: {kl5_ts[target_ci]}")
print(f"   第4分钟时间: {ts1m[tid]}")

# 展示这5分钟内每分钟的数据
st = kl5_ts[target_ci]
minutes_in_candle = []
for k in range(n):
    if st <= ts1m[k] < st + 300000:
        ms = (ts1m[k] - st) // 60000
        minutes_in_candle.append((ms, o1m[k], h1m[k], l1m[k], c1m[k]))

print(f"\n📊 当前蜡烛内每分钟数据:")
print(f"  {'分钟':>5} {'开':>10} {'高':>10} {'低':>10} {'收':>10}")
for ms, o, h, l, c in minutes_in_candle:
    marker = " ← 预测点" if ts1m[tid]==st+ms*60000 else ""
    print(f"  {ms:>5}  {o:>10.1f} {h:>10.1f} {l:>10.1f} {c:>10.1f}{marker}")

print(f"\n  蜡烛最终: O={kl5_o[target_ci]:.1f} H={kl5_h[target_ci]:.1f} L={kl5_l[target_ci]:.1f} C={kl5_c[target_ci]:.1f}")

# ===== 审计Z分数计算 =====
print(f"\n{'='*80}")
print(f"🔬 Z-score 计算审计")
print(f"{'='*80}")

# 原代码方式
c4 = c1m[tid]
print(f"\n  4分钟价格(c4): {c4:.1f}")

# 原代码: kl5_c[i-9:i+1]
ma10_original = kl5_c[target_ci-9:target_ci+1]
print(f"\n  ❌ 原代码 MA10 使用的蜡烛范围: [{target_ci-9}, {target_ci}]")
print(f"    包含当前蜡烛C[ci] = kl5_c[{target_ci}] = {kl5_c[target_ci]:.1f}")
print(f"    当前蜡烛最终收盘 ≠ 4分钟价格 → 泄漏了第5分钟数据")

# 正确方式: 只用前10根已完成蜡烛
ma10_correct = kl5_c[target_ci-10:target_ci]
print(f"\n  ✅ 正确 MA10 应使用: [{target_ci-10}, {target_ci-1}]")
print(f"    不包含当前蜡烛")

print(f"\n  MA10值对比:")
print(f"    原代码(偷看): {ma10_original.mean():.2f}")
print(f"    修正后:       {ma10_correct.mean():.2f}")
print(f"    差异:         {ma10_original.mean()-ma10_correct.mean():.2f}")

# ATR审计
print(f"\n  ATR审计:")
atr_orig = np.zeros(m)
for i in range(1,m): atr_orig[i]=kl5_h[i]-kl5_l[i]
atr_sma_orig = np.zeros(m)
for i in range(14,m): atr_sma_orig[i]=atr_orig[i-13:i+1].mean()

atr_sma_correct = atr_orig[target_ci-14:target_ci].mean()

print(f"    原代码ATR14[{target_ci}]: 包含当前蜡烛高={kl5_h[target_ci]:.1f} 低={kl5_l[target_ci]:.1f}")
print(f"    原代码ATR14:  {atr_sma_orig[target_ci]:.2f}")
print(f"    修正ATR14:    {atr_sma_correct:.2f}")

# ===== 大规模对比：原代码 vs 修正 =====
print(f"\n{'='*80}")
print(f"📊 全量对比：原代码Z分数 vs 修正Z分数")
print(f"{'='*80}")

z_orig = np.full(m, np.nan)
z_fixed = np.full(m, np.nan)

for i in range(200, m):
    idx4 = kl5_idx4[i]
    if idx4 < 3: continue
    c4 = c1m[idx4]
    
    # 原代码(偷看) — 使用 kl5_c[i-9:i+1] 包含当前蜡烛
    if i >= 10 and atr_sma_orig[i] > 0:
        ma10_o = kl5_c[i-9:i+1].mean()
        z_orig[i] = (c4 - ma10_o) / atr_sma_orig[i]
    
    # 修正(诚实) — 只用前10根已完成蜡烛
    if i >= 10 and i >= 14:
        ma10_f = kl5_c[i-10:i].mean()
        atr_f = atr_orig[i-14:i].mean()
        if atr_f > 0:
            z_fixed[i] = (c4 - ma10_f) / atr_f

# 对比差异
valid = ~np.isnan(z_orig) & ~np.isnan(z_fixed)
n_valid = valid.sum()
print(f"  有效样本: {n_valid}")
print(f"  Z原代码 mean: {np.nanmean(z_orig):.4f}  std: {np.nanstd(z_orig):.4f}")
print(f"  Z修正   mean: {np.nanmean(z_fixed):.4f}  std: {np.nanstd(z_fixed):.4f}")
print(f"  相关系数: {np.corrcoef(z_orig[valid], z_fixed[valid])[0,1]:.4f}")

# 绝对值差异分布
diffs = np.abs(z_orig[valid] - z_fixed[valid])
print(f"\n  |差异| 分位数:")
for p in [50, 75, 90, 95, 99]:
    print(f"    P{p}: {np.percentile(diffs, p):.4f}")

# 关键：信号一致性
print(f"\n{'='*80}")
print(f"📡 信号一致性：原代码 Z<-2.0 的信号中，修正后还有多少？")
print(f"{'='*80}")

for z_th in [1.5, 2.0, 2.5]:
    orig_long = (z_orig < -z_th) & valid
    orig_short = (z_orig > z_th) & valid
    fixed_long = (z_fixed < -z_th) & valid
    fixed_short = (z_fixed > z_th) & valid
    
    n_ol = orig_long.sum()
    n_os = orig_short.sum()
    
    # 保持一致 = 修正后仍在阈值内
    overlap_long = (orig_long & fixed_long).sum()
    overlap_short = (orig_short & fixed_short).sum()
    # 新增长 = 修正发现但原代码没触发
    new_long = (~orig_long & fixed_long).sum()
    new_short = (~orig_short & fixed_short).sum()
    # 丢失 = 原代码触发但修正不触发
    lost_long = (orig_long & ~fixed_long).sum()
    lost_short = (orig_short & ~fixed_short).sum()
    
    # 胜率对比
    # 修正版策略
    up_c = dn_c = up_t = dn_t = 0
    for i in range(200, m-1):
        if kl5_idx4[i] < 3: continue
        if np.isnan(z_fixed[i]): continue
        actual = 1 if kl5_c[i+1] > kl5_o[i+1] else -1
        if z_fixed[i] < -z_th:
            up_t += 1
            if actual == 1: up_c += 1
        elif z_fixed[i] > z_th:
            dn_t += 1
            if actual == -1: dn_c += 1
    
    total_f = up_t + dn_t
    acc_f = (up_c + dn_c) / total_f * 100 if total_f > 0 else 0
    
    print(f"\n  Z<{z_th}:")
    print(f"    原代码信号: {n_os+n_ol} (多头{n_ol}/空头{n_os})")
    print(f"    修正信号:   {up_t+dn_t} (多头{up_t}/空头{dn_t})")
    print(f"    重叠: {overlap_long+overlap_short}")
    print(f"    原代码独有(未来泄露): {lost_long+lost_short}")
    print(f"    修正独有: {new_long+new_short}")
    print(f"    修正胜率: {acc_f:.1f}%")
    if total_f > 0:
        print(f"    修正UP: {up_c/up_t*100:.1f}% ({up_t}) DN: {dn_c/dn_t*100:.1f}% ({dn_t})")

print(f"\n{'='*80}")
print(f"📋 结论")
print(f"{'='*80}")

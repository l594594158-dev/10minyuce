#!/usr/bin/env python3
"""
V2.0x+R5<22 策略 — 模拟交易系统
每分钟检查信号，10分钟后自动平仓，滚动日志

策略逻辑:
  做多: Z<-2.0 & RSI30<25 & 量>2.0xSMA & RSI5<22
  做空: Z>2.0 & RSI30>75 & 量>2.0xSMA & RSI5>78
  目标: 10分钟后平仓 (c[i+10] vs c[i])
"""

import json, numpy as np, time, os
from datetime import datetime, timezone
from collections import deque

# ═══════════════ 配置 ═══════════════
DATA_FILE = "/root/.openclaw/btc_1m_4year.json"
TRADE_LOG = "/root/.openclaw/trade_log.json"
SIM_SPEED = "fast"   # "fast"=立即回放  "live"=实时
START_DATE = "2026-05-01"  # 模拟起点
STAKE = 5  # 每仓U
WIN_P = 4  # 盈利
LOSE_P = 5 # 亏损
MAX_CONCURRENT = 10  # 最大同时持仓

# ═══════════════ 加载数据 ═══════════════
print(f"📂 加载数据...")
with open(DATA_FILE) as f:
    raw = json.load(f)
ts = [r[0] for r in raw]
N = len(raw)
print(f"   共{N:,}根K线")

# 找起始索引
start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
start_idx = 0
for i in range(N):
    if ts[i] >= start_ts:
        start_idx = i
        break

# 已加载的历史数据 (滑动窗口)
WARMUP = 300
buffer_c = deque(maxlen=WARMUP+50)
buffer_h = deque(maxlen=WARMUP+50)
buffer_l = deque(maxlen=WARMUP+50)
buffer_v = deque(maxlen=WARMUP+50)
buffer_ts = deque(maxlen=WARMUP+50)

for i in range(max(0, start_idx-WARMUP), start_idx):
    buffer_c.append(float(raw[i][4]))
    buffer_h.append(float(raw[i][2]))
    buffer_l.append(float(raw[i][3]))
    buffer_v.append(float(raw[i][5]))
    buffer_ts.append(ts[i])

# ═══════════════ 指标计算 ═══════════════
def calc_indicators():
    """用缓冲区最新的300根K线计算全部指标"""
    c_arr = np.array(buffer_c)
    h_arr = np.array(buffer_h)
    l_arr = np.array(buffer_l)
    v_arr = np.array(buffer_v)
    n = len(c_arr)
    if n < 30: return None
    
    # ATR20
    atr1 = h_arr - l_arr
    atr20 = atr1[-20:].mean()
    
    # Z-score
    if atr20 > 0:
        z = (c_arr[-1] - c_arr[-20:].mean()) / atr20
    else:
        z = 0
    
    # RSI30
    d30 = np.diff(c_arr[-31:])
    g = np.clip(d30, 0, None).mean()
    l2 = -np.clip(d30, None, 0).mean()
    rsi30 = 100 if l2 < 1e-10 else 100 - 100 / (1 + g / l2)
    
    # RSI5
    d5 = np.diff(c_arr[-6:])
    g5 = np.clip(d5, 0, None).mean()
    l5 = -np.clip(d5, None, 0).mean()
    rsi5 = 100 if l5 < 1e-10 else 100 - 100 / (1 + g5 / l5)
    
    # 量SMA20
    v_sma = v_arr[-20:].mean()
    
    return {
        'z': z, 'rsi30': rsi30, 'rsi5': rsi5,
        'v': v_arr[-1], 'v_sma': v_sma,
        'close': c_arr[-1], 'ts': buffer_ts[-1]
    }

def check_signal(ind):
    """检查信号"""
    if ind is None: return None
    if ind['v'] > ind['v_sma'] * 2.0:
        if ind['z'] < -2.0 and ind['rsi30'] < 25 and ind['rsi5'] < 22:
            return 'LONG'
        if ind['z'] > 2.0 and ind['rsi30'] > 75 and ind['rsi5'] > 78:
            return 'SHORT'
    return None

# ═══════════════ 交易引擎 ═══════════════
positions = []  # [{direction, entry_price, entry_time, entry_idx}]
trade_history = []
balance = 0
total_trades = 0
wins = 0
losses = 0

def settle_trades(current_idx, current_price):
    """平仓到期单"""
    global balance, wins, losses, total_trades
    closed = []
    for pos in positions:
        if current_idx - pos['entry_idx'] >= 10:
            correct = (pos['direction'] == 'LONG' and current_price > pos['entry_price']) or \
                      (pos['direction'] == 'SHORT' and current_price < pos['entry_price'])
            pnl = WIN_P if correct else -LOSE_P
            balance += pnl
            total_trades += 1
            if correct: wins += 1
            else: losses += 1
            
            result = {
                'direction': pos['direction'],
                'entry_time': pos['entry_time'],
                'exit_time': datetime.fromtimestamp(buffer_ts[-1]/1000, tz=timezone.utc).isoformat(),
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'pnl': pnl,
                'correct': correct
            }
            trade_history.append(result)
            closed.append(pos)
    
    for p in closed:
        positions.remove(p)
    return len(closed)

# ═══════════════ 主循环 ═══════════════
print(f"\n🚀 模拟开始: {START_DATE}")
print(f"{'='*70}")
print(f"{'时间':<22} {'价格':>10} {'Z':>7} {'RSI30':>7} {'RSI5':>7} {'量比':>7} {'信号':>8} {'持仓':>5} {'余额':>8}")
print(f"{'-'*70}")

last_report = time.time()
last_tick = start_idx

for i in range(start_idx, N):
    # 入缓冲区
    buffer_c.append(float(raw[i][4]))
    buffer_h.append(float(raw[i][2]))
    buffer_l.append(float(raw[i][3]))
    buffer_v.append(float(raw[i][5]))
    buffer_ts.append(ts[i])
    
    current_price = float(raw[i][4])
    
    # 平仓
    settled = settle_trades(i, current_price)
    
    # 计算信号
    ind = calc_indicators()
    sig = check_signal(ind)
    
    # 开仓
    if sig and len(positions) < MAX_CONCURRENT:
        positions.append({
            'direction': sig,
            'entry_price': current_price,
            'entry_time': datetime.fromtimestamp(ts[i]/1000, tz=timezone.utc).isoformat(),
            'entry_idx': i
        })
    
    # 日志 (每秒一次或关键时刻)
    now = time.time()
    if ind and (sig or settled > 0 or now - last_report > 60):  # 模拟加速：每100根log一次
        if i % 100 == 0 or sig or settled > 0:
            t_str = datetime.fromtimestamp(ts[i]/1000, tz=timezone.utc).strftime('%m-%d %H:%M')
            v_ratio = ind['v'] / ind['v_sma'] if ind['v_sma'] > 0 else 0
            sig_str = f"🔴{sig}" if sig else ("📤" if settled > 0 else "")
            print(f"{t_str:<22} {ind['close']:>10.1f} {ind['z']:>7.2f} {ind['rsi30']:>7.1f} {ind['rsi5']:>7.1f} {v_ratio:>7.2f} {sig_str:>8} {len(positions):>5} {balance:>+8}U")
        last_report = now

# ═══════════════ 最终结算 ═══════════════
# 平所有持仓
for pos in positions:
    current_price = float(raw[-1][4])
    correct = (pos['direction'] == 'LONG' and current_price > pos['entry_price']) or \
              (pos['direction'] == 'SHORT' and current_price < pos['entry_price'])
    pnl = WIN_P if correct else -LOSE_P
    balance += pnl
    total_trades += 1
    if correct: wins += 1
    else: losses += 1
    trade_history.append({
        'direction': pos['direction'],
        'entry_time': pos['entry_time'],
        'exit_time': 'END',
        'entry_price': pos['entry_price'],
        'exit_price': current_price,
        'pnl': pnl,
        'correct': correct
    })

# ═══════════════ 报告 ═══════════════
acc = wins / total_trades * 100 if total_trades else 0
print(f"\n{'='*70}")
print(f"📊 模拟结果")
print(f"{'='*70}")
print(f"  总交易: {total_trades}")
print(f"  胜: {wins}  负: {losses}")
print(f"  胜率: {acc:.1f}%")
print(f"  盈利: {balance:+}U")
print(f"  每笔EV: {balance/total_trades if total_trades else 0:+.3f}U")

# 按月统计
month_stats = {}
for t in trade_history:
    m = t['entry_time'][:7]
    if m not in month_stats:
        month_stats[m] = {'trades': 0, 'wins': 0, 'pnl': 0}
    month_stats[m]['trades'] += 1
    if t['correct']: month_stats[m]['wins'] += 1
    month_stats[m]['pnl'] += t['pnl']

print(f"\n📅 按月:")
print(f"{'月份':<10} {'交易':>5} {'胜率':>7} {'盈利':>8}")
for m in sorted(month_stats.keys()):
    ms = month_stats[m]
    acc_m = ms['wins']/ms['trades']*100 if ms['trades'] else 0
    print(f"  {m:<10} {ms['trades']:>5} {acc_m:>6.1f}% {ms['pnl']:>+8}U")

# 保存日志
with open(TRADE_LOG, 'w') as f:
    json.dump({
        'config': {'stake': STAKE, 'win_p': WIN_P, 'lose_p': LOSE_P, 'strategy': 'V2.0x+R5<22'},
        'summary': {'total': total_trades, 'wins': wins, 'losses': losses, 'acc': acc, 'pnl': balance},
        'trades': trade_history
    }, f, indent=2, ensure_ascii=False)
print(f"\n📁 日志: {TRADE_LOG}")

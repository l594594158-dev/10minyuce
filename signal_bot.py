#!/usr/bin/env python3
"""
⭐ 星子 — BTC 10分钟方向预测系统
策略: V2.0x+R5<22 (Z<-2.0 + RSI30<25 + 量>2xSMA + RSI5<22)
"""
import json, os, time, sys
import requests
import numpy as np
from datetime import datetime, timezone
from collections import deque

# ═══════════ 配置 ═══════════
DATA_FILE = "/root/.openclaw/signal_data.json"      # 历史K线缓存
PRED_FILE = "/root/.openclaw/predictions.json"       # 待验证预测队列
STATS_FILE = "/root/.openclaw/signal_stats.json"     # 统计
LOG_FILE = "/root/.openclaw/signal_log.json"         # 完整日志
KEEP_KLINES = 500  # 保留最近500根1mK线

# ═══════════ 公告通道 — 用message工具推送 ═══════════
def notify(msg):
    """控制台输出 + 文件记录（Telegram由外层调度）"""
    ts = datetime.now().strftime('%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)

# ═══════════ K线数据 ═══════════
def load_klines():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []

def fetch_latest_klines(limit=50):
    """从币安拉最近N根1分钟K线"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit={limit}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [
            [k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
            for k in data
        ]
    except Exception as e:
        notify(f"⚠️ 拉K线失败: {e}")
        return None

def update_klines():
    """更新本地K线缓存"""
    existing = load_klines()
    new_data = fetch_latest_klines(100)
    if not new_data:
        return existing
    
    # 合并去重
    seen = set(r[0] for r in existing)
    for k in new_data:
        if k[0] not in seen:
            existing.append(k)
            seen.add(k[0])
    
    existing.sort(key=lambda x: x[0])
    if len(existing) > KEEP_KLINES:
        existing = existing[-KEEP_KLINES:]
    
    with open(DATA_FILE, 'w') as f:
        json.dump(existing, f)
    return existing

# ═══════════ 指标计算 ═══════════
def calc_indicators(klines):
    """用K线数据计算全部指标"""
    n = len(klines)
    if n < 31: return None
    
    c = np.array([k[4] for k in klines])
    h = np.array([k[2] for k in klines])
    l = np.array([k[3] for k in klines])
    v = np.array([k[5] for k in klines])
    
    # ATR20
    atr1 = h - l
    atr20_val = atr1[-20:].mean()
    
    if atr20_val == 0:
        return None
    
    # Z-score
    ma20 = c[-20:].mean()
    z = (c[-1] - ma20) / atr20_val
    
    # RSI30
    d30 = np.diff(c[-31:])
    g30 = np.clip(d30, 0, None).mean()
    l30 = -np.clip(d30, None, 0).mean()
    rsi30 = 100 if l30 < 1e-10 else 100 - 100/(1 + g30/l30)
    
    # RSI5
    d5 = np.diff(c[-6:])
    g5 = np.clip(d5, 0, None).mean()
    l5 = -np.clip(d5, None, 0).mean()
    rsi5 = 100 if l5 < 1e-10 else 100 - 100/(1 + g5/l5)
    
    # 量比
    v_now = v[-1]
    v_sma20 = v[-20:].mean()
    v_ratio = v_now / v_sma20 if v_sma20 > 0 else 0
    
    return {
        'close': round(c[-1], 2),
        'z': round(z, 2),
        'rsi30': round(rsi30, 1),
        'rsi5': round(rsi5, 1),
        'v_ratio': round(v_ratio, 2),
        'ts': klines[-1][0]
    }

# ═══════════ 信号检测 ═══════════
def check_signal(ind):
    if ind is None: return None, None
    
    if ind['v_ratio'] > 2.0:
        if ind['z'] < -2.0 and ind['rsi30'] < 25 and ind['rsi5'] < 22:
            return 'LONG', f"Z={ind['z']} RSI30={ind['rsi30']} RSI5={ind['rsi5']} 量比={ind['v_ratio']}x"
        if ind['z'] > 2.0 and ind['rsi30'] > 75 and ind['rsi5'] > 78:
            return 'SHORT', f"Z={ind['z']} RSI30={ind['rsi30']} RSI5={ind['rsi5']} 量比={ind['v_ratio']}x"
    
    return None, None

# ═══════════ 历史秒级价格获取 ═══════════
def get_price_at_second(target_ts):
    """从缓存K线中取特定时间戳的收盘价"""
    klines = load_klines()
    for k in klines:
        if k[0] == target_ts:
            return k[4]
    return None

# ═══════════ 预测管理 ═══════════
def load_predictions():
    if os.path.exists(PRED_FILE):
        with open(PRED_FILE) as f:
            return json.load(f)
    return []

def save_predictions(preds):
    with open(PRED_FILE, 'w') as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

def add_prediction(direction, price, ts, reason):
    preds = load_predictions()
    preds.append({
        'direction': direction,
        'entry_price': price,
        'entry_ts': ts,
        'entry_time': datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        'verify_ts': ts + 600000,  # +10分钟
        'reason': reason,
        'verified': False
    })
    save_predictions(preds)

# ═══════════ 验证 ═══════════
def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            return json.load(f)
    return {'total': 0, 'wins': 0, 'losses': 0, 'current_streak_win': 0, 'current_streak_loss': 0,
            'max_win': 0, 'max_loss': 0, 'history': []}

def save_stats(st):
    with open(STATS_FILE, 'w') as f:
        json.dump(st, f, indent=2, ensure_ascii=False)

def verify_predictions():
    """检查所有待验证预测"""
    preds = load_predictions()
    stats = load_stats()
    klines = load_klines()
    price_map = {k[0]: k[4] for k in klines}
    
    verified_any = False
    new_preds = []
    
    for p in preds:
        if p['verified']:
            new_preds.append(p)
            continue
        
        verify_ts = p['verify_ts']
        # 找对应的收盘价（同一秒的K线）
        if verify_ts in price_map:
            exit_price = price_map[verify_ts]
            entry_price = p['entry_price']
            predicted_up = (p['direction'] == 'LONG')
            actual_up = exit_price > entry_price
            correct = predicted_up == actual_up
            
            # 更新统计
            stats['total'] += 1
            if correct:
                stats['wins'] += 1
                stats['current_streak_win'] += 1
                stats['current_streak_loss'] = 0
                stats['max_win'] = max(stats['max_win'], stats['current_streak_win'])
            else:
                stats['losses'] += 1
                stats['current_streak_loss'] += 1
                stats['current_streak_win'] = 0
                stats['max_loss'] = max(stats['max_loss'], stats['current_streak_loss'])
            
            acc = stats['wins'] / stats['total'] * 100
            
            # 结果记录
            result = {
                'entry_time': p['entry_time'],
                'direction': p['direction'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'correct': correct,
                'reason': p['reason']
            }
            stats['history'].append(result)
            if len(stats['history']) > 500:
                stats['history'] = stats['history'][-500:]
            
            # 通知
            emoji = '✅' if correct else '❌'
            dir_cn = '📈做多' if p['direction'] == 'LONG' else '📉做空'
            exit_time = datetime.fromtimestamp(verify_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            notify(f"{emoji} 验证: {dir_cn}")
            notify(f"   入场: {p['entry_time']} | 价格 {entry_price}")
            notify(f"   验证: {exit_time} | 价格 {exit_price}")
            notify(f"   结果: {'正确' if correct else '错误'} | 胜率 {acc:.1f}% ({stats['wins']}/{stats['total']}) | 连中 {stats['current_streak_win']} 连挂 {stats['current_streak_loss']}")
            
            p['verified'] = True
            p['correct'] = correct
            p['exit_price'] = exit_price
            verified_any = True
        else:
            # 还在等待中，检查是否超过12分钟还没数据（可能缺失）
            now_ts = int(time.time() * 1000)
            if now_ts > verify_ts + 720000:  # 超过12分钟
                ongoing_klines = fetch_latest_klines(20)
                if ongoing_klines:
                    for k in ongoing_klines:
                        price_map[k[0]] = k[4]
                    with open(DATA_FILE, 'w') as f:
                        json.dump(ongoing_klines, f)
                    # 重新尝试验证
                    if verify_ts in price_map:
                        # 会下一轮验证
                        pass
        
        new_preds.append(p)
    
    save_predictions(new_preds)
    save_stats(stats)
    return verified_any, stats

# ═══════════ 主流程 ═══════════
def run():
    notify("⭐ 星子启动 — V2.0x+R5<22 10分钟方向预测")
    
    # 1. 更新K线
    klines = update_klines()
    if not klines or len(klines) < 31:
        notify("⚠️ K线数据不足")
        return
    
    # 2. 计算指标
    ind = calc_indicators(klines)
    if ind is None:
        notify("⚠️ 指标计算失败")
        return
    
    # 3. 检查信号
    direction, reason = check_signal(ind)
    
    if direction:
        price = ind['close']
        ts = ind['ts']
        time_str = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        dir_cn = '📈做多' if direction == 'LONG' else '📉做空'
        dir_pred = '涨📈' if direction == 'LONG' else '跌📉'
        
        notify(f"🔴 信号触发!")
        notify(f"   {dir_cn} | 价格 {price} | 时间 {time_str}")
        notify(f"   指标: {reason}")
        notify(f"   预测: 10分钟后价格{dir_pred} | 到期 {datetime.fromtimestamp((ts+600000)/1000, tz=timezone.utc).strftime('%H:%M:%S')}")
        
        # 记录预测
        add_prediction(direction, price, ts, reason)
    
    # 4. 验证过期预测
    verified, stats = verify_predictions()
    
    # 5. 状态
    if not direction and not verified:
        # 无信号也无验证，偶尔输出心跳
        pass
    
    # 清理过期未验证预测（超过15分钟）
    preds = load_predictions()
    now_ts = int(time.time() * 1000)
    active = [p for p in preds if not p['verified'] and now_ts - p['verify_ts'] < 900000]
    expired = len([p for p in preds if not p['verified']]) - len(active)
    if expired > 0:
        for p in preds:
            if not p['verified'] and now_ts - p['verify_ts'] >= 900000:
                p['verified'] = True
                p['correct'] = None
                p['exit_price'] = None
        save_predictions(preds)
        notify(f"⏰ {expired}条预测过期(>15分钟无数据)")
    
    return direction, stats

if __name__ == '__main__':
    direction, stats = run()
    
    if stats and stats['total'] > 0:
        acc = stats['wins'] / stats['total'] * 100
        pending = len([p for p in load_predictions() if not p['verified']])
        print(f"\n📊 总计: {stats['total']}次 | 胜率: {acc:.1f}% | 连中:{stats['current_streak_win']} 连挂:{stats['current_streak_loss']} | 待验证:{pending}")

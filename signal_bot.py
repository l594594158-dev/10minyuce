#!/usr/bin/env python3
"""
⭐ 星子 — BTC 10分钟方向预测系统 (秒级守护进程)
策略: V2.0x+R5<22 (Z<-2.0 + RSI30<25 + 量>2xSMA + RSI5<22)
每秒扫描1分钟K线是否收盘 → 信号触发 → 实时价格记录 → 10分钟后验证
"""
import json, os, time, sys, signal
import requests
import numpy as np
from datetime import datetime, timezone

# ═══════════ 配置 ═══════════
DATA_FILE   = "/root/.openclaw/signal_data.json"
PRED_FILE   = "/root/.openclaw/predictions.json"
STATS_FILE  = "/root/.openclaw/signal_stats.json"
NOTIFY_FILE = "/root/.openclaw/signal_notify.json"
KEEP_KLINES = 500

# ═══════════ 通知 ═══════════
def notify(msg):
    """控制台输出 + 写入通知队列(由外层cron转发Telegram)"""
    ts = datetime.now().strftime('%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    # 写入通知队列
    try:
        queue = []
        if os.path.exists(NOTIFY_FILE):
            with open(NOTIFY_FILE) as f:
                queue = json.load(f)
        queue.append({"ts": ts, "msg": msg})
        with open(NOTIFY_FILE, 'w') as f:
            json.dump(queue[-200:], f)  # 只保留最近200条
    except:
        pass

def drain_notifications():
    """清空通知队列并返回待发送列表"""
    if not os.path.exists(NOTIFY_FILE):
        return []
    with open(NOTIFY_FILE) as f:
        queue = json.load(f)
    # 清空文件
    with open(NOTIFY_FILE, 'w') as f:
        json.dump([], f)
    return queue

# ═══════════ 数据获取 ═══════════
def api_get(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        notify(f"⚠️ API失败: {e}")
        return None

def fetch_klines(limit=100):
    """拉最近N根1m K线"""
    data = api_get(f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit={limit}")
    if not data: return None
    return [[k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]

def fetch_ticker():
    """实时价格"""
    data = api_get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
    if data: return float(data['price'])
    return None

def ms_now():
    return int(time.time() * 1000)

def ts_to_str(ts_ms):
    return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

# ═══════════ K线缓存 ═══════════
def load_cached_klines():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []

def save_klines(klines):
    if len(klines) > KEEP_KLINES:
        klines = klines[-KEEP_KLINES:]
    with open(DATA_FILE, 'w') as f:
        json.dump(klines, f)

# ═══════════ 指标计算 ═══════════
def calc_indicators(klines):
    n = len(klines)
    if n < 31: return None
    
    c = np.array([k[4] for k in klines])
    h = np.array([k[2] for k in klines])
    l = np.array([k[3] for k in klines])
    v = np.array([k[5] for k in klines])
    
    atr20 = (h[-20:] - l[-20:]).mean()
    if atr20 == 0: return None
    
    ma20 = c[-20:].mean()
    z = (c[-1] - ma20) / atr20
    
    d30 = np.diff(c[-31:])
    g30 = np.clip(d30, 0, None).mean()
    l30 = -np.clip(d30, None, 0).mean()
    rsi30 = 100 if l30 < 1e-10 else 100 - 100/(1 + g30/l30)
    
    d5 = np.diff(c[-6:])
    g5 = np.clip(d5, 0, None).mean()
    l5 = -np.clip(d5, None, 0).mean()
    rsi5 = 100 if l5 < 1e-10 else 100 - 100/(1 + g5/l5)
    
    v_ratio = v[-1] / v[-20:].mean() if v[-20:].mean() > 0 else 0
    
    return {
        'close': round(c[-1], 2),
        'z': round(z, 2),
        'rsi30': round(rsi30, 1),
        'rsi5': round(rsi5, 1),
        'v_ratio': round(v_ratio, 2),
        'ts': klines[-1][0]
    }

def check_signal(ind):
    if ind is None or ind['v_ratio'] <= 2.0:
        return None, None
    if ind['z'] < -2.0 and ind['rsi30'] < 25 and ind['rsi5'] < 22:
        return 'LONG', f"Z={ind['z']} RSI30={ind['rsi30']} RSI5={ind['rsi5']} 量比={ind['v_ratio']}x"
    if ind['z'] > 2.0 and ind['rsi30'] > 75 and ind['rsi5'] > 78:
        return 'SHORT', f"Z={ind['z']} RSI30={ind['rsi30']} RSI5={ind['rsi5']} 量比={ind['v_ratio']}x"
    return None, None

# ═══════════ 预测管理 ═══════════
def load_predictions():
    if os.path.exists(PRED_FILE):
        with open(PRED_FILE) as f:
            return json.load(f)
    return []

def save_predictions(preds):
    with open(PRED_FILE, 'w') as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

def add_prediction(direction, reason, rt_price):
    now_ms = ms_now()
    preds = load_predictions()
    preds.append({
        'direction': direction,
        'entry_price': rt_price,
        'entry_ts': now_ms,
        'entry_time': ts_to_str(now_ms),
        'verify_ts': now_ms + 600000,
        'reason': reason,
        'verified': False
    })
    save_predictions(preds)
    return now_ms

# ═══════════ 统计 ═══════════
def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            return json.load(f)
    return {'total': 0, 'wins': 0, 'losses': 0,
            'current_streak_win': 0, 'current_streak_loss': 0,
            'max_win': 0, 'max_loss': 0}

def save_stats(st):
    with open(STATS_FILE, 'w') as f:
        json.dump(st, f, indent=2, ensure_ascii=False)

# ═══════════ 验证 ═══════════
def verify_predictions():
    """用实时价格验证到期的预测"""
    preds = load_predictions()
    stats = load_stats()
    now = ms_now()
    verified_any = False
    
    new_preds = []
    for p in preds:
        if p['verified']:
            new_preds.append(p)
            continue
        
        if now >= p['verify_ts']:
            exit_price = fetch_ticker()
            if exit_price is None:
                new_preds.append(p)
                continue
            
            entry_price = p['entry_price']
            correct = (exit_price > entry_price) == (p['direction'] == 'LONG')
            
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
            vtime = ts_to_str(ms_now())
            change = exit_price - entry_price
            arrow = '🔺' if change > 0 else '🔻' if change < 0 else '➡️'
            emoji = '✅' if correct else '❌'
            dir_cn = '📈做多' if p['direction'] == 'LONG' else '📉做空'
            
            notify(f"{emoji} 验证: {dir_cn}")
            notify(f"   入场时间: {p['entry_time']}")
            notify(f"   入场价格: {round(entry_price, 2)}")
            notify(f"   验证时间: {vtime}")
            notify(f"   验证价格: {round(exit_price, 2)}")
            notify(f"   变动: {change:+.2f} {arrow}")
            notify(f"   胜率: {acc:.1f}% ({stats['wins']}/{stats['total']}) | 连中{stats['current_streak_win']} 连挂{stats['current_streak_loss']}")
            
            p['verified'] = True
            p['correct'] = correct
            p['exit_price'] = exit_price
            p['verify_time'] = vtime
            verified_any = True
        else:
            new_preds.append(p)
    
    # 清理超过15分钟未验证
    for p in list(new_preds):
        if not p['verified'] and now > p['verify_ts'] + 900000:
            p['verified'] = True
            p['correct'] = None
            p['exit_price'] = None
    
    save_predictions(new_preds)
    save_stats(stats)
    return verified_any, stats

# ═══════════ 守护进程主循环 ═══════════
def run_daemon():
    notify("⭐ 星子守护进程启动 — V2.0x+R5<22 每秒扫描")
    
    # 加载初始数据
    klines = load_cached_klines()
    if not klines:
        notify("📡 首次拉取K线...")
        klines = fetch_klines(100) or []
        if klines:
            save_klines(klines)
    
    if len(klines) < 31:
        notify(f"⚠️ K线不足({len(klines)}根)，持续等待...")
    
    last_candle_ts = klines[-1][0] if klines else 0
    tick = 0
    
    while True:
        try:
            time.sleep(1)
            tick += 1
            now_ms = ms_now()
            
            # 每秒: 拉最新1根K线检查是否收盘
            new_klines = fetch_klines(2)
            if not new_klines:
                continue
            
            latest = new_klines[-1]
            latest_ts = latest[0]
            
            # 新蜡烛收盘!
            if latest_ts != last_candle_ts:
                last_candle_ts = latest_ts
                
                # 合并到本地缓存
                seen = {k[0] for k in klines}
                for k in new_klines:
                    if k[0] not in seen:
                        klines.append(k)
                        seen.add(k[0])
                klines.sort(key=lambda x: x[0])
                if len(klines) > KEEP_KLINES:
                    klines = klines[-KEEP_KLINES:]
                save_klines(klines)
                
                # 计算指标 + 检查信号
                ind = calc_indicators(klines)
                if ind:
                    direction, reason = check_signal(ind)
                    if direction:
                        rt_price = fetch_ticker()
                        if rt_price:
                            dir_cn = '📈做多' if direction == 'LONG' else '📉做空'
                            dir_pred = '涨📈' if direction == 'LONG' else '跌📉'
                            now_str = ts_to_str(ms_now())
                            expire_str = ts_to_str(ms_now() + 600000)[-8:]
                            
                            notify(f"🔴 信号触发!")
                            notify(f"   {dir_cn} | 实时价格 {rt_price}")
                            notify(f"   入场时间: {now_str}")
                            notify(f"   指标: {reason}")
                            notify(f"   到期时间: {expire_str}")
                            notify(f"   预测: 10分钟后价格{dir_pred}")
                            
                            add_prediction(direction, reason, rt_price)
            
            # 验证
            verified, stats = verify_predictions()
            
            # 每5分钟输出心跳
            if tick % 300 == 0 and stats['total'] > 0:
                acc = stats['wins'] / stats['total'] * 100
                pending = len([p for p in load_predictions() if not p['verified']])
                notify(f"💓 心跳 | 总计{stats['total']}次 胜率{acc:.1f}% | 连中{stats['current_streak_win']} 连挂{stats['current_streak_loss']} | 待验证{pending}")
        
        except KeyboardInterrupt:
            notify("🛑 收到退出信号")
            break
        except Exception as e:
            notify(f"💥 异常: {e}")
            time.sleep(5)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    run_daemon()

# ═══════════ PID防重复 ═══════════

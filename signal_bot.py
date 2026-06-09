#!/usr/bin/env python3
"""⭐ 星子 v2 — BTC 10分钟方向预测 守护进程"""
import json, os, time, sys, signal as sig
import requests
import numpy as np
from datetime import datetime, timezone

DATA_FILE   = "/root/.openclaw/signal_data.json"
PRED_FILE   = "/root/.openclaw/predictions.json"
STATS_FILE  = "/root/.openclaw/signal_stats.json"
NOTIFY_FILE = "/root/.openclaw/signal_notify.json"
KEEP_KLINES = 500
HB_INTERVAL = 3600  # 每小时心跳

def notify(msg):
    ts = datetime.now().strftime('%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        queue = json.load(open(NOTIFY_FILE)) if os.path.exists(NOTIFY_FILE) else []
        queue.append(line)
        json.dump(queue, open(NOTIFY_FILE,'w'), ensure_ascii=False)
    except: pass

def api_get(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout); r.raise_for_status()
        return r.json()
    except Exception as e:
        notify(f"⚠️ API: {e}"); return None

def fetch_klines(limit=100):
    d = api_get(f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit={limit}")
    return [[k[0],float(k[1]),float(k[2]),float(k[3]),float(k[4]),float(k[5])] for k in d] if d else None

def fetch_ticker():
    d = api_get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
    return float(d['price']) if d else None

def ms_now(): return int(time.time()*1000)
def ts_str(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
def ts_cn(ms): return datetime.fromtimestamp(ms/1000,tz=timezone.utc).strftime('%m-%d %H:%M:%S')

# K线
def load_klines():
    return json.load(open(DATA_FILE)) if os.path.exists(DATA_FILE) else []
def save_klines(k):
    if len(k)>KEEP_KLINES: k=k[-KEEP_KLINES:]
    json.dump(k, open(DATA_FILE,'w'))

# 指标
def calc_indicators(kl):
    n=len(kl)
    if n<31: return None
    c=np.array([k[4] for k in kl]); h=np.array([k[2] for k in kl]); l=np.array([k[3] for k in kl]); v=np.array([k[5] for k in kl])
    a=(h[-20:]-l[-20:]).mean()
    if a==0: return None
    ma=c[-20:].mean(); z=(c[-1]-ma)/a
    d30=np.diff(c[-31:]); g=np.clip(d30,0,None).mean(); l0=-np.clip(d30,None,0).mean()
    r30=100 if l0<1e-10 else 100-100/(1+g/l0)
    d5=np.diff(c[-6:]); g5=np.clip(d5,0,None).mean(); l5=-np.clip(d5,None,0).mean()
    r5=100 if l5<1e-10 else 100-100/(1+g5/l5)
    vr=v[-1]/v[-20:].mean() if v[-20:].mean()>0 else 0
    return {'close':round(c[-1],2),'z':round(z,2),'rsi30':round(r30,1),'rsi5':round(r5,1),'v_ratio':round(vr,2),'ts':kl[-1][0]}

def check_signal(ind):
    if not ind or ind['v_ratio']<=2.0: return None,None
    if ind['z']<-2.0 and ind['rsi30']<25 and ind['rsi5']<22:
        return 'LONG',f"Z={ind['z']} RSI30={ind['rsi30']} RSI5={ind['rsi5']} 量比={ind['v_ratio']}x"
    if ind['z']>2.0 and ind['rsi30']>75 and ind['rsi5']>78:
        return 'SHORT',f"Z={ind['z']} RSI30={ind['rsi30']} RSI5={ind['rsi5']} 量比={ind['v_ratio']}x"
    return None,None

# 预测
def load_preds():
    return json.load(open(PRED_FILE)) if os.path.exists(PRED_FILE) else []
def save_preds(p): json.dump(p, open(PRED_FILE,'w'), indent=2, ensure_ascii=False)
def load_stats():
    if os.path.exists(STATS_FILE):
        return json.load(open(STATS_FILE))
    return {'total':0,'wins':0,'losses':0,'current_streak_win':0,'current_streak_loss':0,'max_win':0,'max_loss':0,'history':[]}
def save_stats(s): json.dump(s, open(STATS_FILE,'w'), indent=2, ensure_ascii=False)

def add_prediction(direction, reason, rt_price):
    now=ms_now()
    p=load_preds()
    p.append({'direction':direction,'entry_price':rt_price,'entry_ts':now,'entry_time':ts_str(now),'verify_ts':now+600000,'reason':reason,'verified':False})
    save_preds(p)

# ── 通知模板 ──
def notify_signal(direction, price, reason, entry_ts):
    dc='做多' if direction=='LONG' else '做空'
    di='📈' if direction=='LONG' else '📉'
    for l in [
        f"🔴 信号触发 — {di} {dc}",
        f"",
        f"入场时间: {ts_cn(entry_ts)}",
        f"入场价格: {price}",
        f"指标: {reason}",
        f"到期时间: {ts_cn(entry_ts+600000)}",
        f"预测: 10分钟后 {'上涨' if direction=='LONG' else '下跌'}",
        f"策略: V2.0x+R5<22",
    ]: notify(l)

def notify_verify(pred, exit_p, stats):
    correct=(exit_p>pred['entry_price'])==(pred['direction']=='LONG')
    chg=exit_p-pred['entry_price']; chp=(chg/pred['entry_price'])*100
    acc=stats['wins']/stats['total']*100 if stats['total']>0 else 0
    for l in [
        f"{'✅' if correct else '❌'} 验证结果 — {'正确' if correct else '错误'} | {'做多' if pred['direction']=='LONG' else '做空'}",
        f"",
        f"入场: {pred['entry_time']} | 价格 {round(pred['entry_price'],2)}",
        f"验证: {ts_str(ms_now())} | 价格 {round(exit_p,2)}",
        f"变动: {chg:+.2f} ({chp:+.3f}%) {'🔺' if chg>0 else '🔻' if chg<0 else '➡️'}",
        f"",
        f"📊 {stats['total']}次 | 胜率 {acc:.1f}% ({stats['wins']}/{stats['total']})",
        f"连中 {stats['current_streak_win']} 连挂 {stats['current_streak_loss']}",
        f"最长连胜 {stats['max_win']} 最长连挂 {stats['max_loss']}",
    ]: notify(l)

def verify_predictions():
    preds=load_preds(); stats=load_stats(); now=ms_now()
    new=[]
    for p in preds:
        if p['verified']: new.append(p); continue
        if now>=p['verify_ts']:
            ep=fetch_ticker()
            if not ep: new.append(p); continue
            correct=(ep>p['entry_price'])==(p['direction']=='LONG')
            stats['total']+=1
            if correct:
                stats['wins']+=1; stats['current_streak_win']+=1; stats['current_streak_loss']=0
                stats['max_win']=max(stats['max_win'],stats['current_streak_win'])
            else:
                stats['losses']+=1; stats['current_streak_loss']+=1; stats['current_streak_win']=0
                stats['max_loss']=max(stats['max_loss'],stats['current_streak_loss'])
            p['verified']=True; p['correct']=correct; p['exit_price']=ep; p['verify_time']=ts_str(ms_now())
            stats.setdefault('history',[]).append({
                'direction':p['direction'],'entry_price':p['entry_price'],
                'entry_time':p['entry_time'],'exit_price':ep,
                'verify_time':p['verify_time'],'correct':correct
            })
            notify_verify(p,ep,stats)
        else: new.append(p)
    exp=0
    for p in list(new):
        if not p['verified'] and now>p['verify_ts']+900000:
            p['verified']=True; p['correct']=None; p['exit_price']=None; exp+=1
    if exp: notify(f"⏰ {exp}条预测过期")
    save_preds(new); save_stats(stats)

# 主循环
def run():
    notify("⭐ 星子 v2 启动 — V2.0x+R5<22 每秒扫描")
    kk=load_klines()
    if not kk:
        notify("📡 拉取K线..."); kk=fetch_klines(100) or []
        if kk: save_klines(kk)
    if len(kk)<31: notify(f"⚠️ K线不足({len(kk)}根)")
    
    last_ts=kk[-1][0] if kk else 0; tick=0
    
    while True:
        try:
            time.sleep(1); tick+=1
            
            nk=fetch_klines(2)
            if not nk: continue
            
            lt=nk[-1][0]
            if lt!=last_ts:
                last_ts=lt
                seen={k[0] for k in kk}
                for k in nk:
                    if k[0] not in seen: kk.append(k); seen.add(k[0])
                kk.sort(key=lambda x:x[0])
                save_klines(kk)
                
                ind=calc_indicators(kk)
                if ind:
                    d,r=check_signal(ind)
                    if d:
                        rp=fetch_ticker()
                        if rp:
                            notify_signal(d,rp,r,ms_now())
                            add_prediction(d,r,rp)
            
            verify_predictions()
            
            # 心跳: 每小时一次
            if tick%HB_INTERVAL==0:
                s=load_stats()
                if s['total']>0:
                    notify(f"💓 心跳 | {s['total']}次 胜率{s['wins']/s['total']*100:.1f}% | 连中{s['current_streak_win']}连挂{s['current_streak_loss']}")
                # 无信号时不发心跳
            
        except KeyboardInterrupt: notify("🛑 退出"); break
        except Exception as e: notify(f"💥 {e}"); time.sleep(5)

if __name__=='__main__':
    sig.signal(sig.SIGTERM, lambda *_: sys.exit(0))
    run()

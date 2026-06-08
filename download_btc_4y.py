#!/usr/bin/env python3
"""币安 BTCUSDT 1分钟K线 4年数据下载 (2022-06 → 2026-06)"""
import requests, json, time, os

SYMBOL="BTCUSDT"; INTERVAL="1m"; LIMIT=1000
# 2022-06-09 00:00 UTC → 2026-06-09 00:00 UTC
START=1654732800000
END=1780963200000
OUT="/root/.openclaw/btc_1m_4year.json"

# 断点续传: 已有数据跳过
existing=[]
if os.path.exists(OUT):
    with open(OUT) as f:
        try: existing=json.load(f)
        except: pass
if existing:
    existing.sort(key=lambda x:x[0])
    last=existing[-1][0]
    print(f"📂 已有 {len(existing):,}根, 续传自 {time.strftime('%Y-%m-%d %H:%M',time.gmtime(last/1000))}")
    current=last+60000
else:
    current=START

all_klines=[]; req=0
total=END-START
print(f"📥 BTC 1m K线 4年 (~2100批次)",flush=True)

while current<END:
    url=f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}&startTime={current}"
    try:
        r=requests.get(url,timeout=30); r.raise_for_status(); data=r.json()
    except Exception as e:
        print(f"  ⚠️ {e}"); time.sleep(5); continue
    if not data or not isinstance(data,list) or len(data)==0: break
    req+=1; all_klines.extend(data)
    if req%50==0:
        t=time.strftime('%Y-%m-%d %H:%M',time.gmtime(data[-1][0]/1000))
        pct=(data[-1][0]-START)/total*100
        print(f"  [{req:4d}] {t} 累计{len(all_klines)+len(existing):>10,}根 ({min(pct,100):.0f}%)",flush=True)
    if len(data)<LIMIT: break
    current=data[-1][0]+60000
    time.sleep(0.15)

# 合并
rows=[]
for k in all_klines:
    rows.append([k[0],float(k[1]),float(k[2]),float(k[3]),float(k[4]),float(k[5])])
rows.extend(existing)
seen=set(); uniq=[]
for r in rows:
    if r[0] not in seen: seen.add(r[0]); uniq.append(r)
uniq.sort(key=lambda x:x[0])

with open(OUT,'w') as f: json.dump(uniq,f)
sz=os.path.getsize(OUT)
print(f"\n✅ {len(uniq):,}根 ({sz/1024/1024:.1f}MB)")
print(f"   {time.strftime('%Y-%m-%d %H:%M',time.gmtime(uniq[0][0]/1000))} → {time.strftime('%Y-%m-%d %H:%M',time.gmtime(uniq[-1][0]/1000))}")

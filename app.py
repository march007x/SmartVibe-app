import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.signal import welch
from streamlit_autorefresh import st_autorefresh

# ==========================================================
# ⚙️ ส่วนตั้งค่าโปรเจกต์ และ Telegram Bot
# ==========================================================
FIREBASE_URL = 'https://smartvibee-22adf-default-rtdb.asia-southeast1.firebasedatabase.app/SmartVibe/History3F.json'
STATE_URL = 'https://smartvibee-22adf-default-rtdb.asia-southeast1.firebasedatabase.app/SmartVibe/State3F.json'

TELEGRAM_BOT_TOKEN = "8816324739:AAHZEKbjTyvLUORVd97t5kzFWy7pIxqFEhY"
TELEGRAM_CHAT_ID = "7360818672"

st.set_page_config(page_title="SmartVibe Layer Analysis", layout="wide")
st.title("SmartVibe: ระบบวิเคราะห์ความสั่นสะเทือนแยก")

st_autorefresh(interval=850, limit=None, key="smartvibe_autorefresh")

QUERY = '?orderBy="$key"&limitToLast=500'
STATE_QUERY = ''

NOMINAL_FS = 50.0
FORCING_FREQ = 8.5
BAND_HZ = 1.5
HISTORY_SIZE = 7
MIN_CONSEC = 2

# ===== Session state =====
if 'http_session' not in st.session_state: st.session_state.http_session = requests.Session()
if 'last_uptime' not in st.session_state: st.session_state.last_uptime = 0
if 'stuck_counter' not in st.session_state: st.session_state.stuck_counter = 0
if 'prev_status' not in st.session_state: st.session_state.prev_status = {0: 'green', 1: 'green', 2: 'green'}

for i in range(3):
    if f'base_amp{i}' not in st.session_state: st.session_state[f'base_amp{i}'] = None
    if f'history_a{i}' not in st.session_state: st.session_state[f'history_a{i}'] = []
    if f'rms_ch{i}' not in st.session_state: st.session_state[f'rms_ch{i}'] = 0.0
    if f'status{i}' not in st.session_state: st.session_state[f'status{i}'] = 'green'
    if f'consec{i}' not in st.session_state: st.session_state[f'consec{i}'] = 0
    if f'consec_dir{i}' not in st.session_state: st.session_state[f'consec_dir{i}'] = None

# ===== Sidebar =====
with st.sidebar:
    st.header("⚙️ ปรับ Threshold")
    G2Y = st.slider("🟢→🟡", 50, 99, 80, 1)
    Y2R = st.slider("🟡→🔴", 50, 99, 65, 1)
    Y2G = st.slider("🟡→🟢", 50, 99, 87, 1)
    R2Y = st.slider("🔴→🟡", 50, 99, 70, 1)

# ===== Functions =====
def send_telegram_notification(message):
    if not TELEGRAM_BOT_TOKEN or "ใส่_" in TELEGRAM_BOT_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: st.session_state.http_session.post(url, json=payload, timeout=3)
    except: pass

def fetch_data():
    try:
        res = st.session_state.http_session.get(FIREBASE_URL + QUERY, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if not data: return pd.DataFrame()
            flat = {}
            for k, v in data.items():
                if not isinstance(v, dict): continue
                if 'uptime_ms' in v: flat[k] = v
                else:
                    for sk, sv in v.items():
                        if isinstance(sv, dict) and 'uptime_ms' in sv: flat[sk] = sv
            if not flat: return pd.DataFrame()
            df = pd.DataFrame.from_dict(flat, orient='index')
            df['uptime_ms'] = pd.to_numeric(df['uptime_ms'], errors='coerce')
            return df.dropna(subset=['uptime_ms']).sort_values('uptime_ms').reset_index(drop=True)
    except: pass
    return pd.DataFrame()

def get_band_power(df, col, ch_idx, is_new_data):
    sig = df[col].values.astype(float)
    sig = sig - np.mean(sig)
    st.session_state[f'rms_ch{ch_idx}'] = float(np.sqrt(np.mean(sig**2)))
    fw, psd = welch(sig, fs=NOMINAL_FS, nperseg=min(256, len(sig)//2), window='hann')
    mask = (fw >= FORCING_FREQ - BAND_HZ) & (fw <= FORCING_FREQ + BAND_HZ)
    band_power = float(np.sum(psd[mask])) if mask.any() else 0.0
    hist = st.session_state[f'history_a{ch_idx}']
    if is_new_data:
        hist.append(band_power)
        if len(hist) > HISTORY_SIZE: hist.pop(0)
        st.session_state[f'history_a{ch_idx}'] = hist
    return float(np.median(hist)) if hist else band_power

def compute_health(amps):
    bases = [st.session_state[f'base_amp{i}'] for i in range(3)]
    if any(b is None for b in bases): return [None]*3
    return [min(amps[i]/bases[i]*100, 100.0) if bases[i] > 0 else 0.0 for i in range(3)]

def update_status(pct, ch_idx, is_new_data, floor_name):
    s = st.session_state[f'status{ch_idx}']
    c = st.session_state[f'consec{ch_idx}']
    if not is_new_data: return s, c
    new_s = s
    if s == 'green':
        c = c+1 if pct < G2Y else 0
        if c >= MIN_CONSEC: new_s, c = 'yellow', 0
    elif s == 'yellow':
        cur_dir = 'up' if pct >= Y2G else ('down' if pct < Y2R else None)
        prev_dir = st.session_state[f'consec_dir{ch_idx}']
        if cur_dir != prev_dir: c = 0
        st.session_state[f'consec_dir{ch_idx}'] = cur_dir
        if cur_dir is not None:
            c += 1
            if c >= MIN_CONSEC: new_s, c = ('green' if cur_dir == 'up' else 'red'), 0
        else: c = 0
    elif s == 'red':
        c = c+1 if pct >= R2Y else 0
        if c >= MIN_CONSEC: new_s, c = 'yellow', 0
    
    if new_s != st.session_state.prev_status[ch_idx]:
        send_telegram_notification(f"🔔 *SmartVibe Alert*\n📍 {floor_name}\nสถานะเปลี่ยน: {st.session_state.prev_status[ch_idx]} ➡️ {new_s}")
        st.session_state.prev_status[ch_idx] = new_s
    st.session_state[f'status{ch_idx}'] = new_s
    st.session_state[f'consec{ch_idx}'] = c
    return new_s, c

# ==========================================
# Main Execution
# ==========================================
df = fetch_data()

# ปรับปรุง Logic การเช็คข้อมูลเพื่อไม่ให้หน้าจอว่าง
if df.empty:
    st.warning("⚠️ ยังไม่มีข้อมูล หรือดึงจาก Firebase ไม่ได้")
    st.info("ตรวจสอบว่า Firebase Rules เป็น public (read: true) และมีข้อมูลส่งเข้าไปที่ Path: /SmartVibe/History3F")
elif len(df) <= 50:
    st.info(f"⏳ กำลังรอข้อมูลสะสม (ข้อมูลปัจจุบัน: {len(df)} รายการ / ต้องการ 50 รายการ)")
    st.progress(len(df)/50)
else:
    # --- เริ่มวิเคราะห์เมื่อข้อมูลพร้อม ---
    cur = df['uptime_ms'].iloc[-1]
    is_new_data = (cur != st.session_state.last_uptime)
    st.session_state.last_uptime = cur
    
    amps = [get_band_power(df, f'AccX_CH{i}', i, is_new_data) for i in range(3)]
    health = compute_health(amps)
    floor_names = ["ชั้น 1", "ชั้น 2", "ชั้น 3"]

    c1, c2 = st.columns(2)
    if c1.button("🔒 ล็อก Baseline"):
        for i in range(3): st.session_state[f'base_amp{i}'] = amps[i]
        st.rerun()
    if c2.button("ล้างค่าทั้งหมด"):
        for i in range(3): st.session_state[f'base_amp{i}'] = None
        st.rerun()

    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            st.subheader(floor_names[i])
            if st.session_state[f'base_amp{i}']:
                pct = health[i]
                status, _ = update_status(pct, i, is_new_data, floor_names[i])
                st.metric("Health %", f"{pct:.1f}%")
                st.progress(min(int(pct), 100))
                st.write(f"Status: {status}")
            else:
                st.info("ยังไม่ได้ล็อก Baseline")

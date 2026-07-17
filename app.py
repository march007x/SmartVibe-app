import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.signal import welch
from streamlit_autorefresh import st_autorefresh
import time

# ==========================================================
# ⚙️ ส่วนตั้งค่าโปรเจกต์ และ Telegram Bot (อัปเดตเซิร์ฟเวอร์ใหม่แล้ว)
# ==========================================================
FIREBASE_URL = 'https://smart-vibe-f944b-default-rtdb.asia-southeast1.firebasedatabase.app/SmartVibe/History3F.json'
STATE_URL = 'https://smart-vibe-f944b-default-rtdb.asia-southeast1.firebasedatabase.app/SmartVibe/State3F.json'

# --- ใส่ Token และ Chat ID ของคุณตรงนี้ เพื่อเปิดระบบแจ้งเตือนภัย ---
TELEGRAM_BOT_TOKEN = "ใส่_TELEGRAM_BOT_TOKEN_ที่ได้จาก_BotFather"
TELEGRAM_CHAT_ID = "ใส่_CHAT_ID_ของคุณ"
# ==========================================================

st.set_page_config(page_title="SmartVibe Layer Analysis", layout="wide")
st.title("SmartVibe: ระบบวิเคราะห์ความสั่นสะเทือนแยกชั้นอาคาร")

# รีเฟรชหน้าจอดึงข้อมูลทุก 850 มิลลิวินาที
st_autorefresh(interval=850, limit=None, key="smartvibe_autorefresh")

# คิวรีดึงข้อมูลในระบบเปิด (Test Mode) โดยไม่ต้องแนบ Token ท้ายประโยค
QUERY = '?orderBy="$key"&limitToLast=500'
STATE_QUERY = ''
NOMINAL_FS = 50.0
FORCING_FREQ = 8.5
BAND_HZ = 1.5
HISTORY_SIZE = 7
MIN_CONSEC = 2

# ===== เปิด Session state เก็บค่าคงที่ภายในคลาสแอปพลิเคชัน =====
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

# ===== แผงควบคุมด้านข้าง (Sidebar) =====
with st.sidebar:
    st.header("⚙️ ปรับระดับเกณฑ์ความไว (Threshold %)")
    G2Y = st.slider("🟢 → 🟡 (เริ่มเฝ้าระวังเมื่อสุขภาพตึกต่ำกว่า)", 50, 99, 80, 1)
    Y2R = st.slider("🟡 → 🔴 (อันตรายร้ายแรงเมื่อสุขภาพตึกต่ำกว่า)", 50, 99, 65, 1)
    Y2G = st.slider("🟡 → 🟢 (ฟื้นตัวกลับสู่ปกติเมื่อสุขภาพตึกสูงกว่า)", 50, 99, 87, 1)
    R2Y = st.slider("🔴 → 🟡 (บรรเทาจากอันตรายมาเฝ้าระวังเมื่อสุขภาพดีกว่า)", 50, 99, 70, 1)

def send_telegram_notification(message):
    """ส่งข้อความแจ้งเตือนผ่าน Telegram API แบบเรียลไทม์"""
    if not TELEGRAM_BOT_TOKEN or "ใส่_" in TELEGRAM_BOT_TOKEN:
        return 
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        st.session_state.http_session.post(url, json=payload, timeout=3)
    except Exception as e:
        st.sidebar.warning(f"ระบบส่ง Telegram ขัดข้อง: {e}")

def fetch_data():
    """ดึงข้อมูลโครงสร้างดิบย้อนหลังจากคลาวด์ Firebase"""
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
                        if isinstance(sv, dict) and 'uptime_ms' in sv:
                            flat[sk] = sv
            if not flat: return pd.DataFrame()
            df = pd.DataFrame.from_dict(flat, orient='index')
            
            # แปลงค่าคอลัมน์สำคัญให้เป็นตัวเลขและกรองข้อมูลเสียทิ้งป้องกันกราฟล่ม
            target_cols = ['uptime_ms', 'AccX_CH0', 'AccX_CH1', 'AccX_CH2']
            for col in target_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['uptime_ms'])
            # แทนที่ข้อมูลช่องว่างที่ขาดหายไประหว่างบอร์ดส่งข้อมูล (Data Interpolation)
            df = df.ffill().bfill()
            return df.sort_values('uptime_ms').reset_index(drop=True)
    except Exception as e:
        st.sidebar.error(f"การดึงข้อมูลผิดพลาด: {e}")
    return pd.DataFrame()

def push_baseline_to_firebase(amps):
    """บันทึกข้อมูลค่าความสั่นสะเทือนอ้างอิงเริ่มต้นขึ้นสู่เซิร์ฟเวอร์"""
    payload = {f"base_amp{i}": amps[i] for i in range(3)}
    try:
        res = st.session_state.http_session.patch(STATE_URL + STATE_QUERY, json=payload, timeout=3)
        return res.status_code == 200
    except Exception:
        return False

def fetch_remote_state():
    """ดึงข้อมูลบันทึกสเตตัสจำลองในฝั่งเซิร์ฟเวอร์มามอนิเตอร์"""
    try:
        res = st.session_state.http_session.get(STATE_URL + STATE_QUERY, timeout=3)
        if res.status_code == 200: return res.json() or {}
    except Exception: pass
    return {}

def get_band_power(df, col, ch_idx, is_new_data):
    """คำนวณสเปกตรัมความถี่คลื่นความเร็วสูงเฉพาะกลุ่มความถี่บังคับ (Forcing Frequency)"""
    if col not in df.columns or len(df) < 10:
        return 0.0
        
    sig = df[col].values.astype(float)
    sig = sig - np.mean(sig) # กำจัดส่วนประกอบไฟฟ้ากระแสตรง (DC Offset)
    st.session_state[f'rms_ch{ch_idx}'] = float(np.sqrt(np.mean(sig**2)))
    
    # คำนวณหาความหนาแน่นเชิงสเปกตรัมของกำลัง (PSD)
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
    """วิเคราะห์เปอร์เซ็นต์ความแข็งแรงสุขภาพของตึกเทียบกับฐานข้อมูลเดิม (Baseline)"""
    bases = [st.session_state[f'base_amp{i}'] for i in range(3)]
    if any(b is None for b in bases): return [None]*3
    # ยิ่งค่าสั่นสะเทือน (Amplitude) สูงกว่าฐานตั้งต้น เปอร์เซ็นต์สุขภาพตึกจะยิ่งลดลงลงต่ำกว่า 100%
    return [min(max(amps[i]/bases[i]*100, 0.0), 100.0) if bases[i] > 0 else 0.0 for i in range(3)]

def update_status(pct, ch_idx, is_new_data, floor_name):
    """สลับโหมดควบคุมตรรกะสถานะความปลอดภัยของตึกพร้อมเชื่อมระบบ Telegram แจ้งเตือนภัย"""
    s = st.session_state[f'status{ch_idx}']
    c = st.session_state[f'consec{ch_idx}']
    
    if not is_new_data:
        return s, c
        
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
            if c >= MIN_CONSEC:
                new_s = 'green' if cur_dir == 'up' else 'red'
                c = 0
        else:
            c = 0
    elif s == 'red':
        c = c+1 if pct >= R2Y else 0
        if c >= MIN_CONSEC: new_s, c = 'yellow', 0

    # 🔔 หากมีการเปลี่ยนแปลงระดับสถานะอาคาร ให้ส่งสัญญาณเตือนภัยเข้ามือถือทันที
    if new_s != st.session_state.prev_status[ch_idx]:
        status_emojis = {'green': '🟢 ปลอดภัย/ปกติ', 'yellow': '⚠️ เฝ้าระวังใกล้ชิด', 'red': '🚨 โครงสร้างอันตราย!'}
        old_status_text = status_emojis.get(st.session_state.prev_status[ch_idx], st.session_state.prev_status[ch_idx])
        new_status_text = status_emojis.get(new_s, new_s)
        
        msg = f"🔔 *[SmartVibe SHM Alert]*\n📍 ตรวจพบความผิดปกติที่: *{floor_name}*\n"
        msg += f"🔄 ระดับการสั่นเปลี่ยนจาก: {old_status_text} ➡️ *{new_status_text}*\n"
        msg += f"📊 ดัชนีสุขภาพอาคาร (Health Index): `{pct:.1f}%`"
        
        send_telegram_notification(msg)
        st.session_state.prev_status[ch_idx] = new_s

    st.session_state[f'status{ch_idx}'] = new_s
    st.session_state[f'consec{ch_idx}'] = c
    return new_s, c

def get_fft_graph_data(df):
    """สกัดวิเคราะห์ข้อมูลการแปลงฟาสต์ฟูริเยร์ (Fast Fourier Transform)"""
    result_freqs, result_psds = None, []
    for col in ['AccX_CH0', 'AccX_CH1', 'AccX_CH2']:
        if col not in df.columns or len(df) < 50: return None, None, None, None
        sig = df[col].values.astype(float) - df[col].mean()
        fw, psd = welch(sig, fs=NOMINAL_FS, nperseg=min(256, len(sig)//2), window='hann')
        valid = fw >= 0.5
        if result_freqs is None: result_freqs = fw[valid]
        result_psds.append(psd[valid])
    return result_freqs, result_psds[0], result_psds[1], result_psds[2]

# ==========================================
# ส่วนเริ่มต้นการรันอินเตอร์เฟสหลักบนหน้าเว็บ
# ==========================================
df = fetch_data()

if not df.empty and len(df) > 50:
    cur = df['uptime_ms'].iloc[-1]
    is_new_data = (cur != st.session_state.last_uptime)
    
    if is_new_data:
        st.session_state.stuck_counter = 0
        st.session_state.last_uptime = cur
    else:
        st.session_state.stuck_counter += 1
        
    if st.session_state.stuck_counter >= 10:
        st.error("🚨 สัญญาณขัดข้อง: ข้อมูลหยุกนิ่งไม่มีการขยับ! กรุณาเช็กไฟเลี้ยงบอร์ด ESP32 หรือการเชื่อมต่ออินเทอร์เน็ตหน้างาน")

    amps = [get_band_power(df, f'AccX_CH{i}', i, is_new_data) for i in range(3)]
    health = compute_health(amps)
    floor_names = ["ชั้น 1 (โครงสร้างรากฐาน)", "ชั้น 2 (กลางอาคาร)", "ชั้น 3 (ดาดฟ้ายอดตึก)"]

    st.info(f"🔊 ความถี่สั่นสะเทือนไฟต์บังคับ (Forcing Frequency): **{FORCING_FREQ} Hz** (แถบควบคุม ±{BAND_HZ} Hz)")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔒 ล็อกสภาวะอ้างอิง Baseline (ตึกนิ่ง + น็อตเซ็นเซอร์แน่นหนา)", type="primary", key="btn_lock"):
            for i in range(3):
                st.session_state[f'base_amp{i}'] = amps[i]
                st.session_state[f'status{i}'] = 'green'
                st.session_state[f'consec{i}'] = 0
                st.session_state[f'consec_dir{i}'] = None
                st.session_state.prev_status[i] = 'green' 
            ok = push_baseline_to_firebase(amps)
            if ok: st.success("✅ ล็อกค่า Baseline อ้างอิงปัจจุบันขึ้นสู่ระบบฐานข้อมูลคลาวด์เรียบร้อยแล้ว")
            st.rerun()
    with c2:
        if st.button("🔄 รีเซ็ตล้างค่าแอปพลิเคชันทั้งหมด", key="btn_reset"):
            for i in range(3):
                st.session_state[f'base_amp{i}'] = None
                st.session_state[f'history_a{i}'] = []
                st.session_state[f'status{i}'] = 'green'
                st.session_state[f'consec{i}'] = 0
                st.session_state[f'consec_dir{i}'] = None
                st.session_state.prev_status[i] = 'green'
            st.rerun()

    st.markdown("---")
    cols = st.columns(3)

    for i in range(3):
        with cols[i]:
            st.subheader(floor_names[i])
            rms_now = st.session_state[f'rms_ch{i}']
            hist = st.session_state[f'history_a{i}']
            base = st.session_state[f'base_amp{i}']

            st.markdown(f"ความเร่งเฉลี่ยรากกำลังสอง RMS: `{rms_now:.4f} g`")
            # ป้องกันค่าติดลบหรือเกินร้อยในเปอร์เซ็นต์บาร์แสดงผล
            st.progress(min(max(int(rms_now / 0.15 * 100), 0), 100))

            if base and base > 0:
                delta_pct = (amps[i] - base) / base * 100
                st.metric(f"กำลังในแถบความถี่ ({FORCING_FREQ}±{BAND_HZ} Hz)", f"{amps[i]:.5f}", delta=f"{delta_pct:+.1f}% เทียบตัวแปรตั้งต้น")
            else:
                st.metric(f"กำลังในแถบความถี่ ({FORCING_FREQ}±{BAND_HZ} Hz)", f"{amps[i]:.5f}")

            if len(hist) >= 3:
                cv = np.std(hist)/np.mean(hist)*100 if np.mean(hist) > 0 else 0
                st.caption(f"จำนวนการอ่านสะสม: {len(hist)}/{HISTORY_SIZE} | สัมประสิทธิ์ความแปรปรวน CV={cv:.1f}% {'✅ สัญญาณนิ่งเสถียร' if cv < 15 else '⚠️ สัญญาณแกว่งตัวสูง'}")

            if base and base > 0 and health[i] is not None:
                pct = health[i]
                status, cnt = update_status(pct, i, is_new_data, floor_names[i]) 
                st.metric("ดัชนีความแข็งแรงสุขภาพชั้นอาคาร", f"{pct:.1f}%")
                st.progress(min(max(int(pct), 0), 100))

                if status == 'green': st.success(f"🟢 ปลอดภัย: {pct:.1f}%")
                elif status == 'yellow': st.warning(f"🟡 เฝ้าระวัง: {pct:.1f}% [{cnt}/{MIN_CONSEC}]")
                else: st.error(f"🔴 อันตรายรุนแรง: {pct:.1f}% [{cnt}/{MIN_CONSEC}]")
            else:
                st.info("ℹ️ กรุณากดปุ่ม 🔒 ล็อกสภาวะอ้างอิง Baseline เพื่อเริ่มต้นคำนวณเปอร์เซ็นต์สุขภาพโครงสร้างอาคาร")

    st.markdown("---")
    st.subheader("📊 กราฟวิเคราะห์ความหนาแน่นสเปกตรัมกำลังสั่นสะเทือน (FFT / PSD Graph)")
    result = get_fft_graph_data(df)
    if result[0] is not None:
        xf, m0, m1, m2 = result
        chart_df = pd.DataFrame({"ชั้น 1 (ฐาน)": m0, "ชั้น 2 (กลาง)": m1, "ชั้น 3 (ยอด)": m2}, index=xf)
        # ตีกรอบพล็อตความถี่เฉพาะช่วงต่ำกว่า 20Hz ที่เป็นย่านหลักของการเคลื่อนไหวอาคารตึกสั่น
        st.line_chart(chart_df[chart_df.index <= 20], x_label="ความถี่โครงสร้าง (Frequency - Hz)", y_label="ความหนาแน่นเชิงสเปกตรัมกำลัง (PSD)")

        with st.expander("ℹ️ ข้อมูลการดีบักระบบสตรีมมิ่งความเร็วสูง"):
            dts = df['uptime_ms'].diff().dropna()
            nd = dts[(dts >= 15) & (dts <= 40)]
            st.write("ช่วงเวลาห่างของการส่งข้อมูล (Delta-Uptime) ในหน่วยมิลลิวินาที:", nd.describe())

    st.markdown("---")
    with st.expander("🤖 ตรวจสอบค่าสถานะบันทึกบน Cloud Node (Firebase Remote State)"):
        remote_state = fetch_remote_state()
        if not remote_state:
            st.caption("ยังไม่มีข้อมูลโครงสร้างใดๆ บันทึกในโหนดควบคุมระยะไกลบนคลาวด์")
        else:
            cols2 = st.columns(3)
            for i in range(3):
                with cols2[i]:
                    st.caption(floor_names[i])
                    st.write(f"สถานะที่จำลองไว้: `{remote_state.get(f'status{i}', '-')}`")
                    st.write(f"ค่าแอมพลิจูดอ้างอิงเริ่มต้น: `{remote_state.get(f'base_amp{i}', '-')}`")
else:
    st.info("⏳ กำลังรอเชื่อมต่อรับส่งสัญญาณข้อมูลแบบกระจายความถี่ (Fast Streaming Batch) ก้อนแรกจากบอร์ด ESP32 ผ่านโครงสร้างเซิร์ฟเวอร์ใหม่... กรุณาเปิด Serial Monitor เพื่อตรวจสอบสัญญาณฮาร์ดแวร์คู่ขนานกันครับ")

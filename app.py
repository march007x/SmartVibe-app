import streamlit as pd_st
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from scipy.signal import welch
import plotly.graph_objects as go

# 1. ตั้งค่าการเชื่อมต่อ คลาวด์ และบอทแจ้งเตือน (แก้ค่าโทเคนของคุณที่นี่)
FIREBASE_URL = "https://smartvibe-2768d-default-rtdb.asia-southeast1.firebasedatabase.app/SmartVibe/History3F.json"
AUTH_TOKEN = "bmVF3XzxEMSVzX8oYMGpN9NxG4TbohM3xxnWFtbO"
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # ใส่ Token ของคุณ
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"      # ใส่ Chat ID ของคุณ

# ตั้งค่าหน้าเว็บหน้าตาแอปพลิเคชัน
st.set_page_config(page_title="Structural Health Monitoring Dashboard", layout="wide")
st.title("🏢 ระบบตรวจวัดพฤติกรรมความสั่นสะเทือนโครงสร้างอาคาร 3 ชั้น (SHM)")

# สร้าง HTTP Session ทิ้งไว้เพื่อลดการโหลดเปิด-ปิดพอร์ตเชื่อมต่อซ้ำๆ
if 'http_session' not in st.session_state:
    st.session_state.http_session = requests.Session()
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0

# --- แถบตั้งค่าควบคุมขวา/ซ้าย (Sidebar) ---
st.sidebar.header("⚙️ การตั้งค่าระบบ")
sampling_rate = st.sidebar.number_input("ความถี่ในการเก็บข้อมูล (Hz)", value=50, disabled=True)
alert_threshold = st.sidebar.slider("เกณฑ์แจ้งเตือนความเร่งวิกฤต (g)", 0.1, 2.0, 0.5, step=0.05)

st.sidebar.subheader("📊 พารามิเตอร์ Welch's Method")
nperseg_val = st.sidebar.selectbox("ความยาวหน้าต่างสัญญาณ (Nperseg)", [64, 128, 256], index=1)
overlap_val = st.sidebar.slider("เปอร์เซ็นต์การซ้อนทับ (Overlap)", 0, 90, 50, step=10) / 100.0

# ฟังก์ชันยิงแจ้งเตือนผ่าน Telegram (พร้อมระบบป้องกันสแปม ทำงานห่างกันอย่างน้อย 5 วินาที)
def trigger_telegram_alert(floor, current_val):
    current_time = time.time()
    if current_time - st.session_state.last_alert_time > 5:
        msg = f"⚠️ แจ้งเตือนวิกฤตโครงสร้าง! พบความสั่นสะเทือนสูงเกินกำหนดที่ [{floor}] ค่าปัจจุบัน: {current_val:.3f} g (เกณฑ์ปลอดภัย: {alert_threshold} g)"
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=2)
            st.session_state.last_alert_time = current_time
        except Exception as e:
            st.sidebar.error(f"ไม่สามารถส่ง Telegram ได้: {e}")

# ฟังก์ชันดึงข้อมูลจาก Firebase Realtime Database
def fetch_data():
    try:
        query_url = f"{FIREBASE_URL}?auth={AUTH_TOKEN}&orderBy=\"$key\"&limitToLast=400"
        res = st.session_state.http_session.get(query_url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if not data: 
                return pd.DataFrame()
            
            # แปลงโครงสร้างข้อมูล JSON ให้เป็นตาราง DataFrame
            df = pd.DataFrame.from_dict(data, orient='index')
            df['uptime_ms'] = pd.to_numeric(df['uptime_ms'], errors='coerce')
            df = df.dropna(subset=['uptime_ms'])
            
            # เรียงดัชนีคีย์ข้อความ (คีย์ 0000... จะเรียงตัวสมบูรณ์แบบที่จุดนี้)
            return df.sort_index().reset_index(drop=True)
    except Exception as e:
        st.sidebar.error(f"การเชื่อมต่อคลาวด์ขัดข้อง: {e}")
    return pd.DataFrame()

# --- ส่วนหลักการประมวลผลและแสดงผล ---
df_data = fetch_data()

if not df_data.empty and len(df_data) > 10:
    # คำนวณความถี่ตัวอย่างจริงเพื่อความแม่นยำทางสถิติ
    dt_series = df_data['uptime_ms'].diff().dropna() / 1000.0
    actual_fs = 1.0 / dt_series.mean() if dt_series.mean() > 0 else 50.0

    # ตรวจสอบค่าความเร่งสูงสุดเพื่อแจ้งเตือนความปลอดภัยโครงสร้าง
    max_ch0 = max(df_data['AccZ_CH0'].abs().max(), df_data['AccX_CH0'].abs().max())
    max_ch1 = max(df_data['AccZ_CH1'].abs().max(), df_data['AccX_CH1'].abs().max())
    max_ch2 = max(df_data['AccZ_CH2'].abs().max(), df_data['AccX_CH2'].abs().max())

    # ตรวจสอบสถานะการแจ้งเตือนพังทลาย
    if max_ch2 > alert_threshold: trigger_telegram_alert("ชั้นที่ 3 (Top)", max_ch2)
    elif max_ch1 > alert_threshold: trigger_telegram_alert("ชั้นที่ 2 (Mid)", max_ch1)
    elif max_ch0 > alert_threshold: trigger_telegram_alert("ชั้นที่ 1 (Base)", max_ch0)

    # 1. แสดงกล่องสถานะค่าความเร่งสูงสุดแบบเรียลไทม์ (Real-time Metrics)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("🔺 ความเร่งสูงสุดชั้น 3 (Top)", f"{max_ch2:.3f} g", delta=f"{max_ch2-alert_threshold:.3f} g" if max_ch2 > alert_threshold else "ปกติ")
    with col2: st.metric("🏢 ความเร่งสูงสุดชั้น 2 (Mid)", f"{max_ch1:.3f} g", delta=f"{max_ch1-alert_threshold:.3f} g" if max_ch1 > alert_threshold else "ปกติ")
    with col3: st.metric("🧱 ความเร่งสูงสุดชั้น 1 (Base)", f"{max_ch0:.3f} g", delta=f"{max_ch0-alert_threshold:.3f} g" if max_ch0 > alert_threshold else "ปกติ")

    # แยกหน้าจัดสัดส่วนการพล็อตกราฟเป็น 2 แท็บข้อมูล
    tab1, tab2, tab3 = st.tabs(["📉 สัญญาณโดเมนเวลา (Time-Series)", "📊 สเปกตรัมความถี่ (Welch PSD)", "🛠️ ตัวช่วยวิเคราะห์ระบบ (Debug)"])

    with tab1:
        st.subheader("กราฟแสดงความเร่งแกนแนวตั้ง (Acc Z) ของอาคารทั้ง 3 ชั้น")
        fig_time = go.Figure()
        # แปลงแกนเวลาเป็นวินาทีสัมพัทธ์เพื่อให้ดูง่าย
        relative_time = (df_data['uptime_ms'] - df_data['uptime_ms'].iloc[0]) / 1000.0
        
        fig_time.add_trace(go.Scatter(x=relative_time, y=df_data['AccZ_CH2'], name="ชั้น 3 (Top)", line=dict(color='firebrick', width=2)))
        fig_time.add_trace(go.Scatter(x=relative_time, y=df_data['AccZ_CH1'], name="ชั้น 2 (Mid)", line=dict(color='royalblue', width=2)))
        fig_time.add_trace(go.Scatter(x=relative_time, y=df_data['AccZ_CH0'], name="ชั้น 1 (Base)", line=dict(color='forestgreen', width=1.5)))
        
        fig_time.update_layout(xaxis_title="เวลาสัมพันธ์ (วินาที)", yaxis_title="ความเร่ง (g)", margin=dict(l=20, r=20, t=20, b=20), height=400)
        st.plotly_chart(fig_time, use_container_width=True)

    with tab2:
        st.subheader("ความหนาแน่นสเปกตรัมกำลัง (Power Spectral Density - Welch Method)")
        st.caption("กราฟนี้ใช้สำหรับหาค่าความถี่ธรรมชาติ (Natural Frequency) ของตัวอาคารเพื่อประเมินความเสียหาย")
        
        fig_psd = go.Figure()
        nperseg_actual = min(len(df_data), nperseg_val)
        noverlap_actual = int(nperseg_actual * overlap_val)

        # คำนวณ Welch Method แยกรายเซ็นเซอร์ความเคลื่อนไหวหลัก
        for ch_name, label, color in [('AccZ_CH0', 'ชั้น 1 (Base)', 'forestgreen'), 
                                      ('AccZ_CH1', 'ชั้น 2 (Mid)', 'royalblue'), 
                                      ('AccZ_CH2', 'ชั้น 3 (Top)', 'firebrick')]:
            
            signal_data = df_data[ch_name].values - df_data[ch_name].mean() # ลบค่าเฉลี่ย DC Offset ออกก่อนคำนวณ
            f, Pxx = welch(signal_data, fs=actual_fs, nperseg=nperseg_actual, noverlap=noverlap_actual)
            
            # ค้นหาตำแหน่งความถี่สูงสุด (Peak Frequency) 
            peak_freq = f[np.argmax(Pxx)]
            fig_psd.add_trace(go.Scatter(x=f, y=Pxx, name=f"{label} [Peak: {peak_freq:.2f} Hz]", line=dict(color=color, width=2)))

        fig_psd.update_layout(xaxis_title="ความถี่ (Hz)", yaxis_title="กำลังสเปกตรัม (g²/Hz)", yaxis_type="log", margin=dict(l=20, r=20, t=20, b=20), height=450)
        st.plotly_chart(fig_psd, use_container_width=True)

    with tab3:
        st.subheader("🔍 ข้อมูลตรวจเช็กระบบสัญญาณเชิงลึก")
        st.write(f"ค่าความถี่สุ่มตัวอย่างตรวจวัดจริงจากบอร์ด: **{actual_fs:.2f} Hz**")
        st.write("ตารางแสดงความคลาดเคลื่อนช่วงเวลาสุ่มตัวอย่าง (Delta T) ปัจจุบัน:")
        st.line_chart(dt_series)
        if dt_series.max() > 0.04:
            st.warning("⚠️ มีสัญญาณกระตุกเล็กน้อย แต่ระบบคิว FreeRTOS จะพยายามดึงเวลากลับมาให้คงเดิมอัตโนมัติ")

else:
    st.info("⌛ กำลังรอการเชื่อมต่อสตรีมข้อมูลความเร็วสูงจากบอร์ด ESP32 หรือฐานข้อมูลว่างเปล่า...")

# สั่งให้สตรีมลิตรีเฟรชหน้าตัวเองทุกๆ 1 วินาที เพื่อจำลองระบบ Real-time Dashboard
time.sleep(1)
st.rerun()

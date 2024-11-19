import streamlit as st
import psutil
import GPUtil
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np

def create_monitoring_dashboard():
    # 대시보드 제목
    st.title("실시간 시스템 모니터링 대시보드")
    st.write("2초마다 자동 업데이트")

    # 데이터 저장용 리스트
    if 'cpu_data' not in st.session_state:
        st.session_state.cpu_data = []
        st.session_state.memory_data = []
        st.session_state.gpu_data = []
        st.session_state.timestamps = []

    # 실시간 차트를 위한 플레이스홀더
    chart_placeholder = st.empty()

    while True:
        # CPU 사용량
        cpu_percent = psutil.cpu_percent()
        
        # 메모리 사용량
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU 사용량
        try:
            gpus = GPUtil.getGPUs()
            gpu_percent = gpus[0].load * 100 if gpus else 0
        except:
            gpu_percent = 0

        # 데이터 저장
        st.session_state.cpu_data.append(cpu_percent)
        st.session_state.memory_data.append(memory_percent)
        st.session_state.gpu_data.append(gpu_percent)
        st.session_state.timestamps.append(datetime.now())

        # 최근 30개 데이터포인트만 유지
        if len(st.session_state.cpu_data) > 30:
            st.session_state.cpu_data = st.session_state.cpu_data[-30:]
            st.session_state.memory_data = st.session_state.memory_data[-30:]
            st.session_state.gpu_data = st.session_state.gpu_data[-30:]
            st.session_state.timestamps = st.session_state.timestamps[-30:]

        # 차트 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(st.session_state.timestamps, st.session_state.cpu_data, 
                label='CPU Usage (%)', color='#00ff00')
        ax.plot(st.session_state.timestamps, st.session_state.memory_data, 
                label='Memory Usage (%)', color='#0000ff')
        ax.plot(st.session_state.timestamps, st.session_state.gpu_data, 
                label='GPU Usage (%)', color='#ff0000')

        # 차트 스타일링
        ax.set_ylim(0, 100)
        ax.set_xlabel('Time')
        ax.set_ylabel('Usage (%)')
        ax.legend()
        ax.grid(True)

        # 현재 사용량 표시
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("CPU 사용량", f"{cpu_percent:.1f}%")
        with metrics_col2:
            st.metric("메모리 사용량", f"{memory_percent:.1f}%")
        with metrics_col3:
            st.metric("GPU 사용량", f"{gpu_percent:.1f}%")

        # 차트 업데이트
        chart_placeholder.pyplot(fig)
        plt.close(fig)

        # 2초 대기
        time.sleep(2)

if __name__ == "__main__":
    create_monitoring_dashboard()
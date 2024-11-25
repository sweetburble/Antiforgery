import streamlit as st
import psutil
import GPUtil
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
import numpy as np

def get_detailed_cpu_usage():
    # CPU 코어별 사용량
    core_usage = psutil.cpu_percent(interval=1, percpu=True)
    core_data = [{'Core': f'Core {i}', 'Usage %': usage} 
                 for i, usage in enumerate(core_usage)]
    
    # 프로세스별 스레드 정보
    thread_data = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'num_threads']):
        try:
            # 스레드 정보 수집
            proc_info = proc.info
            if proc_info['cpu_percent'] > 0.1:  # CPU 사용량이 0.1% 이상인 프로세스만
                thread_data.append({
                    'PID': proc_info['pid'],
                    'Name': proc_info['name'],
                    'Threads': proc_info['num_threads'],
                    'CPU %': proc_info['cpu_percent']
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return core_data, sorted(thread_data, key=lambda x: x['CPU %'], reverse=True)[:10]

def create_monitoring_dashboard():
    st.title("상세 CPU 모니터링 대시보드")
    st.write("2초마다 자동 업데이트")

    # 세션 상태 초기화
    if 'cpu_data' not in st.session_state:
        st.session_state.cpu_data = []
        st.session_state.memory_data = []
        st.session_state.gpu_data = []
        st.session_state.timestamps = []

    # 레이아웃 분할
    chart_col, detail_col = st.columns([2, 1])

    with chart_col:
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()

    with detail_col:
        st.subheader("CPU 코어별 사용량")
        core_placeholder = st.empty()
        st.subheader("프로세스별 스레드 정보")
        thread_placeholder = st.empty()

    while True:
        # 기본 시스템 메트릭
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_percent = gpus[0].load * 100 if gpus else 0
        except:
            gpu_percent = 0

        # 상세 CPU 정보 수집
        core_data, thread_data = get_detailed_cpu_usage()

        # 데이터 저장 및 관리
        st.session_state.cpu_data.append(cpu_percent)
        st.session_state.memory_data.append(memory_percent)
        st.session_state.gpu_data.append(gpu_percent)
        st.session_state.timestamps.append(datetime.now())

        if len(st.session_state.cpu_data) > 30:
            st.session_state.cpu_data = st.session_state.cpu_data[-30:]
            st.session_state.memory_data = st.session_state.memory_data[-30:]
            st.session_state.gpu_data = st.session_state.gpu_data[-30:]
            st.session_state.timestamps = st.session_state.timestamps[-30:]

        # 차트 업데이트
        with chart_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(st.session_state.timestamps, st.session_state.cpu_data, 
                    label='CPU Usage (%)', color='#00ff00')
            ax.plot(st.session_state.timestamps, st.session_state.memory_data, 
                    label='Memory Usage (%)', color='#0000ff')
            ax.plot(st.session_state.timestamps, st.session_state.gpu_data, 
                    label='GPU Usage (%)', color='#ff0000')
            ax.set_ylim(0, 100)
            ax.set_xlabel('Time')
            ax.set_ylabel('Usage (%)')
            ax.legend()
            ax.grid(True)
            chart_placeholder.pyplot(fig)
            plt.close(fig)

            # 메트릭 업데이트
            with metrics_placeholder.container():
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("CPU 전체", f"{cpu_percent:.1f}%")
                with m2:
                    st.metric("메모리", f"{memory_percent:.1f}%")
                with m3:
                    st.metric("GPU", f"{gpu_percent:.1f}%")

        # 상세 정보 업데이트
        with detail_col:
            # CPU 코어 정보
            core_df = pd.DataFrame(core_data)
            core_placeholder.dataframe(core_df, hide_index=True)

            # 스레드 정보
            thread_df = pd.DataFrame(thread_data)
            thread_placeholder.dataframe(thread_df, hide_index=True)

        time.sleep(2)

if __name__ == "__main__":
    create_monitoring_dashboard()
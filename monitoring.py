import streamlit as st
import psutil
import GPUtil
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
import numpy as np
import pynvml  # NVIDIA GPU 상세 정보를 위한 라이브러리
import os
from py3nvml import py3nvml   # 새로운 접근 방식

# Windows의 경우 NVML DLL 경로 설정
os.environ['NVML_DLL_PATH'] = 'C:/Windows/System32/nvml.dll'


def initialize_gpu():
    try:
        if not hasattr(pynvml, 'nvmlInit'):
            raise ImportError("NVML 라이브러리를 찾을 수 없습니다")
        
        # NVML 초기화 전에 DLL 경로 확인
        if os.path.exists(os.environ.get('NVML_DLL_PATH', '')):
            pynvml.nvmlInit()
            return True
        else:
            print(f"NVML DLL not found at: {os.environ.get('NVML_DLL_PATH', 'Not Set')}")
            return False
    except Exception as e:
        print(f"GPU 초기화 실패: {str(e)}")
        return False

def get_detailed_cpu_gpu_usage():
    # CPU 코어별 사용량 (기존 코드 유지)
    core_usage = psutil.cpu_percent(interval=1, percpu=True)
    core_data = [{'Core': f'Core {i}', 'Usage %': usage} 
                 for i, usage in enumerate(core_usage)]
    
    # GPU 상세 정보
    gpu_data = []
    try:
        py3nvml.nvmlInit()
        deviceCount = py3nvml.nvmlDeviceGetCount()
        
        for i in range(deviceCount):
            handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
            
            # 기본 정보
            info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = py3nvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = py3nvml.nvmlDeviceGetTemperature(handle, py3nvml.NVML_TEMPERATURE_GPU)
            
            # 추가 정보
            name = py3nvml.nvmlDeviceGetName(handle)
            # decode 처리를 조건부로 수행
            gpu_name = name.decode("utf-8") if isinstance(name, bytes) else name
            
            try:
                power_usage = py3nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = py3nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                power_info = f"{power_usage:.1f}W/{power_limit:.1f}W"
            except:
                power_info = "N/A"

            try:
                fan_speed = py3nvml.nvmlDeviceGetFanSpeed(handle)
                fan_info = f"{fan_speed}%"
            except:
                fan_info = "N/A"

            try:
                clock_info = py3nvml.nvmlDeviceGetClockInfo(handle, py3nvml.NVML_CLOCK_GRAPHICS)
                clock_str = f"{clock_info}MHz"
            except:
                clock_str = "N/A"

            gpu_data.append({
                'GPU': gpu_name,
                'Usage %': utilization.gpu,
                'Memory Used': f"{info.used/(1024*1024):.0f}MB/{info.total/(1024*1024):.0f}MB",
                'Memory %': f"{(info.used/info.total)*100:.1f}%",
                'Temperature °C': temperature,
                'Power': power_info,
                'Fan Speed': fan_info,
                'Clock': clock_str
            })
            
        py3nvml.nvmlShutdown()
    except Exception as e:
        print(f"GPU 정보 수집 실패: {str(e)}")
        gpu_data.append({
            'GPU': 'GPU 0',
            'Usage %': 'N/A',
            'Memory Used': 'N/A',
            'Memory %': 'N/A',
            'Temperature °C': 'N/A',
            'Power': 'N/A',
            'Fan Speed': 'N/A',
            'Clock': 'N/A'
        })
    
    return core_data, gpu_data

def create_monitoring_dashboard():
    st.title("CPU/GPU 상세 모니터링 대시보드")
    st.write("2초마다 자동 업데이트")

    # 세션 상태 초기화
    if 'cpu_data' not in st.session_state:
        st.session_state.cpu_data = []
        st.session_state.memory_data = []
        st.session_state.gpu_data = []
        st.session_state.timestamps = []

    # 첫 번째 행: 그래프와 CPU 정보
    row1_col1, row1_col2 = st.columns([2, 1])
    
    # 그래프 영역
    with row1_col1:
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()
    
    # CPU 정보 영역
    with row1_col2:
        st.subheader("CPU 코어별 사용량")
        core_placeholder = st.empty()

    # 두 번째 행: GPU 정보 (전체 너비 사용)
    st.subheader("GPU 상세 정보")
    gpu_placeholder = st.empty()

    while True:
        # 기존 모니터링 코드는 동일하게 유지
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_percent = gpus[0].load * 100 if gpus else 0
        except:
            gpu_percent = 0

        # 상세 정보 수집
        core_data, gpu_data = get_detailed_cpu_gpu_usage()

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
        with row1_col1:
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

        # CPU 정보 표시
        with row1_col2:
            core_placeholder.dataframe(
                pd.DataFrame(core_data),
                hide_index=True
            )

        # GPU 정보 표시 (전체 너비 사용)
        gpu_placeholder.dataframe(
            pd.DataFrame(gpu_data),
            hide_index=True,
            column_config={
                "GPU": st.column_config.TextColumn(width="medium"),
                "Memory Used": st.column_config.TextColumn(width="medium"),
                "Memory %": st.column_config.TextColumn(width="small"),
                "Power": st.column_config.TextColumn(width="medium"),
                "Fan Speed": st.column_config.TextColumn(width="small"),
                "Clock": st.column_config.TextColumn(width="small")
            }
        )

        time.sleep(2)

if __name__ == "__main__":
    create_monitoring_dashboard()
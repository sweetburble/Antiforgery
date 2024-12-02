import psutil
import pynvml
import time
import logging
from datetime import datetime
import statistics
import os

def setup_monitoring_logger(log_dir='gpu_monitoring_logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'resource_usage_{timestamp}.txt')
    
    # 리소스 모니터링용 독립 로거 생성
    logger = logging.getLogger('resource_monitor_logger')
    logger.setLevel(logging.INFO)
    
    # 이전 핸들러 제거
    if logger.handlers:
        logger.handlers.clear()
    
    # 새로운 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    
    return logger

class ResourceMonitor:
    def __init__(self, sampling_interval=1):
        self.sampling_interval = sampling_interval
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.logger = setup_monitoring_logger()
        self.do_run = True  # 스레드 종료를 위한 플래그
        
        # GPU 모니터링 초기화
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]

    def collect_metrics(self):
        while self.do_run:  # 스레드 종료 플래그 확인
            # CPU 사용률
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.append(cpu_percent)
            
            # GPU 사용률과 메모리
            for handle in self.handles:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_usage.append(gpu_util)
                self.gpu_memory.append(gpu_mem.used / gpu_mem.total * 100)
            
            time.sleep(self.sampling_interval)
    
    def log_statistics(self):
        self.logger.info(f"\n=== 리소스 사용 통계 ===")
        self.logger.info(f"CPU 사용률:")
        self.logger.info(f"  평균: {statistics.mean(self.cpu_usage):.2f}%")
        self.logger.info(f"  최소: {min(self.cpu_usage):.2f}%")
        self.logger.info(f"  최대: {max(self.cpu_usage):.2f}%")
        
        for i in range(self.device_count):
            gpu_stats = self.gpu_usage[i::self.device_count]
            mem_stats = self.gpu_memory[i::self.device_count]
            
            self.logger.info(f"\nGPU {i} 사용률:")
            self.logger.info(f"  평균: {statistics.mean(gpu_stats):.2f}%")
            self.logger.info(f"  최소: {min(gpu_stats):.2f}%")
            self.logger.info(f"  최대: {max(gpu_stats):.2f}%")
            
            self.logger.info(f"GPU {i} 메모리:")
            self.logger.info(f"  평균: {statistics.mean(mem_stats):.2f}%")
            self.logger.info(f"  최소: {min(mem_stats):.2f}%")
            self.logger.info(f"  최대: {max(mem_stats):.2f}%")

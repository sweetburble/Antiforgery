import logging
from datetime import datetime
import os

def setup_logger(log_dir='time_logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'AntiForgery_log_{timestamp}.txt')
    
    # 로거 이름을 지정하여 독립적인 로거 생성
    logger = logging.getLogger('execution_time_logger')
    logger.setLevel(logging.INFO)
    
    # 이전 핸들러 제거
    if logger.handlers:
        logger.handlers.clear()
    
    # 새로운 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    
    return logger
import logging
from datetime import datetime
import os

def setup_logger(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'Antiforgery_log_{timestamp}.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # 타임스탬프 제거
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8')  # UTF-8 인코딩 설정, 콘솔 출력 제거
        ]
    )
    return logging.getLogger()
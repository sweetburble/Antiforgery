from calendar import c
from math import e
import torch
import numpy as np
import argparse
import os
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F

from color_space import *
from data_loader import get_loader
from utils import *

from model import Generator, Discriminator

import time
import ray
from logger import setup_logger

def main():
    logger = setup_logger() # 로깅을 위한 logger 설정
    total_time = time.time() # 시작 시간 저장

    ray.init() # Ray 초기화

    # 설정값 초기화
    config = {
        'celeba_image_dir': 'C:/Users/Bandi/Desktop/Fork/AntiForgery/data/celeba/images',
        'attr_path': 'C:/Users/Bandi/Desktop/Fork/AntiForgery/data/celeba/list_attr_celeba.txt',
        'selected_attrs': ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
        'batch_size': 4,
        'image_size': 256,
        'g_conv_dim': 64,
        'c_dim': 5,
        'g_repeat_num': 6,
        'num_workers': 4,
        'model_save_dir': 'stargan_celeba_256/models',
        'model_path': 'stargan_celeba_256/models/200000-D.ckpt',  # 생성자 가중치 경로 추가
        'resume_iters': 200000,
    }

    # 데이터 로더 초기화
    # 데이터 로더 설정
    celeba_loader = get_loader(
        image_dir=config['celeba_image_dir'],
        attr_path=config['attr_path'],
        selected_attrs=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
        mode="test"
    )

    # data loader
    start_time = time.time()
    
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"************ celeba_loader 실행 시간: {execution_time} 초")
    print(f"************ celeba_loader 실행 시간: {execution_time} 초")

    # 모델 설정 및 로드
    start_time = time.time()

    # 모델 설정
    # G = Generator(conv_dim=64, c_dim=5, repeat_num=6).cuda()
    # D = Discriminator(image_size=256, conv_dim=64, c_dim=5, repeat_num=6).cuda()

    G_path = os.path.join(config['model_save_dir'], '{}-G.ckpt'.format(config['resume_iters']))
    D_path = os.path.join(config['model_save_dir'], '{}-D.ckpt'.format(config['resume_iters']))

    # G.load_state_dict(torch.load(G_path))
    # D.load_state_dict(torch.load(D_path))

    # 이미지 처리 루프
    for i, (x_real_batch, c_org_batch) in enumerate(celeba_loader):
        x_real_batch = x_real_batch.cuda()  # 배치를 GPU로 이동

        # create_labels 호출 후 리스트 내부 텐서를 CPU로 이동
        c_trg_list_batch = create_labels(c_org_batch, c_dim=5, dataset='CelebA', selected_attrs=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"])

        # Ray 태스크 호출 (배치 단위로 처리)
        future = process_batch.remote(x_real_batch.cpu(), [c.cpu() for c in c_trg_list_batch], G_path)
        results = ray.get(future)  # 결과 가져오기

        # 결과 저장
        for j, (x_adv, pert) in enumerate(results):
            result_path = os.path.join("results", f"{i}_{j}_adv.jpg")
            save_image(denorm(x_adv), result_path)
    
    total_end_time = time.time()
    total = total_end_time-total_time
    print(f"*********** main 실행시간: {total} 초")
    logger.info(f"*********** main 실행시간: {total} 초")    

'''
여기서부터 병렬/분산 처리를 위한 코드 추가

'''


@ray.remote(num_gpus=0.5)  # GPU 리소스 할당 (0.5 GPU 사용)
def process_batch(x_real_batch, c_trg_list_batch, model_path):
    """Ray 태스크로 LAB 공격 수행."""
    from model import Generator  # Ray 태스크 내에서 모델 로드
    from utils import lab_attack

    # 모델 로드 및 GPU 이동
    G = Generator(conv_dim=64, c_dim=5, repeat_num=6).cuda()
    G.load_state_dict(torch.load(model_path))
    G.eval()

    # 결과 저장
    results = []
    for x_real, c_trg in zip(x_real_batch, c_trg_list_batch):
        x_adv, pert = lab_attack(x_real.unsqueeze(0).cuda(), [c_trg.cuda()], G)
        results.append((x_adv.cpu(), pert.cpu()))
    return results


# 사용 예시
if __name__ == "__main__":
    main()
import logging
import torch
import os
from multiprocessing import Pool
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")


def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    logger.info(f"devices info {devices_info}")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occupy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    device = torch.device(f'cuda:{cuda_device}')
    print(f"cuda device {device}")
    torch.cuda.set_device(device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    print(f"max mem {max_mem}")
    block_mem = max_mem - used - 1000
    x = torch.cuda.FloatTensor(256, 1024, block_mem).to(device)
    while True:
        x += 1
        x -= 1
    del x


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    mp.set_start_method('spawn')
    # Now this script will use all available devices one applied on Slurm.
    # Each GPU will run an independent process.
    num_of_gpus = torch.cuda.device_count()
    print(f"num_of_gpus: {num_of_gpus}")
    for gpu in range(num_of_gpus):
        print(f"gpu: {gpu}")
        print(f"total, used: {check_mem(gpu)}")
    with Pool(num_of_gpus) as p:
        p.map(occupy_mem, [str(g) for g in range(num_of_gpus)])



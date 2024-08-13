import logging
import random
import time

import torch
import os
from multiprocessing import Pool
import multiprocessing as mp



logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)
logger = logging.getLogger(__name__)

# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")


def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    # logger.info(f"devices info {devices_info}")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used

def get_running_processes(gid):
    return os.popen(f"nvidia-smi -i {gid} --query-compute-apps pid --format=csv,noheader").read().strip().split("\n")

def gpus_are_empty(gid):
    return get_running_processes(gid) == ['']

def occupy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    device = torch.device(f'cuda:{cuda_device}')
    logger.info(f"cuda device {device}")
    torch.cuda.set_device(device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.5)
    logger.info(f"max mem {max_mem}")
    block_mem = (max_mem - used) * random.randint(5, 9) * 0.1
    x = torch.FloatTensor(256, 1024, int(block_mem)).to(device)
    time.sleep(5)
    running_processes = get_running_processes(cuda_device)
    occupy_process = running_processes[0]
    logger.info(f"occupy running processes {running_processes}")
    while True:
        x += 1
        x -= 1
        time.sleep(0.1)
        if running_processes != get_running_processes(cuda_device):
            logger.info(f"New processes detected, stop occupying the GPU")
            break

    logger.info(f"Stop occupying the GPU")
    os.system("kill -9 " + occupy_process)
    #
    # x.to("cpu")
    # del x
    # torch.cuda.empty_cache()



def main(gid):
    idle_start = None
    wait_for = 1
    # mp.set_start_method('spawn')
    while True:
        if gpus_are_empty(gid):
            if idle_start is None:
                idle_start = time.time()
            elif time.time() - idle_start > wait_for:
                logger.info(f"No processes are running on the GPUs over the last {wait_for}s")
                logger.info("Starting the processes")

                p = mp.Process(target=occupy_mem, args=(gid,))
                p.start()
                p.join()

                # with Pool(1) as p:
                #     p.map(occupy_mem, [gid])
                # occupy_mem(gid)
                # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
                # Now this script will use all available devices one applied on Slurm.
                # Each GPU will run an independent process.
                # num_of_gpus = torch.cuda.device_count()
                # logger.info(f"num_of_gpus: {num_of_gpus}")
                # for gpu in range(num_of_gpus):
                #     logger.info(f"gpu: {gpu}")
                #     logger.info(f"total, used: {check_mem(gpu)}")
                # with Pool(num_of_gpus) as p:
                #     p.map(occupy_mem, [str(g) for g in range(num_of_gpus)])
        else:
            wait_for = 600
            processes_list = get_running_processes(gid)
            logger.info(f"Processes are running on the GPUs: {processes_list}")
            logger.info("Waiting for the GPUs to be free")
            logger.info("Checking again in 10 mins")
            idle_start = None
            time.sleep(600)


if __name__ == "__main__":
    # main(gid=0)
    # mp.set_start_method('spawn')
    process_list = []
    num_of_gpus = torch.cuda.device_count()
    logger.info(f"num_of_gpus: {num_of_gpus}")
    for gpu in range(num_of_gpus):
        logger.info(f"gpu: {gpu}")
        logger.info(f"total, used: {check_mem(gpu)}")
    # with Pool(num_of_gpus) as p:
    #     p.map(main, [str(g) for g in range(num_of_gpus)])
    for i in range(num_of_gpus):
        p = mp.Process(target=main, args=(i,), daemon=False)
        p.start()
        process_list.append(p)

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()
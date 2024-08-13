# -*- coding: utf-8 -*-

"""Check the running and queueing jobs on a worker on LRZ-Slurm.
Copyright @ Shuo Chen
"""

import logging
import os
from tabulate import tabulate
import argparse
import pandas as pd


logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--worker", type=str, default="mcml-dgx-001")
args = parser.parse_args()
WORKER_NAME = args.worker

partition2nodelist = {
    "lrz-dgx-a100-80x8": [
        "lrz-dgx-a100-001",
        "lrz-dgx-a100-002",
        "lrz-dgx-a100-004",
        "lrz-dgx-a100-005",
    ],
    "mcml-dgx-a100-40x8": [
        "mcml-dgx-001",
        "mcml-dgx-002",
        "mcml-dgx-003",
        "mcml-dgx-004",
        "mcml-dgx-005",
        "mcml-dgx-006",
        "mcml-dgx-007",
        "mcml-dgx-008",
    ],
    "mcml-hgx-a100-80x4": [
        "mcml-hgx-a100-001",
        "mcml-hgx-a100-002",
        "mcml-hgx-a100-003",
        "mcml-hgx-a100-004",
        "mcml-hgx-a100-005",
        "mcml-hgx-a100-006",
        "mcml-hgx-a100-007",
        "mcml-hgx-a100-008",
        "mcml-hgx-a100-009",
        "mcml-hgx-a100-010",
        "mcml-hgx-a100-011",
        "mcml-hgx-a100-012",
        "mcml-hgx-a100-013",
        "mcml-hgx-a100-014",
    ],
    "mcml-hgx-a100-80x4-mig": [
        "mcml-hgx-a100-015",
        "mcml-hgx-a100-016",
        "mcml-hgx-a100-017",
        "mcml-hgx-a100-018",
        "mcml-hgx-a100-019",
        "mcml-hgx-a100-020",
        "mcml-hgx-a100-021",
    ],
    "lrz-hgx-a100-80x4": [
        "lrz-hgx-a100-001",
        "lrz-hgx-a100-002",
        "lrz-hgx-a100-003",
        "lrz-hgx-a100-004",
        "lrz-hgx-a100-005",
    ],
}

partition2info = {
    "lrz-dgx-a100-80x8": {"cpu": 256, "gpu": 8, "gpu-mem": "80GB", "mem": "2000GB"},
    "mcml-dgx-a100-40x8": {"cpu": 256, "gpu": 8, "gpu-mem": "40GB", "mem": "1000GB"},
    "mcml-hgx-a100-80x4": {"cpu": 48, "gpu": 4, "gpu-mem": "80GB", "mem": "1000GB"},
    "mcml-hgx-a100-80x4-mig": {"cpu": 48, "gpu": 4, "gpu-mem": "80GB", "mem": "1000GB"},
    "lrz-hgx-a100-80x4": {"cpu": 96, "gpu": 4, "gpu-mem": "80GB", "mem": "1000GB"},
}

partitions = list(partition2nodelist.keys())


def get_all_relevant_jobs(worker_name):
    """
    relevant jobs are:
    1. jobs running on this worker
    2. jobs waiting on this worker and we can know it from the NODELIST(REASON)
    3. jobs waiting but no worker info on NODELIST(REASON), e.g. Resources, Priority
    """
    stream = os.popen(f"squeue")
    output = str(stream.read()).split("\n")
    output = [" ".join(line.split()) for line in output]
    relevant_jobs = []
    for job in output:
        if worker_name in job:
            # if worker_name in job or "Resources" in job or "Priority" in job:
            job_id = job.split(" ")[0]
            if "_" in job_id:
                job_id = job_id.split("_")[0]
            relevant_jobs.append(job_id)
    return relevant_jobs


def get_job_status(jobids, worker_name, partition, total_cpu, total_gpu, mem):
    """
    jobids: list of job ids, e.g. ["1112475", "1112345"]
    """
    logger.info(
        f"************** Partition: {partition}; Worker: {worker_name}; Mem per GPU: {mem} ********************"
    )
    # logger.info(
    #     f"JobId\t    JobName\t\t UserId\t\t JobState\t NodeList\t NumCPUs   GPUs")
    count_cpu = 0
    count_gpu = 0
    for job in jobids:
        stream = os.popen(f"scontrol show job {job}")
        output = str(stream.read()).split("\n")
        # logger.info(output)
        flag = False
        for info in output:
            if worker_name in info:
                flag = True
                break
        if not flag:
            continue
        output = [" ".join(line.split()) for line in output]

        ReqNodeList = ""
        NodeList = ""
        TresPerNode = ""
        for info in output:
            if "JobId" in info:
                JobId = info.split(" ")[0].split("=")[-1]
                JobName = info.split(" ")[1].split("=")[-1]
                JobName = JobName[:15]
                if len(JobName) < 15:
                    JobName += " " * (15 - len(JobName))
            if "UserId" in info:
                UserId = info.split(" ")[0].split("=")[-1]
                UserId = UserId.split("(")[0]
                UserId = UserId[:8]
                if len(UserId) < 8:
                    UserId += " " * (8 - len(UserId))
            if "JobState" in info:
                JobState = info.split(" ")[0].split("=")[-1]
                Reason = info.split(" ")[1].split("=")[-1]
            if "ReqNodeList" in info:
                req_worker = info.split(" ")[0].split("=")[1]
                ReqNodeList = req_worker
            if "NodeList" in info:
                NodeList = info.split("=")[1]
                if NodeList == "(null)":
                    NodeList = ReqNodeList
            if "NumCPUs" in info:
                NumCPUs = info.split(" ")[1].split("=")[-1]
                # logger.debug(f"NumCPUs: {NumCPUs}")
                count_cpu += int(NumCPUs)
            if "TresPerNode" in info:
                TresPerNode = info.split("=")[-1]
                if "gpu" in TresPerNode:
                    GPUs = TresPerNode.split(":")[-1]
                    # logger.debug(f"GPUs: {GPUs}")
                    count_gpu += int(GPUs)
        if ReqNodeList == "(null)":
            ReqNodeList = NodeList
        if ReqNodeList == worker_name or NodeList == worker_name:
            logger.info(
                f"{JobId: <10} {JobName:<10}\t {UserId}\t {JobState}\t {NodeList}\t {NumCPUs}\t {TresPerNode}"
            )

    logger.info(
        f"                               CPU Usage: {count_cpu} / {total_cpu}; GPU Usage: {count_gpu}/{total_gpu}"
    )
    # logger.info(f"****************************************************************************************************")
    logger.info(f"\n")


def get_detailed_info():
    for partition, nodelist in partition2nodelist.items():
        for worker_name in nodelist:
            relevant_jobs = get_all_relevant_jobs(worker_name)
            logger.info(relevant_jobs)
            get_job_status(
                relevant_jobs,
                worker_name=worker_name,
                partition=partition,
                total_cpu=partition2info[partition]["cpu"],
                total_gpu=partition2info[partition]["gpu"],
                mem=partition2info[partition]["gpu-mem"],
            )


def get_summary_on_node(partition, worker_name, jobs):
    count_cpu = 0
    count_gpu = 0
    total_cpu = partition2info[partition]["cpu"]
    total_gpu = partition2info[partition]["gpu"]
    for job in jobs:
        stream = os.popen(f"scontrol show job {job}")
        output = str(stream.read()).split("\n")
        # logger.info(output)
        flag = False
        for info in output:
            if worker_name in info:
                flag = True
                break
        if not flag:
            continue
        output = [" ".join(line.split()) for line in output]

        for info in output:
            if "NumCPUs" in info:
                NumCPUs = info.split(" ")[1].split("=")[-1]
                # logger.debug(f"NumCPUs: {NumCPUs}")
                count_cpu += int(NumCPUs)
            if "TresPerNode" in info:
                TresPerNode = info.split("=")[-1]
                if "gpu" in TresPerNode:
                    GPUs = TresPerNode.split(":")[-1]
                    # logger.debug(f"GPUs: {GPUs}")
                    # count_gpu += int(GPUs) if type(GPUs) == int else 0
                    count_gpu += int(GPUs)
    # logger.info(f"{partition}\t {worker_name}\t {count_cpu} / {total_cpu} \t {count_gpu}/{total_gpu}")
    return count_cpu, count_gpu



def get_problematic_nodes():
    stream = os.popen(f"sinfo -R")
    outputs = str(stream.read()).split("\n")
    nodes = []
    for output in outputs:
        node = output.split(" ")[-1]
        if "NODELIST" in node:
            continue
        if "[" not in node and "]" not in node:
            nodes.append(node)
        else:
            node_prefix = node.split("[")[0]
            # print(node_prefix)
            node_range = node.split("[")[1].split("]")[0]
            if "-" in node_range:
                node_range = node_range.split("-")
            elif "," in node_range:
                node_range = node_range.split(",")
            # print(node_range)
            for n in node_range:
                nodes.append(node_prefix + n)
    # print(f"Problematic Nodes: {nodes}")
    return nodes


def get_report():
    df = pd.DataFrame(
        {"Partition": [], "Node": [], "CPU Usage": [], "GPU Usage": [], "GPU Mem": []},
    )
    print(f"Available Resources: ")
    avai_partitions = set()
    avai_nodes = set()
    problematic_nodes = get_problematic_nodes()
    for partition, nodelist in partition2nodelist.items():
        for worker_name in nodelist:
            relevant_jobs = get_all_relevant_jobs(worker_name)
            # logger.info(relevant_jobs)
            cpu, gpu = get_summary_on_node(partition, worker_name, relevant_jobs)
            if worker_name in problematic_nodes:
                continue
            if gpu < partition2info[partition]["gpu"]:
                avai_partitions.add(partition)
                avai_nodes.add(worker_name)
                # add new line to df
                df = df._append(
                    {
                        "Partition": partition,
                        "Node": worker_name,
                        "CPU Usage": f"{cpu} / {partition2info[partition]['cpu']}",
                        "GPU Usage": f"{gpu} / {partition2info[partition]['gpu']}",
                        "GPU Mem": partition2info[partition]["gpu-mem"],
                        # "Comment": comment,
                    },
                    ignore_index=True,
                )
    return avai_partitions, avai_nodes, df


def get_queue_report():
    df = pd.DataFrame(
        {"Partition": [], "#Job Running": [], "#Job Queueing": [], "GPU Mem": []},
    )
    print(f"Busy Resources: ")
    busy_partitions = set()
    squeue_script = f"squeue --format=%13i%25j%15u%10T%10M%15l%20P%20R%Q --sort=Q -p "
    for partition, nodelist in sorted(partition2nodelist.items()):
        squeue_result = os.popen(squeue_script + partition).read()
        squeue_result = squeue_result.split("\n")
        job_running = 0
        job_queueing = 0

        for line in squeue_result:
            if "RUNNING" in line:
                job_running += 1
            if "PENDING" in line:
                job_queueing += 1
        df = df._append(
            {
                "Partition": partition,
                "#Job Running": f"{job_running:.0f}",
                "#Job Queueing": f"{job_queueing:.0f}",
                "GPU Mem": partition2info[partition]["gpu-mem"],
                # "Comment": comment,
            },
            ignore_index=True,
        )

    return df



if __name__ == "__main__":
    _, _, df = get_report()
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))
    df = get_queue_report()
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))

    # get_problematic_nodes()

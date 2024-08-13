
import os
from monitor_lrz import partition2info, partition2nodelist, get_report, get_queue_report
from tabulate import tabulate


avai_partitions, avai_nodes, df = get_report()
print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))

df = get_queue_report()
print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))

avai_partitions = sorted(list(avai_partitions))

# number2partition = {i: p for i, p in enumerate(avai_partitions)}
number2partition = {i: p for i, p in enumerate(partition2info.keys())}
number2node = {i: n for i, n in enumerate(avai_nodes)}

print("Input the number of the partition you want to use:")
for i, p in number2partition.items():
    print(f"{i}: {p}")


partition_number = int(input())
partition = number2partition[partition_number]

print(f"Require #GPUs on {partition}: ")
gpu_number = int(input())

ratio = gpu_number / partition2info[partition]["gpu"]

default_cpu = int(ratio * partition2info[partition]["cpu"]) - 2
default_mem = int(ratio * int(partition2info[partition]["mem"].split("G")[0])) - 100
default_time = "4-00:00:00" if "mcml" in partition else "3-00:00:00"
default_name = "vl"

print(f"Require CPUs (default {default_cpu}): ")
inp = input()
cpu_number = int(inp) if inp else default_cpu

print(f"Require Memory (default {default_mem}): ")
inp = input()
mem = int(inp) if inp else default_mem

print(f"Require Time (default {default_time}) : ")
inp = input()
running_time = inp if inp else default_time

print(f"Job Name (default {default_name}) : ")
inp = input()
job_name = inp if inp else default_name

print(f"Will Allocate {cpu_number} CPUs, {gpu_number} GPUs, {mem}G Memory, {running_time} Time on {partition} with job name {job_name}, Enter to continue:")
input()

if "mcml" in partition:
    os.system(f"salloc -p {partition} -q mcml --gres=gpu:{gpu_number} --cpus-per-task={cpu_number} --mem={mem}G --time={running_time} --job-name={job_name}")
else:
    os.system(f"salloc -p {partition} --gres=gpu:{gpu_number} --cpus-per-task={cpu_number} --mem={mem}G --time={running_time} --job-name={job_name}")
import os

disk_info = os.popen(
    "dssusrinfo all"
).read()

used = 0
total = 0
dss_info = disk_info.split("\n")
for info in dss_info:
    if "pn34sa-dss-0000" in info and "GB" in info:
        l = info.split(" ")
        l = [x for x in l if x != ""]
        # print(l)
        total = l[-4]
        used = l[-6]
        print(f"DSS Container:\t Used: {used} GB / Total: {total} GB")


dss_ratio = int(used) / int(total)

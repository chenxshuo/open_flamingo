import os

disk_info = os.popen(
    "dssusrinfo all"
).read()

home_infos = disk_info.split("************************************")
for home_info in home_infos:
    if "DSS Homedir" in home_info:
        home_info = home_info.split("\n")
        for info in home_info:
            if "GB" in info:
                l = info.split(" ")
                l = [x for x in l if x != ""]
                total = l[-4]
                used = l[-6]
                print(f"DSS Homedir:\t Used: {used} GB / Total: {total} GB")


dss_info = disk_info.split("\n")
for info in dss_info:
    if "pn34sa-dss-0000" in info and "GB" in info:
        if "Container MAX" in info:
            l = info.split(" ")
            l = [x for x in l if x != ""]
            # print(l)
            mine = l[2]
            print(f"DSS Container:\t di93zun: {mine} GB")
        else:
            l = info.split(" ")
            l = [x for x in l if x != ""]
            # print(l)
            total = l[-4]
            used = l[-6]
            print(f"DSS Container:\t Used: {used} GB / Total: {total} GB")



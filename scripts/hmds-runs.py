import os
import subprocess
import itertools
import random

ranks = [10, 20]

for file in os.listdir(".\data\hmds-graphs"):
    file_base = file.split('.')[0]

    cmd_base  = "julia hMDS\hmds-simple.jl"
    cmd_edges = " -d data\edges\\" + file_base + ".edges"
    cmd_emb   = " -k data\emb\\" + file_base + ".emb"
    cmd_rank  = " -r "
    cmd_scale = " -t "

    for rank in ranks:
        print("Rank = ", rank)

        for i in range(10):
            scale = 0.1*(i+1)
            cmd = cmd_base + cmd_edges + cmd_emb + cmd_rank + str(rank) + cmd_scale + str(scale)
            #print(cmd)

            result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
            res_string = result.stdout.decode('utf-8')
            res_lines = res_string.splitlines()

            # grab our values
            distortion = res_lines[16].split()[5].strip(",")
            mapval     = res_lines[17].split()[2]

            print("Scale \t", scale, "\t distortion \t", distortion, "\t mAP \t", mapval)
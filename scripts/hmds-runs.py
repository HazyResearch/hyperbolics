import os
import subprocess
import itertools
import random

ranks = [5, 10]

for file in os.listdir("./data/hmds-graphs/seventree/"):
    file_base = file.split('.')[0]

    cmd_base  = "julia hMDS/hmds-simple.jl"
    cmd_edges = " -d data/edges/" + file_base + ".edges"
    cmd_emb   = " -k data/emb/" + file_base + ".emb"
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
            edge_acc   = res_lines[21].split()[7]

            if i == 0:
                input_distortion = res_lines[18].split()[6].strip(",")
                input_map        = res_lines[19].split()[3]
                input_edge_acc   = res_lines[20].split()[5]
                print("Input distortion \t", input_distortion, "\t input mAP \t", input_map, "\t input Edge Acc from MST \t", input_edge_acc, "\n") 

            print("Scale \t", scale, "\t distortion \t", distortion, "\t mAP \t", mapval, "\t Edge Acc from MST \t", edge_acc) 

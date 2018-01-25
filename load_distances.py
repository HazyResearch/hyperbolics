# load the first 3 graphs's distance matrices:
import data_prep as dp
import load_dist as ld

for i in (6,12,13):
    G = dp.load_graph(i)
    ld.save_dist_mat(G,"dists/dist_mat"+str(i)+".p") 


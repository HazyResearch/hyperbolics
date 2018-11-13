import glob, os, sys
import pandas as pd

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import iso_comp

if __name__ == '__main__':
    run_name = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(run_name + '/*.emb.final')):
        line = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0] + ' '
        dataset = os.path.splitext(os.path.splitext(line)[0])[0]
        iso_comp.run_isomap(f, dataset, 2)
        
        # with open(f, "r") as g:
        #     line += g.readline()
        # rows.append(line)
        #name = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
        #row = pd.read_csv(f, delim_whitespace=True)
        #row.index = [name]
        #rows.append(row)
        #print(f)
        # print(row)
        # os.remove(f)
    #table = pd.concat(rows)
    # print(table)
    #print(table.to_string())
    # .to_csv(f"{run_name}/{run_name}.stats", )
    #with open(f"{run_name}/{run_name}.stats", "w") as f:
        # f.write('\n'.join(lines))
        #f.write(table.to_string())


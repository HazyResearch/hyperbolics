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
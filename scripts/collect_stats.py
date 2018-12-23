import glob, os, sys
import pandas as pd

if __name__ == '__main__':
    run_name = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(run_name + '/*.stat')):
        # line = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0] + ' '
        # with open(f, "r") as g:
        #     line += g.readline()
        # rows.append(line)
        name = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
        row = pd.read_csv(f, delim_whitespace=True)
        row.index = [name]
        rows.append(row)
        # print(row)
        # os.remove(f)
    table = pd.concat(rows)
    # print(table)
    print(table.to_string())
    # .to_csv(f"{run_name}/{run_name}.stats", )
    with open(f"{run_name}/{run_name}.stats", "w") as f:
        # f.write('\n'.join(lines))
        f.write(table.to_string())


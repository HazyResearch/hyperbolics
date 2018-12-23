import numpy as np
import os

filename = 'ha30.txt'
# fileout = 'usca312.edges'
if __name__ == '__main__':
    base, ext = os.path.splitext(filename)
    fileout = f'{base}.edges'
    D = np.loadtxt(filename)
    print(D.shape)
    n = D.shape[0]
    with open(fileout, 'w') as fout:
        for i in range(n):
            for j in range(i+1,n):
                e = np.minimum(D[i][j], D[j][i])
                fout.write(f'{i} {j} {e/1000.}\n')

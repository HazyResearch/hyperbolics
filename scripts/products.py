import os
import argh
import subprocess
import itertools
import random

# ranks = [2,5,10,50,100,200]

datasets = [
    # "synthetic/sierp-C50-2",
    # "synthetic/sierp-C5-6",
    # "synthetic/diamond7"
    # "synthetic/sierp-K3-8"
    # "synthetic/tree-20-3"
    # "smalltree"
    # "bio-yeast", # 1458
    # "web-edu", # 3031
    # "grqc", # 4158
    # "ca-CSphd",
    # "facebook_combined",
    # "inf-power", # 4941
    # "california", # 5925
    "usca312",
    "bookend"
]
datasets = datasets[:-1]

# 100 dimensions
models100 = [
    #{'dim': 100, 'hyp': 1, 'edim': 0, 'euc': 0, 'sdim': 0, 'sph': 0},
    {'dim': 0, 'hyp': 0, 'edim': 100, 'euc': 1, 'sdim': 0, 'sph': 0},
    #{'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 100, 'sph': 1},
    # {'dim': 10, 'hyp': 10, 'edim': 0, 'euc': 0, 'sdim': 0, 'sph': 0},
    # {'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 10, 'sph': 10},
    #{'dim': 5, 'hyp': 20, 'edim': 0, 'euc': 0, 'sdim': 0, 'sph': 0},
    #{'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 5, 'sph': 20},
    #{'dim': 2, 'hyp': 50, 'edim': 0, 'euc': 0, 'sdim': 0, 'sph': 0},
    #{'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 2, 'sph': 50},
    # {'dim': 50, 'hyp': 1, 'edim': 0, 'euc': 0, 'sdim': 50, 'sph': 1},
    # {'dim': 5, 'hyp': 10, 'edim': 0, 'euc': 0, 'sdim': 5, 'sph': 10},
    # {'dim': 2, 'hyp': 50, 'edim': 0, 'euc': 0, 'sdim': 2, 'sph': 50},
]
# 20 dimensions
models20 = [
    {'dim': 20, 'hyp': 1, 'edim': 0, 'euc': 0, 'sdim': 0, 'sph': 0},
    {'dim': 0, 'hyp': 0, 'edim': 20, 'euc': 1, 'sdim': 0, 'sph': 0},
    {'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 20, 'sph': 1},
    {'dim': 10, 'hyp': 2, 'edim': 0, 'euc': 0, 'sdim': 0, 'sph': 0},
    {'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 10, 'sph': 2},
    {'dim': 2, 'hyp': 10, 'edim': 0, 'euc': 0, 'sdim': 0, 'sph': 0},
    {'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 2, 'sph': 10},
    {'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 1, 'sph': 20},
    {'dim': 10, 'hyp': 1, 'edim': 0, 'euc': 0, 'sdim': 10, 'sph': 1},
    {'dim': 2, 'hyp': 5, 'edim': 0, 'euc': 0, 'sdim': 2, 'sph': 5},
    {'dim': 4, 'hyp': 2, 'edim': 4, 'euc': 1, 'sdim': 4, 'sph': 2},
    {'dim': 2, 'hyp': 4, 'edim': 4, 'euc': 1, 'sdim': 2, 'sph': 4},
]
# 10 dimensions
models10 = [
    {'dim': 10, 'hyp': 1, 'edim': 0, 'euc': 0, 'sdim': 0, 'sph': 0},
    {'dim': 0, 'hyp': 0, 'edim': 10, 'euc': 1, 'sdim': 0, 'sph': 0},
    {'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 10, 'sph': 1},
    {'dim': 5, 'hyp': 2, 'edim': 0, 'euc': 0, 'sdim': 0, 'sph': 0},
    {'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 5, 'sph': 2},
    {'dim': 2, 'hyp': 5, 'edim': 0, 'euc': 0, 'sdim': 0, 'sph': 0},
    {'dim': 0, 'hyp': 0, 'edim': 0, 'euc': 0, 'sdim': 2, 'sph': 5},
    {'dim': 5, 'hyp': 1, 'edim': 0, 'euc': 0, 'sdim': 5, 'sph': 1},
    {'dim': 2, 'hyp': 2, 'edim': 2, 'euc': 1, 'sdim': 2, 'sph': 2},
    # {'dim': 2, 'hyp': 2, 'edim': 0, 'euc': 0, 'sdim': 2, 'sph': 3},
    # {'dim': 2, 'hyp': 3, 'edim': 0, 'euc': 0, 'sdim': 2, 'sph': 2},
    # {'dim': 2, 'hyp': 1, 'edim': 6, 'euc': 1, 'sdim': 2, 'sph': 1},
    # {'dim': 8, 'hyp': 1, 'edim': 2, 'euc': 1, 'sdim': 0, 'sph': 0},
    # {'dim': 2, 'hyp': 4, 'edim': 2, 'euc': 1, 'sdim': 0, 'sph': 0},
    # {'dim': 0, 'hyp': 0, 'edim': 2, 'euc': 1, 'sdim': 8, 'sph': 1},
    # {'dim': 0, 'hyp': 0, 'edim': 2, 'euc': 1, 'sdim': 2, 'sph': 4}
]
models = models10 #+ models100

# lrs = [30, 100, 300, 1000]
# lrs = [10, 20, 40]
# lrs = [5, 10, 20]
lrs = [.001, .003, .01]

burn_ins = [0]

# CUDA_VISIBLE_DEVICES=1 python pytorch/pytorch_hyperbolic.py learn data/edges/synthetic/sierp-C50-2.edges --batch-size 65536 -d 50 --hyp 0 --euc 0 --edim 50 --sph 1 --sdim 51 -l 100.0 --epochs 1000 --checkpoint-freq 100 --resample-freq 500 -g --subsample 1024 --riemann --log-name C50-2.S50.log

def run_pytorch(run_name, gpus, gpc, epochs, batch_size):
    params = []
    # with open(f"{run_name}/pytorch.params", "w") as param_file:
    #     param_file.writelines("\n".join(params))
    stuff = itertools.product(datasets, models, lrs, burn_ins)
    hparams = list(stuff)
    random.shuffle(hparams)
    for dataset, model, lr, burn_in in hparams:
        # log_w = ".w" if warm_start else ""
        # log_name = f"{run_name}/{dataset}{log_w}.r{rank}.log"
        H_name = "" if model['hyp' ]== 0 else f"H{model['dim']}-{model['hyp']}."
        E_name = "" if model['euc' ]== 0 else f"E{model['edim']}-{model['euc']}."
        S_name = "" if model['sph' ]== 0 else f"S{model['sdim']}-{model['sph']}."
        log_name = f"{run_name}/{os.path.basename(dataset)}.{H_name}{E_name}{S_name}lr{lr}"
        savefile = f"{run_name}/{os.path.basename(dataset)}.{H_name}{E_name}{S_name}lr{lr}"
        if burn_in > 0:
            log_name += f".burnin{burn_in}"
        log_name += ".log"
        savefile += ".emb"
        param = [
            f"data/edges/{dataset}.edges",
            '--dim', str(model['dim']),
            '--hyp', str(model['hyp']),
            '--edim', str(model['edim']),
            '--euc', str(model['euc']),
            '--sdim', str(model['sdim']),
            '--sph', str(model['sph']),
            '--model-save-file', savefile,
            # '--log',
            '--log-name', log_name,
            '--batch-size', str(batch_size),
            '--epochs', str(epochs),
            '--checkpoint-freq', '100',
            '--resample-freq', '5000',
            # '--use-svrg',
            # '-T 0',
            '-g', '--subsample 1024',
            '--riemann',
            '--learn-scale',
            # '--logloss',
            # '--distloss',
            # '--squareloss',
            # '--symloss',
            '--burn-in', str(burn_in),
            # '--momentum', '0.9',
            '--learning-rate', str(lr)]
        params.append(" ".join(param))

    cmds = []
    for i in range(gpus):
        header = " ".join([ 'CUDA_VISIBLE_DEVICES='+str(i%gpc), 'python', 'pytorch/pytorch_hyperbolic.py', 'learn' ])
        cmds = [f'{header} {p}' for p in params[i::gpus]]
        with open(f"{run_name}/cmds{i}.sh", "w") as cmd_log:
            cmd_log.writelines('\n'.join(cmds))

    # all_cmds = [f'"{cmd0} {p}"' for p in params[0::2]] \
    # + [f'"{cmd1} {p}"' for p in params[1::2]]
    # parallel_cmd = " ".join(['parallel',
    #         ':::',
    #         *all_cmds
    #         ])
    # print(parallel_cmd)
    # with open(f"{run_name}/cmds.sh", "w") as cmd_log:
    #     cmd_log.writelines('\n'.join(all_cmds))
    # subprocess.run(parallel_cmd, shell=True)


@argh.arg("run_name", help="Directory to store the run; will be created if necessary")
@argh.arg("--gpus", help="Total number of GPUs to use")
@argh.arg("--gpc", help="GPUs per machine")
# @argh.arg('-d', "--datasets", nargs='+', type=str, help = "Datasets")
@argh.arg("--epochs", help="Number of epochs to run Pytorch optimizer")
@argh.arg("--batch-size", help="Batch size")
def run(run_name, gpus=1, gpc=1, epochs=1000, batch_size=65536):
    os.makedirs(run_name, exist_ok=True)

    run_pytorch(run_name, gpus=gpus, gpc=gpc, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    _parser = argh.ArghParser()
    _parser.set_default_command(run)
    _parser.dispatch()

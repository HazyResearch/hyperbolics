import argh
import os
import subprocess
import itertools

# ranks = [2,5,10,50,100,200]
ranks = [200]
def run_comb2(run_name, datasets):
    os.makedirs(f"{run_name}/comb_dim2", exist_ok=True)
    params = []
    rank = 2
    epss = [1.0, 0.1]
    precision = 8192
    for dataset, eps in itertools.product(datasets, epss):
        # julia comb.jl -d ../data/edges/phylo_tree.edges -e 1.0 -p 256 -s -r 200 -c -m phylo_tree.save -a
        param = [
            '-d', f"data/edges/{dataset}.edges",
            # '-m', f"data/comb/{dataset}.r{rank}.p{precision}.e{eps}.emb",
            '-m', f"{run_name}/comb_embeddings/{dataset}.r{rank}.p{precision}.e{eps}.emb",
            '-p', str(precision),
            '-e', str(eps),
            '-r', str(rank)
            ]
        params.append(" ".join(param))

    cmd = 'julia combinatorial/comb.jl'
    with open(f"{run_name}/comb.r2.cmds", "w") as cmd_log:
        cmd_log.writelines('\n'.join(params))
    with open(f"{run_name}/comb.r2.log", "w") as log:
        subprocess.run(" ".join(['parallel', ':::', *[f'"{cmd} {p}"' for p in params]]),
                shell=True, stdout=log)


def run_comb(run_name, datasets, precision=256):
    os.makedirs(f"{run_name}/comb_embeddings", exist_ok=True)

    params = []
    for dataset, rank in itertools.product(datasets, ranks):
        # julia comb.jl -d ../data/edges/phylo_tree.edges -e 1.0 -p 256 -s -r 200 -c -m phylo_tree.save -a
        param = [
                '-d', f"data/edges/{dataset}.edges",
                # '-m', f"data/comb/{dataset}.r{rank}.p{precision}.emb",
                '-m', f"{run_name}/comb_embeddings/{dataset}.r{rank}.p{precision}.emb",
                '-p', str(precision),
                '-e', '1.0',
                '-r', str(rank),
                '-a']
        if rank > 10:
            param.append('-c')
        params.append(" ".join(param))

    cmd = 'julia combinatorial/comb.jl'
    with open(f"{run_name}/comb.p{precision}.cmds", "w") as cmd_log:
        cmd_log.writelines('\n'.join(params))
    with open(f"{run_name}/comb.p{precision}.log", "w") as log:
        full_cmd = " ".join(['parallel', ':::', *[f'"{cmd} {p}"' for p in params]])
        print(full_cmd)
        subprocess.run(" ".join(['parallel', ':::', *[f'"{cmd} {p}"' for p in params]]),
                shell=True, stdout=log)


def run_pytorch(run_name, datasets, epochs, batch_size, warm_start=False, comb=False):
    precision = None
    if warm_start:
        # run combinatorial code first in double precision
        precision = 53
        if comb:
            run_comb(run_name, datasets, precision=precision)
    learning_rate = 5

    params = []
    # with open(f"{run_name}/pytorch.params", "w") as param_file:
    #     param_file.writelines("\n".join(params))
    for dataset, rank in itertools.product(datasets, ranks):
        log_w = ".w" if warm_start else ""
        log_name = f"{run_name}/{dataset}{log_w}.r{rank}.log"
        param = [
                f"data/edges/{dataset}.edges",
                '--log-name', log_name,
                '--batch-size', str(batch_size),
                '--epochs', str(epochs),
                '-r', str(rank),
                '--checkpoint-freq', '100',
                '--use-svrg',
                '-T 0',
                # '--subsample 2000',
                '--learning-rate', str(learning_rate)]
        if warm_start:
            param += ['--warm-start', f"{run_name}/comb_embeddings/{dataset}.r{rank}.p{precision}.emb"]
        params.append(" ".join(param))

    cmd = " ".join([ 'CUDA_VISIBLE_DEVICES=0', 'python', 'pytorch/pytorch_hyperbolic.py', 'learn' ])
    # print(*[f'"{cmd} {p}"' for p in params])
    # subprocess.run(['parallel',
    #         ':::',
    #         *[f'"{cmd} {p}"' for p in params]
    #         ], shell=True)

    parallel_cmd = " ".join(['parallel',
            ':::',
            *[f'"{cmd} {p}"' for p in params]
            ])
    print(parallel_cmd)
    subprocess.run(parallel_cmd, shell=True)


@argh.arg("run_name", help="Directory to store the run; will be created if necessary")
@argh.arg('-d', "--datasets", nargs='+', type=str, help = "Datasets")
@argh.arg("--epochs", help="Number of epochs to run Pytorch optimizer")
@argh.arg("--batch-size", help="Batch size")
def run(run_name, datasets=[], epochs=5000, batch_size=1024):
    os.makedirs(run_name, exist_ok=True)

    # combinatorial high dim
    # run_comb(run_name, datasets)
    # 2d combinatorial
    # run_comb2(run_name, datasets)
    # pytorch by itself
    run_pytorch(run_name, datasets, epochs=epochs, batch_size=batch_size, warm_start=False)
    # pytorch with warmstart
    run_pytorch(run_name, datasets, epochs=epochs, batch_size=batch_size, warm_start=True, comb=True)



if __name__ == '__main__':
    _parser = argh.ArghParser() 
    # _parser.add_commands([build])
    # _parser.dispatch()
    _parser.set_default_command(run)
    _parser.dispatch()

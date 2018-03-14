import argh
import os
import subprocess
import itertools

datasets = ["phylo_tree"]
ranks = [2,5,10,50,100,200]
def run_comb(run_name):
    params = []
    for dataset, rank in itertools.product(datasets, ranks):
        # julia comb.jl -d ../data/edges/phylo_tree.edges -e 1.0 -p 256 -s -r 200 -c -m phylo_tree.save -a
        param = [
            '-d', f"data/edges/{dataset}.edges",
            '-m', f"data/comb/{dataset}.r{rank}.emb",
            '-p', str(256),
            '-e', '1.0',
            '-r', str(rank),
            '-a', '-s']
        if rank > 10:
            param.append('-c')
        params.append(" ".join(param))

    cmd = 'julia combinatorial/comb.jl'
    with open(f"{run_name}/comb.log", "w") as log:
        subprocess.run(" ".join(['parallel', ':::', *[f'"{cmd} {p}"' for p in params]]),
                shell=True, stdout=log)


def run_pytorch(run_name, epochs, batch_size):
    os.makedirs(run_name, exist_ok=True)

    params = []
    # with open(f"{run_name}/pytorch.params", "w") as param_file:
    #     param_file.writelines("\n".join(params))
    for dataset, rank in itertools.product(datasets, ranks):
        log_name = f"{run_name}/{dataset}.r{rank}.log"
        params.append(" ".join([
            f"data/edges/{dataset}.edges",
            '--warm-start', f"data/comb/{dataset}.r{rank}.emb",
            '--log-name', log_name,
            '--batch-size', str(batch_size),
            '--epochs', str(epochs),
            '-r', str(rank),
            '--checkpoint-freq', '100',
            '--learning-rate', '5']))


    cmd = " ".join([ 'CUDA_VISIBLE_DEVICES=0', 'python', 'pytorch/pytorch_hyperbolic.py', 'learn' ])
    print(*[f'"{cmd} {p}"' for p in params])
    # subprocess.run(['parallel',
    #         ':::',
    #         *[f'"{cmd} {p}"' for p in params]
    #         ], shell=True)

    subprocess.run(" ".join(['parallel',
            ':::',
            *[f'"{cmd} {p}"' for p in params]
            ]), shell=True)


@argh.arg("run_name", help="Directory to store the run")
@argh.arg("--epochs", help="Number of epochs to run")
@argh.arg("--batch-size", help="Batch Size")
def run(run_name, epochs=500, batch_size=1024):
    # TODO: only run this if files don't already exist
    # run_comb(run_name)
    run_pytorch(run_name, epochs=epochs, batch_size=batch_size)


# with open(f"{run_name}/{exp_name}", "w") as output_file:
# with open(run_name + '/' + exp_name, "w") as output_file:
    # subprocess.run(['CUDA_VISIBLE_DEVICES="1"',
    #                 'python', 'pytorch/pytorch_hyperbolic.py',
    #                 'learn', f"data/edges/{dataset}.edges",
    #                 '--log-name', f"{run_name}/{exp_name}",
    #                 '--batch-size', str(batch_size),
    #                 '--epochs', str(epochs),
    #                 '-r', str(rank),
    #                 '--checkpoint-freq', '100',
    #                 '--learning-rate', '5'],
    #                 # stderr=output_file
    # )

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    # _parser.add_commands([build])
    # _parser.dispatch()
    _parser.set_default_command(run)
    _parser.dispatch()

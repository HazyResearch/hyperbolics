import argh, os
from collections import defaultdict

#cat run_file.sh | parallel -P 4 "source path.src; bash -c {}"


def work_command(run_name, dataset, rank, gpu, batch_size, epochs, scale):
    run_stem = f"{run_name}/dataset_{dataset}.r={rank}"
    exec_str = f"CUDA_VISIBLE_DEVICES=\"{gpu}\" python pytorch/pytorch_hyperbolic.py learn {dataset} -s {scale} --model-save-file {run_stem}.model  -r {rank} --batch-size {batch_size} --epochs {epochs} --log-name {run_stem}.log --print-freq 10"
    return exec_str


def get_scale_dict(scale_file):
     with open(scale_file) as fh: ls = fh.readlines()
     d = dict()
     for l in ls:
         l.strip()
         k,v = l.strip().split("\t")
         d[k] = v
     return d
 
@argh.arg("run_name", help="Director to store the run")
@argh.arg("--epochs", help="Number of epochs to run")
@argh.arg("--batch-size", help="Batch Size")
@argh.arg("--gpus", help="Number of GPUS")
@argh.arg("--nParallel", help="Number of Concurrent jobs")
@argh.arg("--scale-file", help="File with dictionary of scalings for datatsets")
def build(run_name, epochs=100, batch_size=16384, gpus=2, nParallel=3, scale_file="scripts/scale_eps_1.txt"):
    os.mkdir(run_name)

    scale_dict = get_scale_dict(scale_file)
    cmds       = defaultdict(list)
    for dataset in [12,13,6,7,11]:
          gpu = dataset % gpus
          for rank in [2,5,10,50,100,200]:
            cmds[gpu].append(work_command(run_name, dataset, rank, gpu, batch_size, epochs, scale_dict[str(dataset)]))

    cmd_files = []
    for gpu in range(gpus):
          fname = f"{run_name}/run.{gpu}.cmds"
          with open(fname,"w") as fh:
             fh.writelines("\n".join(cmds[gpu]))
          cmd_files.append(fname)

    exec_cmd = "\"source path.src; bash -c {}\""
    with open(f"{run_name}/drive.sh", "w") as fh:
        cmds = []
        for cmd_f in cmd_files:
             cmd = f"cat {cmd_f} | parallel --gnu -P {nParallel} {exec_cmd}"
             cmds.append(cmd)
        fh.writelines("\n".join(cmds))

    with open(f"{run_name}/main.sh", "w") as fh:
        fh.writelines(f"cat {run_name}/drive.sh | parallel --gnu -P {gpus} {exec_cmd}")

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    _parser.add_commands([build])
    _parser.dispatch()

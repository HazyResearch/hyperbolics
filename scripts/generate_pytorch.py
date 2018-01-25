import argh, os
from collections import defaultdict

#cat run_file.sh | parallel -P 4 "source path.src; bash -c {}"


def work_command(run_name, dataset, rank, gpu, batch_size, epochs):
    run_stem = f"{run_name}/dataset_{dataset}.r={rank}"
    exec_str = f"CUDA_VISIBLE_DEVICES=\"{gpu}\" python pytorch/pytorch_hyperbolic.py learn {dataset} --model-save-file {run_stem}.model  -r {rank} --batch-size {batch_size} --epochs {epochs} --log-name {run_stem}.log"
    return exec_str

@argh.arg("run_name", help="Director to store the run")
@argh.arg("--epochs", help="Number of epochs to run")
@argh.arg("--batch-size", help="Batch Size")
@argh.arg("--gpus", help="Number of GPUS")
def build(run_name, epochs=100, batch_size=16384, gpus=2, nParallel=3):
    os.mkdir(run_name)

    cmds = defaultdict(list)
    for dataset in range(1,13):
          gpu = dataset % gpus
          for rank in [2,5,10,50,100,200]:
            cmds[gpu].append(work_command(run_name, dataset, rank, gpu, batch_size, epochs))

    cmd_files = []
    for gpu in range(gpus):
          fname = f"{run_name}/run.{gpu}.cmds"
          with open(fname,"w") as fh:
             fh.writelines("\n".join(cmds[gpu]))
          cmd_files.append(fname)
          
    with open(f"{run_name}/drive.sh", "w") as fh:
        cmds = []
        for cmd_f in cmd_files:
             exec_cmd = "\"source path.src; bash -c {}\" &"
             cmd = f"cat {cmd_f} | parallel -P {nParallel} {exec_cmd}"
             cmds.append(cmd)
        fh.writelines("\n".join(cmds))
          
if __name__ == '__main__':
    _parser = argh.ArghParser() 
    _parser.add_commands([build])
    _parser.dispatch()

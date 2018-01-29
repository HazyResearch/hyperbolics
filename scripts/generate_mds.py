import argh, os

def work_command(run_name, dataset, rank, scale, prec, tol):
    run_stem = f"{run_name}/dataset_{dataset}.r={rank}"
    exec_str = f" julia mds-scale.jl {dataset} {rank} {scale} {prec} {tol} > {run_stem}.log"
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
@argh.arg("--prec", help="Precision")
@argh.arg("--max-k", help="Max-k")
@argh.arg("--nParallel", help="Parallel")
@argh.arg("--scale-file", help="Scale File")
def tri(run_name, prec="2048", max_k=200, nParallel=6, scale_file="scripts/scale_eps_1.txt"):
    os.mkdir(run_name)
    scale_dict = get_scale_dict(scale_file)
    cmds       = list()
    for dataset in range(1,13):
        scale = scale_dict[str(dataset)]
        cmds.append(f"julia serialize_helper.jl --prec {prec} --max_k {max_k} --scale {scale} {dataset} {run_name}/tri.{dataset}.jld --stats-file {run_name}/tri.{dataset}.stats")
        
    fname = f"{run_name}/tri.run.cmds"
    with open(fname,"w") as fh:
        fh.writelines("\n".join(cmds))

    exec_cmd = "\"source path.src; bash -c {}\""
    with open(f"{run_name}/main.sh", "w") as fh:
        fh.writelines(f"cat {run_name}/tri.run.cmds | parallel --gnu -P {nParallel} {exec_cmd}")

@argh.arg("run_name", help="Director to store the run")
@argh.arg("--prec", help="Precision")
@argh.arg("--tol", help="Tolerance")
def build(run_name, prec="2048", tol="100"):
    os.mkdir(run_name)
    scale_dict = get_scale_dict()
    cmds       = list()
    for dataset in range(12,13):
        for rank in [2,5,10,50,100,200]:
            cmds.append(work_command(run_name, dataset, rank, scale_dict[str(dataset)], prec, tol))

    cmd_files = []
    fname = f"{run_name}/run.cmds"
    with open(fname,"w") as fh:
        fh.writelines(cmds)
    cmd_files.append(fname)

    exec_cmd = "\"source path.src; bash -c {}\""
    with open(f"{run_name}/drive.sh", "w") as fh:
        cmds = []
        for cmd_f in cmd_files:
             cmd = f"cat {cmd_f} | {exec_cmd}"
             cmds.append(cmd)
        fh.writelines("\n".join(cmds))

    with open(f"{run_name}/main.sh", "w") as fh:
        fh.writelines(f"cat {run_name}/drive.sh | {exec_cmd}")
                
                
if __name__ == '__main__':
    _parser = argh.ArghParser() 
    _parser.add_commands([build, tri])
    _parser.dispatch()

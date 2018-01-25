import argh, os

@argh.arg("run_name", help="Director to store the run")
@argh.arg("--epochs", help="Number of epochs to run")
@argh.arg("--batch-size", help="Batch Size")
def build(run_name, epochs=100, batch_size=16384):
    os.mkdir(run_name)

    for dataset in range(1,13):
        for rank in [2,5,10,50,100,200]:
            run_stem = f"{run_name}/dataset_{dataset}.r={rank}"
            exec_str = f"CUDA_VISIBLE_DEVICES=\"1\" python pytorch/pytorch_hyperbolic.py learn {dataset} --model-save-file {run_stem}.model  -r {rank} --batch-size {batch_size} --epochs {epochs} --log-name {run_stem}.log"
            print(exec_str)

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    _parser.add_commands([build])
    _parser.dispatch()

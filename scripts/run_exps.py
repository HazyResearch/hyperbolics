import argh
import os
import subprocess

@argh.arg("run_name", help="Directory to store the run")
@argh.arg("--epochs", help="Number of epochs to run")
@argh.arg("--batch-size", help="Batch Size")
def run(run_name, epochs=100, batch_size=64):
    os.makedirs(run_name, exist_ok=True)

    dataset = "phylo_tree"
    exp_name = dataset
    rank = 100
    # with open(f"{run_name}/{exp_name}", "w") as output_file:
    # with open(run_name + '/' + exp_name, "w") as output_file:
    subprocess.run(['python', 'pytorch/pytorch_hyperbolic.py',
                    'learn', f"data/edges/{dataset}.edges",
                    '--log-name', f"{run_name}/{exp_name}",
                    '--batch-size', str(batch_size),
                    '--epochs', str(epochs),
                    '-r', str(rank),
                    '--checkpoint-freq', '100',
                    '--learning-rate', '5'],
                    # stderr=output_file
    )

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    # _parser.add_commands([build])
    # _parser.dispatch()
    _parser.set_default_command(run)
    _parser.dispatch()

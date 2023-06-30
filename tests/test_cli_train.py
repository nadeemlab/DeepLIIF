import subprocess
import os

def test_cli_train(tmp_path):
    dir_input = 'Datasets/Sample_Dataset' # cli.py train looks for subfolder "train" under dataroot
    dir_save = tmp_path
    
    fns_input = [f for f in os.listdir(dir_input + '/train') if os.path.isfile(os.path.join(dir_input + '/train', f)) and f.endswith('png')]
    num_input = len(fns_input)
    assert num_input > 0
    
    res = subprocess.run(f'python cli.py train --dataroot {dir_input} --name test_local --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1',shell=True)
    assert res.returncode == 0
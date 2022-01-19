import os
#root_folder='/mnts/deepliif-data/DeepLIIF_Datasets'#os.getenv("DATA_DIR")
root_folder='/mnts/DeepLIIFData/DeepLIIF_Datasets/'#os.getenv("DATA_DIR")
# output_model_folder = os.environ["RESULT_DIR"]
# output_model_path = os.path.join(output_model_folder,"model")
# output_model_path_file = os.path.join(output_model_path,"trained_model.pt")
# output_model_path_onnx = os.path.join(output_model_path,"trained_model.onnx")


import subprocess
print('Install packages...')
#subprocess.run('pip install dominate visdom numba gpustat --user; pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html',shell=True)
subprocess.run('pip install dominate visdom numba gpustat --user', shell=True)

# print('Detecting keyword DLI in env vars from python...')
# subprocess.run('env | grep DLI | wc -l',shell=True)

# print('Starting gpu monitor...')
# subprocess.Popen('bash monitor_gpu.sh',shell=True)


if __name__ == '__main__':
    print('-------- os.environ ---------')
    print(os.environ)
    print('-------- ls -lh --------')
    subprocess.run('ls -lh',shell=True)
    print('-------- start training... --------')
    #subprocess.run(f'python cli.py train --dataroot {root_folder} --name Test_Model --remote True --pickle-transfer-cmd custom_save.save_to_storage_volume --gpu-ids 0',shell=True)
    #subprocess.run(f'python cli.py train --dataroot {root_folder} --name Test_Model --remote True --checkpoints-dir /mnts/DeepLIIFData/checkpoints/ --gpu-ids 0',shell=True)
    subprocess.run(f'torchrun \
                            -t 3 \
                            --nproc_per_node 2 \
                            cli.py train \
                            --dataroot {root_folder} \
                            --name local_ddp_test \
                            --checkpoints-dir /mnts/DeepLIIFData/checkpoints/ \
                            --batch-size 2 \
                            --gpu-ids 0 \
                            --remote True',shell=True)

import os
import shutil


image_extensions = ['png', 'jpg', 'tif']

DeepLIIF_path = '/media/parmida/Work/DeepLIIF'   # path to the DeppLIIF directory

# input_path = '/media/parmida/Work/Marker_Datasets/TMA'   # path to the folder containing images
input_path = '/home/parmida/Desktop/Tissues4'   # path to the folder containing images
# output_images_path = '/media/parmida/Work/Marker_Datasets/TMA_modalities'   # path to save the output images (modalities)
output_images_path = '/media/parmida/Work/Marker_Datasets/Tissues_Results2'   # path to save the output images (modalities)

python_run_path = '/home/parmida/miniconda3/envs/pytorch_env/bin/python'    # path to the python run file


test_file_path = os.path.join(DeepLIIF_path, 'test.py')
preprocessing_file_path = os.path.join(DeepLIIF_path, 'preprocessing.py')
postprocessing_file_path = os.path.join(DeepLIIF_path, 'postprocessing.py')

# model_name = 'deepLIIF_model'
model_name = 'DeepLIIF_Model'
# checkpoints_path = '/home/parmida/pytorch_multi_task_pix2pix/checkpoints'
checkpoints_path = os.path.join(DeepLIIF_path, 'checkpoints')
# results_path = '/home/parmida/pytorch_multi_task_pix2pix/results'
results_path = os.path.join(DeepLIIF_path, 'results')

model_output_path = os.path.join(results_path, model_name, 'test_latest', 'images')


tile_size = 200
overlap_size = 20
resize_self = True
resize_size = None

parent_dir = os.path.dirname(input_path)
directories = [x[0] for x in os.walk(input_path)]
print(directories)
for directory in directories:
    images = os.listdir(directory)
    does_contain_image = False
    for img_name in images:
        if img_name.split('.')[-1] in image_extensions:
            does_contain_image = True
            break
    if does_contain_image:
        crops_dir = os.path.join(parent_dir, 'Dataset', 'test')

        if not os.path.exists(crops_dir):
            os.makedirs(crops_dir)
        os.system(
            python_run_path + ' ' + preprocessing_file_path +
            ' --input_dir ' + directory +
            ' --output_dir ' + crops_dir +
            ' --tile_size ' + str(tile_size) +
            ' --overlap_size ' + str(overlap_size) +
            ' --resize_self ' + str(resize_self) +
            ' --resize_size ' + str(resize_size)
        )
        os.system(
            python_run_path + ' ' + test_file_path +
            ' --dataroot ' + os.path.join(parent_dir, 'Dataset') +
            ' --name ' + model_name +
            ' --model ' + 'DeepLIIF' +
            ' --netG ' + 'resnet_9blocks' +
            ' --checkpoints_dir ' + checkpoints_path +
            ' --results_dir ' + results_path
        )
        os.system(
            python_run_path + ' ' + postprocessing_file_path +
            ' --input_dir ' + model_output_path +
            ' --output_dir ' + directory +
            ' --input_orig_dir ' + directory +
            ' --tile_size ' + str(tile_size) +
            ' --overlap_size ' + str(overlap_size) +
            ' --resize_self ' + str(resize_self) +
            ' --resize_size ' + str(resize_size)
        )
        shutil.rmtree(os.path.join(results_path, model_name))
        shutil.rmtree(os.path.join(parent_dir, 'Dataset'))

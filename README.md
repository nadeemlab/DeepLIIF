# DeepLIIF
DeepLIIF project

**Preprocessing:**
python preprocessing.py **--input_di**r /path/to/input/images **--output_dir** /path/to/output/images **--tile_size** size_of_each_cropped_tile **--overlap_size** overlap_size_between_crops **--resize_self** if_True_resizes_to_the_closest_rectangle_if_False_resize_size_need_to_be_set **--resize_size**

**Postprocessing:**
python postprocessing.py **--input_dir** /path/to/preprocessed/images **--output_dir** /path/to/output/images **--input_orig_dir** /path/to/original/input/images --tile_size size_of_each_cropped_tile **--overlap_size** overlap_size_between_crops **--resize_self** if_True_resizes_to_the_closest_rectangle_if_False_resize_size_need_to_be_set **--resize_size**

**Training:**
python train.py **--dataroot** /path/to/input/images **--name** model_name **--model** DeepLIIF **--netG** resnet_9blocks 

**Testing:**
python test.py **--dataroot** /path/to/input/images **--name** model_name **--model** DeepLIIF **--netG** resnet_9blocks 

**Demo:**
Change the DeepLIIF_path to the path of DeepLIIF project. Set input_path to the directory containing the input images and python_run_path to the path of python run file. It saves the modalities to the input directory next to each image.

**More options?**
You can find more options in the options/base_option.py and option/train_options.py and option/test_options.py

# Testing

## Serialize Model
The installed `deepliif` uses Dask to perform inference on the input IHC images.
Before running the `test` command, the model files must be serialized using Torchscript.
To serialize the model files:
```
deepliif serialize --models-dir /path/to/input/model/files
                   --output-dir /path/to/output/model/files
```
* By default, the model files are expected to be located in `DeepLIIF/model-server/DeepLIIF_Latest_Model`.
* By default, the serialized files will be saved to the same directory as the input model files.

## Testing
To test the model:
```
deepliif test --input-dir /path/to/input/images 
              --output-dir /path/to/output/images 
              --tile-size 512
```
* The latest version of the pretrained models can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).
* Before running test on images, the model files must be serialized as described above.
* The serialized model files are expected to be located in `DeepLIIF/model-server/DeepLIIF_Latest_Model`.
* The test results will be saved to the specified output directory, which defaults to the input directory.
* The default tile size is 512.
* Testing datasets can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).

If you prefer, it is possible to run the model using Torchserve.
Please see below for instructions on how to deploy the model with Torchserve and for an example of how to run the inference.
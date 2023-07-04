# Notes for Code Testing
The implemented testing uses `pytest`, consisting of the following files:
```
- conftest.py
- tests
  |
  - test_args.py
  - test_cli_inference.py
  - test_cli_train.py
  - test_model_download.py
```
Constant variables that will be shared across test cases are stored as fixtures, specified in `conftest.py`.

|Test Env|Time|
|--------|----|
|Local|00:11:24|

Added arguments:
- `model_type`: the type of model to use, currently only allows "latest"; at the moment this is not used if `model_dir` is specified
- `model_dir`: the directory to your local model files if you have, using which can avoid the time downloading the models from Zenodo (10min+)

Note that **in order to specify `model_dir`, you have to specify `model_type`**. This seems to be some issue in `pytest` custom arguments.

Added test cases:
- test_args.py: tests the access to argument values as fixture
- test_model_download.py: tests the access to remote model if local model directory is not provided, by checking if the final model directory for testing is valid
- test_cli_inference.py: tests if `cli.py test` a) can run, and b) yields consistent predictions across two trials
- test_cli_train.py: tests if `cli.py train` can run on a) cpu, b) single gpu, and c) multiple gpus (DP)

## To Run Testing...
### ... Locally
In `DeepLIIF` folder, execute the following command:
```
pytest -v -s --model_type latest --model_dir <your local model dir>
```
This will use the model files you provided to run all the test cases.

The default temporary directory is `/tmp`. Use option `--basetemp` for a different temporary directory like the following:
```
pytest -v -s --basetemp=../tmp/ --model_type latest --model_dir <your local model dir>
```

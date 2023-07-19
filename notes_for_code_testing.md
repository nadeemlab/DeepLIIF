# Notes for Code Testing
The implemented testing uses `pytest`, consisting of the following files:
```
- conftest.py
- tests
  |
  - test_args.py
  - test_cli_train.py
  - test_cli_serialize.py
  - test_cli_inference.py
```
Constant variables that will be shared across test cases are stored as fixtures, specified in `conftest.py`.

The **automatic testing** with GitHub Actions is not enabled due to the need of GPUs in testing.

Added arguments:
- `model_type`: the type of model to use, currently allows "latest" or "ext"
- `model_dir`: the directory to your local model files if you have, using which can avoid the time downloading the models from Zenodo (10min+)

Note that **in order to specify `model_dir`, you have to specify `model_type`**. This seems to be some issue in `pytest` custom arguments.

|Model Type|Example Execution Time|
|--------|----|
|latest|13:18|
|ext|38:17|

Added test cases:
- test_args.py: tests the access to argument values as fixture
- test_cli_train.py: tests if `cli.py train` can run on a) CPU, b) single GPU, and c) two GPUs (data parallel)
- test_cli_serialize.py: tests if `cli.py serialize` can run
- test_cli_inference.py: tests if `cli.py test` a) can run, and b) yields consistent predictions across two trials or two approaches (serialized model files vs. original model files)

## To Run Testing...
### ... Locally
In `DeepLIIF` folder, execute the following command:
```
pytest -v -s --model_type latest --model_dir <your local model dir>
```
This will use the model files you provided to run all the test cases.

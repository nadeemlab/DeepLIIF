import subprocess

def test_cli_serialize(tmp_path, model_dir_final, model_info):
    dir_model = model_dir_final
    dir_output = tmp_path
    
    res = subprocess.run(f'python cli.py serialize --models-dir {dir_model} --output-dir {dir_output}',shell=True)
    assert res.returncode == 0
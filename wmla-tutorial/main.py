"""
Utility script to be run in a training job to download files from COS into the WMLA storage.
Your code folder (submitted as .zip file to WMLA) must contain both this script, and a JSON file
cos_credentials.json containing an access key, secret key, bucket name,
and the correct public endpoint to connect to this bucket.
"""
import os

# STEP 0 - IDENTIFY WHERE THE DATA WILL BE STORED IN WML-A
DATA_DIR = os.environ['DATA_DIR']
RESULT_DIR = os.environ["RESULT_DIR"]
print(f"Data will be stored in $DATA_DIR {DATA_DIR}")
print("Current content of this folder is:")
print(os.listdir(DATA_DIR))

# os.system(f'mv download.sh {DATA_DIR}/download.sh')
# os.system(f'cd {DATA_DIR}; bash download.sh; rm -rf download.sh')
    
# STEP 5 - LET WMLA KNOW WE SUCCEEDED BY SAVING IN THE /model FOLDER
os.makedirs(os.path.join(RESULT_DIR, 'model'), exist_ok=True)
with open(os.path.join(RESULT_DIR, 'model', 'done.txt'), 'w') as f:
    f.write('Done.')

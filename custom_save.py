import os
import requests
import time
import urllib
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


USER_ACCESS_TOKEN = os.environ['USER_ACCESS_TOKEN']
BASE_URL = os.getenv('BASE_URL', os.getenv('RUNTIME_ENV_APSX_URL'))

volume_display_name = 'DeepLIIFData'

def save_to_storage_volume(path_source, verbose=1, retry=5):
    """
    Given file to path, save to designated storage volume directory.
    retry: number of times to retry if request fails
    """
    path_target_encoded = urllib.parse.quote(path_source,safe='')
    url = f'/zen-volumes/{volume_display_name}/v1/volumes/files/{path_target_encoded}'

    count_trial = 0
    while count_trial < retry:
        count_trial += 1
        res = requests.put(url=BASE_URL+url,
                           headers={'Authorization': 'Bearer ' + USER_ACCESS_TOKEN},
                           files={'upFile':(path_source,open(path_source, 'rb'))},
                           verify=False)

        if res.status_code == 200:
            if verbose > 0:
                print(f'Success: {path_source} is saved to storage volume {volume_display_name} at {path_source}')
                return None
        else:
            print(f'FAILED: status code {res.status_code}')
            if count_trial < retry:
                print(f'FAILED: Unable to save file to storage volume, {res.text}; wait for 3s and retry...')
                time.sleep(3)
            else:
                raise Exception(f'FAILED: Unable to save file to storage volume, {res.text}; maximum retry reached')

#!/usr/bin/env python3
###############################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2017, 2021 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
###############################################################################

from __future__ import print_function

import argparse
import datetime
import getpass
import json
import os
from pathlib import PurePath
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import shutil
from collections import OrderedDict
import base64

import requests
import requests.utils

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

try:
    # for python2
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except:
    pass

try:
    # for python3
    import requests.packages.urllib3
    requests.packages.urllib3.disable_warnings()
except:
    pass

MODEL_DIR_SUFFIX = ".modelDir.tar"

IS_POD = os.path.exists("/var/run/secrets/kubernetes.io")

if IS_POD:
    DEFAULT_DLI_REST_PORT = 9243
else:
    DEFAULT_DLI_REST_PORT = -1

# MAX_HISTORY_LENGTH = 2147483647
MAX_HISTORY_LENGTH = 100000
DEFAULT_HISTORY_LENGTH = 10000
DEFAULT_ANALYZE_HOURS = 2

RP_PADDING_SPACE = "|   "
RP_PADDING_CHILD = "|-- "

class TerminalColor:
    HEADER = '\033[35m'
    BLUE = '\033[34m'
    GREEN = '\033[32m'
    CYAN = '\033[33m'
    RED = '\033[31m'
    LIGHT_BLUE = '\033[94m'
    LIGHT_GREEN = '\033[92m'
    LIGHT_CYAN = '\033[93m'
    LIGHT_RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class OutputFormat:
    JSON = "json"
    CUSTOM_COLUMNS = "custom-columns"

flags = None

unknown_flags = None

DLI_REST = "/platform/rest/deeplearning/v1"
COOKIE_FILE = os.path.expanduser('~/.dlicmd')

ERROR = '[Error]:'
INFO = '[Info]:'
WARN = '[Warn]:'
DEBUG = '[Debug]:'

APP_ACTIVE_STATES = ["NEW", "SUBMITTED", "PENDING_JOB_SCHEDULER", "PENDING_CRD_SCHEDULER", "LAUNCHING", "RUNNING", "RECOVERING", "UNKNOWN"]
APP_NONACTIVE_STATES = ["FINISHED", "KILLED", "ERROR"]

def my_print(level, *args, **kw):
    if level == DEBUG and str(flags.debug_level).lower() != 'debug':
        pass
    else:
        print(level, *args, **kw)

def get_terminal_size():
    try:
        columns, lines = shutil.get_terminal_size()
        return columns, lines
    except:
        pass

    try:
        columns, lines = os.get_terminal_size()
        return columns, lines
    except:
        pass

    try:
        lines, columns = os.popen('stty size', 'r').read().split()
        columns = int(columns)
        lines = int(lines)
        return columns, lines
    except:
        pass

    return 80, 24

def print_line_separator(sep="="):
    columns, lines = get_terminal_size()
    print(sep * columns)

def strip_path(p, sep=os.path.sep):
    ret = p
    while ret and ret.endswith(sep):
        ret = ret.rstrip(sep)
    return ret

def loads_json_with_order(json_text):
    try:
        return json.loads(json_text, object_pairs_hook=OrderedDict)
    except:
        return json.loads(json_text)


def is_valid_ipv4_address(address):
    try:
        socket.inet_pton(socket.AF_INET, address)
    except AttributeError:  # no inet_pton here, sorry
        try:
            socket.inet_aton(address)
        except socket.error:
            return False
        return address.count('.') == 3
    except socket.error:  # not a valid address
        return False

    return True


def is_valid_ipv6_address(address):
    try:
        socket.inet_pton(socket.AF_INET6, address)
    except socket.error:  # not a valid address
        return False
    return True


def which(cmd):
    if cmd:
        try:
            return shutil.which(cmd)
        except AttributeError:
            # python 2
            path_list = os.environ.get("PATH")
            if not path_list:
                return None
            path_list = path_list.split(os.pathsep)
            for path in path_list:
                file_name = str(PurePath(path, cmd))
                if os.path.exists(file_name) and os.access(file_name, os.X_OK) and not os.path.isdir(file_name):
                    return file_name

    return None

def find_kubectl():
    if which("oc"):
        return "oc"

    if which("kubectl"):
        return "kubectl"

    # this is wmla master pod
    tmp_kubectl = "/tmp/kubedir/kubectl"
    if os.path.exists(tmp_kubectl) and os.access(tmp_kubectl, os.X_OK) and not os.path.isdir(tmp_kubectl):
        return tmp_kubectl

    return None

def parse_utc_time(date_string):
    if not date_string:
        return None

    for date_format in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"]:
        try:
            return datetime.datetime.strptime(date_string, date_format)
        except ValueError:
            pass

    return None

def get_short_hostname(address):
    #print("get_short_hostname, address=%s" % address)
    if not address:
        return address

    try:
        address = str(address)

        if is_valid_ipv4_address(address) or is_valid_ipv6_address(address):
            return address

        if "." in address:
            return str(address.split(".")[0])

        return address
    except:
        return address

def get_files_size(file_list):
    totoal_size = 0
    try:
        for file in file_list:
            if os.path.isfile(file):
                totoal_size += os.path.getsize(file)
    except:
        pass

    return totoal_size


def xpath_get(mydict, path, default=None, sep="/", ignore_case=False):
    elem = mydict
    try:
        for x in path.strip(sep).split(sep):
            try:
                x = int(x)
                elem = elem[x]
            except ValueError:
                if ignore_case:
                    target = None
                    for k, v in elem.items():
                        if k.lower() == x.lower():
                            target = v
                            break
                    if target:
                        elem = target
                    else:
                        return default
                else:
                    elem = elem.get(x, default)
    except:
        return default

    return elem

def timestamp_to_string(timestamp):
    if not timestamp:
        return ""
    
    timestamp = timestamp if isinstance(timestamp, int) else int(timestamp)
    if timestamp > 0:
        date_time = datetime.datetime.utcfromtimestamp(timestamp)
        time_str = date_time.strftime("%m-%d %H:%M")
        return time_str
    else:
        return ""

def pprint_json(json_object):
    json_formatted_str = json.dumps(json_object, indent=2)
    print(json_formatted_str)

def pprint_resp(resp):
    try:
        body = loads_json_with_order(resp.text)
        pprint_json(body)
    except:
        if resp.content:
            print(resp.content.decode("utf-8"))
        #print(resp.text)

def parse_token(token):
    token_json = None
    try:
        token_parts = token.split('.')
        token_padded = token_parts[1] + '=' * (len(token_parts[1]) % 4)
        token_json = json.loads(base64.b64decode(token_padded).decode('utf-8'))
    except:
        pass

    return token_json

def get_token_username(token):
    ret = ""

    token_json = parse_token(token)
    if token_json:
        if "username" in token_json:
            ret = token_json["username"]
        elif "sub" in token_json:
            ret = token_json["sub"]

    return ret

def get_token_expiration_datetime(token):
    ret = None
    token_json = parse_token(token)
    if token_json:
        ret = token_json.get('exp')
    return ret

def get_custom_columns():
    ret = None
    if flags.output_format and flags.output_format.startswith(OutputFormat.CUSTOM_COLUMNS + "="):
        columns = flags.output_format[len(OutputFormat.CUSTOM_COLUMNS + "="):]
        if columns:
            ret = columns.split(",")

    return ret

def print_table(columns, rows, na_rep='n/a', col_padding_len=2, show_header=True, show_index=True, show_h_border=True, h_border="-", callback=None):
    if not columns:
        raise Exception("columns can not be empty")

    if rows is None:
        rows = []

    col_row_index_len = len(str(len(rows))) + col_padding_len

    # init col_max_len_dict
    col_max_len_dict = {}
    for i in range(len(columns)):
        col_name = columns[i]
        if not col_name:
            raise Exception("column name can not be empty")

        col_max_len_dict[col_name] = len(col_name)
        if i != len(columns) - 1:
            col_max_len_dict[col_name] += col_padding_len

    # update col_max_len_dict
    for row in rows:
        if row is None:
            raise Exception("row can not be None")

        for i in range(len(columns)):
            col_name = columns[i]
            col_max_len = col_max_len_dict[col_name]
            col_val = row.get(col_name)
            if col_val is None:
                col_val = na_rep

            col_len = len(str(col_val)) if col_val else 0
            if i != len(columns) - 1:
                col_len += col_padding_len

            if col_len > col_max_len:
                col_max_len_dict[col_name] = col_len

    # calculate horizon borders length
    col_total_len = 0
    if show_h_border:
        if show_index:
            col_total_len += col_row_index_len

        for col_name, col_max_len in col_max_len_dict.items():
            col_total_len += col_max_len

        column_size, _ = get_terminal_size()
        # draw H border
        col_total_len = col_total_len if col_total_len < column_size else column_size
        print(h_border * col_total_len)

    # draw header
    if show_header:
        header = ""

        if show_index:
            header += col_row_index_len * ' '

        for col_name in columns:
            col_max_len = col_max_len_dict[col_name]
            header += col_name.ljust(col_max_len, ' ')
        print(header)

    # draw rows
    row_index = 1
    for row in rows:
        line = ""

        if show_index:
            line += str(row_index).ljust(col_row_index_len, ' ')

        for col_name in columns:
            col_max_len = col_max_len_dict[col_name]
            col_val = row.get(col_name)
            if col_val is None:
                col_val = na_rep

            line += str(col_val).ljust(col_max_len, ' ')
        print(line)
        if callback:
            callback(row)
        row_index += 1

    if show_h_border:
        print(h_border * col_total_len)


def get_status():
    my_print(DEBUG, "get_status()")

    req_args = get_request_args()
    resp = requests.get(flags.dli_rest_url + "/status", **req_args)

    pprint_resp(resp)
    
    if not resp.ok:
        sys.exit(1)
    
# display all supported frameworks in json array form to stdout
#
def list_framework():
    my_print(DEBUG, "list_framework()")

    req_args = get_request_args()
    
    url = flags.dli_rest_url + "/execs/frameworks"
    my_print(DEBUG, url)
    
    resp = requests.get(url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    frameworks = json.loads(resp.text)

    if flags.output_format == OutputFormat.JSON:
        pprint_json(frameworks)
    else:
        for framework in frameworks:
            print("Name: %s" % framework.get("name", ""))
            print("Version: %s" % framework.get("frameworkVersion", ""))
            desc = framework.get("desc", None)
            if desc:
                for i in range(0, len(desc)):
                    if i == 0:
                        print("Desc: %s" % desc[i])
                    else:
                        print("      %s" % desc[i])
            print("")


def get_batch(batch_id):
    my_print(DEBUG, "get_batch: batchId='%s'" % batch_id)
    req_args = get_request_args()

    url = flags.dli_rest_url + "/execs/" + batch_id
    my_print(DEBUG, url)
    
    resp = requests.get(url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    # batch = json.loads(resp.text)
    batch = loads_json_with_order(resp.text)

    my_print(DEBUG, batch)
    return batch

def get_age(t_start, t_end):
    # t_start and t_end should be timestamp or utc datetime
    if t_start and t_end:
        t_start = datetime.datetime.utcfromtimestamp(t_start) if isinstance(t_start, int) else t_start
        t_end = datetime.datetime.utcfromtimestamp(t_end) if isinstance(t_end, int) else t_end
        #print("%s " % str(t_start))
        #print("%s " % str(t_end))

        duration = t_end - t_start
        duration_in_s = duration.total_seconds()
        years = divmod(duration_in_s, 86400 * 365.2425)
        days = divmod(duration_in_s, 86400)
        hours = divmod(days[1], 3600)
        minutes = divmod(hours[1], 60)
        seconds = divmod(minutes[1], 1)

        if years[0] > 0:
            age = str(int(years[0])) + "y," + str(int(days[0])) + "d"
        elif days[0] > 0:
            age = str(int(days[0])) + "d," + str(int(hours[0])) + "h"
        elif hours[0] > 0:
            age = str(int(hours[0])) + "h," + str(int(minutes[0])) + "m"
        elif minutes[0] > 0:
            age = str(int(minutes[0])) + "m," + str(int(seconds[0])) + "s"
        else:
            age = str(int(seconds[0])) + "s"

        return age
    else:
        return ""

def batch_to_row_dict(batch):
    row_dict = {}

    row_dict["id"] = batch.get('id')
    row_dict["submit_user"] = batch.get('creator')
    row_dict["name"] = batch.get('appName')
    row_dict["framework"] = batch.get('framework')
    
    if batch.get('elastic'):
        row_dict["type"] = "ELASTIC"
    else:
        row_dict["type"] = "PARALLEL"
    
    row_dict["state"] = batch.get('state')
    
    row_dict["num"] = batch.get('numWorker')
    
    row_dict["submit_time"] = parse_utc_time(batch.get('createTime'))
    
    t_start = parse_utc_time(batch.get('createTime'))
    row_dict["age"] = get_age(t_start, datetime.datetime.utcnow())

    app_id = batch.get('appId')
    if not app_id:
        app_id = batch.get('submissionId')
    row_dict["app_id"] = batch.get('app_id')
    
    return row_dict

def list_batch():
    my_print(DEBUG, "list_batch()")
    req_args = get_request_args()
    
    url = flags.dli_rest_url + "/execs/"
    if flags.query_args:
        url = url + "?" + flags.query_args
    my_print(DEBUG, url)
    
    resp = requests.get(url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    batches = json.loads(resp.text)

    if flags.output_format == OutputFormat.JSON:
        pprint_json(batches)
    else:
        columns = get_custom_columns()
        if not columns:
            columns = ["id", "submit_user", "name", "framework", "type", "state", "num", "submit_time", "age"]

        rows = []
        for batch in batches:
            # print(batch)
            row_dict = batch_to_row_dict(batch)
            rows.append(row_dict)

        print_table(columns, rows)

def get_paralleljob_info(batch_info):
    state = batch_info.get('state')
    if state in APP_NONACTIVE_STATES:
        return None  # there's no pj to be queried

    ret = {}

    pj_id = batch_info.get('appId', "")

    if not pj_id:
        return ret

    req_args = get_request_args()

    url = flags.dli_rest_url + "/scheduler/kubernetesobjects?type=%s&name=%s" % ("paralleljob", pj_id)
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)
    if resp.ok:
        pj = json.loads(resp.text)
        #pprint_json(items)
    else:
        return ret

    # pprint.pprint(pj)
    ret['status'] = {}
    ret['status']['jobStatus'] = xpath_get(pj, "/status/jobStatus", "")

    # handling pending job
    if ret['status']['jobStatus'] == 'Pending':
        ret['status']['jobPendingReason'] = xpath_get(pj, "/status/jobPendingReason", "")

        t_start_str = xpath_get(pj, "/status/creationTimestamp", "")
        if t_start_str:
            t_start = parse_utc_time(t_start_str)
            ret['status']['pendingDuration'] = get_age(t_start, datetime.datetime.utcnow())

    t_start_str = xpath_get(pj, "/status/jobStartTime", "")
    if t_start_str:
        t_start = parse_utc_time(t_start_str)
        ret['status']['runningDuration'] = get_age(t_start, datetime.datetime.utcnow())

    taskgrp_list = xpath_get(pj, "/spec/taskGroups")
    spec_list = []
    if taskgrp_list:
        for taskgrp in taskgrp_list:
            spec = {}
            spec_list.append(spec)
            spec['taskGroups'] = {}
            spec['taskGroups']['metadata'] = {}
            spec['taskGroups']['metadata']['name'] = xpath_get(taskgrp, "/metadata/name", "")
            spec['taskGroups']['metadata']['annotations'] = {}
            spec['taskGroups']['metadata']['annotations']['lsf.ibm.com/gpu'] = xpath_get(taskgrp, "#metadata#annotations#lsf.ibm.com/gpu", "", "#")
            spec['taskGroups']['spec'] = {}
            spec['taskGroups']['spec']['replica'] = xpath_get(taskgrp, "/spec/replica", "")

            container_list = xpath_get(taskgrp, "/spec/template/spec/containers")
            containers = container_list[0] if container_list and len(container_list) > 0 else None
            if containers:
                spec['taskGroups']['spec']['template'] = {}
                spec['taskGroups']['spec']['template']['spec'] = {}
                spec['taskGroups']['spec']['template']['spec']['containers'] = {}
                spec['taskGroups']['spec']['template']['spec']['containers']['resources'] = {}
                spec['taskGroups']['spec']['template']['spec']['containers']['resources']['requests'] = {}
                spec['taskGroups']['spec']['template']['spec']['containers']['resources']['limits'] = {}
                spec['taskGroups']['spec']['template']['spec']['containers']['resources']['requests']['memory'] = xpath_get(containers, "/resources/requests/memory", "")

                limit_obj = xpath_get(containers, "/resources/limits", None)
                if limit_obj:
                    spec['taskGroups']['spec']['template']['spec']['containers']['resources']['limits']['memory'] = xpath_get(containers, "/resources/limits/memory", "")

    ret['spec'] = spec_list

    return ret


def get_batch_detail(batch_id):
    my_print(DEBUG, "get_batch_detail()")
    batch = get_batch(batch_id)
    if not batch:
        return

    pj = get_paralleljob_info(batch)
    if pj:
        if 'status' in pj and 'pendingDuration' in pj['status']:
            batch['pendingDuration'] = pj['status']['pendingDuration']
            del (pj['status']['pendingDuration'])

        if 'status' in pj and 'runningDuration' in pj['status']:
            batch['runningDuration'] = pj['status']['runningDuration']
            del (pj['status']['runningDuration'])

        batch['parallelJobInfo'] = pj

    pprint_json(batch)


"""
Get Executor log
"""
def get_batch_exec_log(cmd, batch_id):
    my_print(DEBUG, "get_batch_exec_log: cmd=%s, batch_id=%s" % (cmd, batch_id))
    batch = get_batch(batch_id)
    if batch:
        get_app_logs(cmd, batch.get('appId'))

def get_app_logs(cmd, app_id):
    my_print(DEBUG, "get_app_logs: cmd=%s, app_id=%s" % (cmd, app_id))

    if not app_id:
        sys.exit(1)

    if 'launcher' in cmd:
        log_type = "launcherlog"
    elif 'out' in cmd:
        log_type = "stdout"
    elif 'err' in cmd:
        log_type = "stderr"
    else:
        raise Exception("Not supported command %s" % cmd)

    rest_req_args = get_request_args()
    log_req_args = get_request_args('text/plain')

    url = flags.dli_rest_url + "/scheduler/applications?applicationid=" + app_id
    my_print(DEBUG, url)
    
    resp = requests.get(url, **rest_req_args)
    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    apps = json.loads(resp.text)
    for app in apps:
        task0 = app.get("task0")
        if task0:
            print_line_separator()
            print("Driver %s" % log_type)
            
            url = flags.dli_rest_url + "/scheduler/applications/%s/driver/logs/%s" % (app_id, log_type)
            my_print(DEBUG, url)
            
            resp = requests.get(url, **log_req_args)
            print(resp.text)
            print("")

        task12ns = app.get("task12n")
        if task12ns:
            for i in range(0, len(task12ns)):
                executor_id = i + 1
                print_line_separator()
                print("Executor %d %s" % (executor_id, log_type))
                
                url = flags.dli_rest_url + "/scheduler/applications/%s/executor/%s/logs/%s" % (app_id, str(executor_id), log_type)
                my_print(DEBUG, url)
                
                resp = requests.get(url, **log_req_args)
                print(resp.text)
                print("")

"""
Get train logs
"""
def get_batch_train_log(cmd, batch_id, argv):
    my_print(DEBUG, "get_batch_train_log: cmd=%s, execId=%s" % (cmd, batch_id))

    args = ' '.join(argv)

    if cmd == '--exec-trainoutlogs':
        log_type = "outlog"
    elif cmd == '--exec-trainerrlogs':
        log_type = "errlog"
    else:
        log_type = "outerrlog"

    my_print(DEBUG, "args=%s" % args)

    req_args = get_request_args('text/plain')
    
    url = flags.dli_rest_url + "/execs/%s/log?logType=%s&args=%s" % (batch_id, log_type, args)
    my_print(DEBUG, url)
    
    resp = requests.get(url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    print(resp.text)


"""
Get train result
"""
def get_train_result(batch_id, argv):
    my_print(DEBUG, "get_train_result() batchId=%s" % batch_id)

    args = ' '.join(argv)

    my_print(DEBUG, "args=%s" % args)

    req_args = get_request_args('text/plain')

    url = flags.dli_rest_url + "/execs/%s/result?args=%s" % (batch_id, args)
    my_print(DEBUG, url)
    
    resp = requests.get(url, stream=True, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    filename = 'train_result.zip'
    if resp.headers["Content-Disposition"]:
        key = "filename="
        for item in resp.headers["Content-Disposition"].split(";"):
            if item.startswith("filename="):
                filename = item[len(key):]
                break

    with open(filename, 'wb') as fd:
        for chunk in resp.iter_content(chunk_size=128):
            fd.write(chunk)

    #print(r.headers)
    #with open(filename, 'wb') as out_file:
        #shutil.copyfileobj(r.raw, out_file)
    print("Train result is saved to %s" % filename)


def stop_batch(batch_id):
    my_print(DEBUG, "stop_batch()")

    req_args = get_request_args()
    
    url = flags.dli_rest_url + "/execs/%s/stop" % batch_id
    my_print(DEBUG, url)
    
    resp = requests.post(url, **req_args)

    resp_body = resp.content.decode('utf-8') if resp.content else ""
    if resp.ok:
        print("Stop exec %s succeed. HTTP:%s %s" % (batch_id, str(resp.status_code), resp_body))
        return 0
    else:
        print("Stop exec %s failed. HTTP:%s %s" % (batch_id, str(resp.status_code), resp_body))
        return 1


"""
Delete batch:
"""
def delete_batch(batch_id):
    my_print(DEBUG, "delete_batch()")
    req_args = get_request_args()
    
    url = flags.dli_rest_url + "/execs/%s" % batch_id
    my_print(DEBUG, url)
    
    resp = requests.delete(url, **req_args)

    resp_body = resp.content.decode('utf-8') if resp.content else ""
    if resp.ok:
        print("Delete exec %s succeed. HTTP:%s %s" % (batch_id, str(resp.status_code), resp_body))
        return 0
    else:
        print("Delete exec %s failed. HTTP:%s %s" % (batch_id, str(resp.status_code), resp_body))
        return 1


"""
Delete all batch
"""
def delete_batch_all():
    my_print(DEBUG, "delete_batch_all()")

    req_args = get_request_args()
    
    url = flags.dli_rest_url + "/execs"
    my_print(DEBUG, url)
    
    resp = requests.delete(url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    print("Execs are deleted")

def get_dlirest_port():
    # handle OCP route
    if flags.rest_port == -1:
        return ""
    else:
        return ":" + str(flags.rest_port)


"""
Get info from DLI
"""
def get_info_from_dli_rest():
    my_print(DEBUG, "get_info_from_dli_rest()")

    req_args = get_request_args()

    # always try with https first since this is default
    flags.dli_rest_url = "https://" + flags.rest_host + get_dlirest_port() + DLI_REST
    try:
        resp = requests.get(flags.dli_rest_url + "/conf", **req_args)
    except requests.exceptions.SSLError:
        flags.dli_rest_url = "http://" + flags.rest_host + get_dlirest_port() + DLI_REST
        resp = requests.get(flags.dli_rest_url + "/conf", **req_args)

    if not resp.ok:
        if resp.status_code == 401:
            print("Authentication error. The request was denied.")
            pprint_resp(resp)
        elif resp.status_code == 403:
            print("The request was denied. Try to run --logon again.")
        else:
            pprint_resp(resp)

        sys.exit(1)

    body = json.loads(resp.text)
    # pprint_json(body)

    flags.msd_enabled = (str(body.get('MSD_ENABLED', "")).lower() == "true")
    #flags.conductor_rest_url = x['CwS_REST_URL'] + "/platform/rest/conductor/v1"

    my_print(DEBUG, "dli_rest_url=%s" % (flags.dli_rest_url))

def make_tarfile(output_filename, source_dir):
    source_dir = strip_path(source_dir)

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def save_cookies(cookies, token):
    directory = os.path.dirname(COOKIE_FILE)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.open(COOKIE_FILE, os.O_CREAT | os.O_WRONLY, 0o600), 'wt') as text_file:
        text_file.write(token)

def get_cookies():
    my_print(DEBUG, "get_cookies()")

    cookie_dict = {}
    token = None

    if not flags.jwt_token:
        if os.path.isfile(COOKIE_FILE):
            with open(COOKIE_FILE, 'rt') as f:
                token = f.read()

    return cookie_dict, token

def get_request_args(accept='application/json'):
    headers = {'Accept': accept}
    cookies = None
    auth = None

    if flags.jwt_token:
        headers = {
            'Accept': accept,
            'Authorization': 'Bearer ' + flags.jwt_token
        }
    else:
        cookies, token = get_cookies()
        headers = {
            'Accept': accept,
            'Authorization': 'Bearer ' + token
        }
    return {'cookies': cookies, 'auth': auth, 'headers': headers, "verify": False}

def logon():
    if not flags.username:
        flags.username = get_username()
    if not flags.password:
        flags.password = get_userpass()

    headers = {'Accept': 'application/json'}

    # try wmla console route path
    url = "https://" + flags.rest_host + get_dlirest_port() + "/auth/v1/logon"
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, verify=False, headers=headers, auth=(flags.username, flags.password))
    if not resp.ok:
        os.system("rm -f %s" % COOKIE_FILE)

        if resp.status_code == 401:
            print("Authentication error. The request was denied.")
        elif resp.status_code == 400:
            print("Authentication error. Bad request.")

        pprint_resp(resp)
        sys.exit(1)

    pprint_resp(resp)
    body = json.loads(resp.text)
    token = body.get("accessToken", "")
    if token:
        #print("Logon succeed")
        save_cookies(resp.cookies, token)
    else:
        print("Logon failed. Check your input, and try again.")
        sys.exit(1)

def logoff():
    cookies, token = get_cookies()
    os.system("rm -f %s" % COOKIE_FILE)
    if token:
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer ' + token
        }
        url = "https://" + flags.rest_host + get_dlirest_port() + "/auth/v1/logoff"
        my_print(DEBUG, url)
        
        resp = requests.post(url=url, verify=False, headers=headers)

        pprint_resp(resp)
        if not resp.ok:
            sys.exit(1)

def check_cookies():
    my_print(DEBUG, "check_cookies()")

    if not flags.jwt_token:
        cookies, token = get_cookies()
        if not token:
            print("Unable to find token from cookies. Please logon first.")
            sys.exit(1)

# call DLI to submit
#
def create_batch(argv, data_source=None):
    my_print(DEBUG, "submit() model_dir=%s, model_main=%s" % (flags.model_dir, flags.model_main))

    args = ' '.join(argv)
    #print(args)

    my_print(DEBUG, "args=%s" % args)
    print("Copying files and directories ...")  # show progress ...
    temp_file = None
    multiple_files = []
    file_list = []

    if flags.model_dir:
        # tar up the dir
        _, temp_file = tempfile.mkstemp(MODEL_DIR_SUFFIX)

        make_tarfile(temp_file, flags.model_dir)
        my_print(DEBUG, "modelFiles=%s" % temp_file)
        file_list.append(temp_file)
    else:
        # if only model_main is specified, copy model main. The submit function will shorten the model_main before calling DLI
        if flags.model_main:
            my_print(DEBUG, "modelFiles=%s" % flags.model_main)
            file_list.append(flags.model_main)

    if flags.conda_env_yaml:
        my_print(DEBUG, "conda_env_yaml=%s" % flags.conda_env_yaml)
        file_list.append(flags.conda_env_yaml)

    if flags.conda_package:
        my_print(DEBUG, "conda_package=%s" % flags.conda_package)
        file_list.append(flags.conda_package)

    totoal_size = get_files_size(file_list)
    print("Content size: %s" % (byte_to_human_readable(totoal_size, "")))

    for file in file_list:
        multiple_files.append(('file', open(file, 'rb')))

    req_args = get_request_args()
    try:
        url = flags.dli_rest_url + "/execs/?args=" + args
        my_print(DEBUG, url)
        
        if data_source:
            my_print(DEBUG, "data_source=%s" % data_source)
            ds = dict(dataSource=json.loads(data_source))
            data = {'data': json.dumps(ds)}
            
            resp = requests.post(url, data=data, files=multiple_files, **req_args)
        else:
            resp = requests.post(url, files=multiple_files, **req_args)
    except:
        if temp_file and os.path.isfile(temp_file):
            os.remove(temp_file)  # always remove first
        raise

    if temp_file and os.path.isfile(temp_file):
        os.remove(temp_file)  # always remove first

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    body = json.loads(resp.text)

    json_obj = {
        "execId": body['id'],
        "appId": body['appId'],
    }

    pprint_json(json_obj)

def deploy_batch(deploy_args):

    def extract_attrs_envs(attrs_envs):
        attributes = dict()
        attrs = attrs_envs.split(',')
        for attr in attrs:
            attr = attr.strip()
            key_value = attr.split('=')
            if len(key_value) == 2:
                attributes[key_value[0]] = key_value[1]
        return attributes

    subparser = argparse.ArgumentParser(description='dlicmd options')

    subparser.add_argument("--exec-deploy", type=str, default=None, dest='batch_id', required=True)
    subparser.add_argument("--name", type=str, default=None)
    subparser.add_argument("--runtime", type=str, default=None)
    subparser.add_argument("--kernel", type=str, default=None, required=True)
    subparser.add_argument("--weight", type=str, default=None)
    subparser.add_argument("--tag", type=str, default=None)
    subparser.add_argument("--attributes", type=str, default=None)
    subparser.add_argument("--envs", type=str, default=None)

    subflags, _ = subparser.parse_known_args(args=deploy_args)

    deploy_body = dict()
    if subflags.name:
        deploy_body['name'] = subflags.name
    if subflags.runtime:
        deploy_body['runtime'] = subflags.runtime
    if subflags.kernel:
        deploy_body['kernel'] = subflags.kernel
    if subflags.weight:
        deploy_body['weight'] = subflags.weight
    if subflags.tag:
        deploy_body['tag'] = subflags.tag
    if subflags.attributes:
        deploy_body['attributes'] = extract_attrs_envs(subflags.attributes)
    if subflags.envs:
        deploy_body['environments'] = extract_attrs_envs(subflags.envs)

    print(deploy_body)
    print(subflags.batch_id)

    req_args = get_request_args()
    try:
        url = flags.dli_rest_url + "/execs/{}/deploy".format(subflags.batch_id)
        my_print(DEBUG, url)
        
        resp = requests.post(url, json=deploy_body, **req_args)
    except:
        raise

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    print(json.dumps(resp.json(), indent=4))

def get_username():
    while True:
        try:
            sys.stdout.write('Username: ')
            if PY3:
                user_input = input()
            else:
                user_input = raw_input()

            if user_input:
                user_input = user_input.strip()
                if user_input:
                    break

        except Exception as error:
            print(repr(error))

    return user_input

def get_userpass():
    while True:
        try:
            user_input = getpass.getpass()
            if user_input:
                break

        except Exception as error:
            print(repr(error))

    return user_input

def get_resource_dimension(res_req):
    res_dict = {}
    if res_req:
        pair_list = res_req.split(",")
        for pair in pair_list:
            key, val = pair.split("=")
            res_dict[key] = val
    return res_dict

def get_pj_id(pj):
    return xpath_get(pj, "#metadata#labels#msd/job", "", sep="#")

def get_pj_namespace(pj):
    return xpath_get(pj, "/metadata/namespace", "")

def get_pj_tag(pj):
    consumer = xpath_get(pj, "#metadata#annotations#lsf.ibm.com/consumer", "", "#")
    return consumer

def pj_to_row_dict(pj, pod_items=None):
    row_dict = {}

    pj_id = get_pj_id(pj)
    row_dict["id"] = pj_id

    row_dict["_type"] = "pj"

    row_dict["consumer"] = xpath_get(pj, "#metadata#annotations#lsf.ibm.com/consumer", "", "#")
    row_dict["submit_user"] = xpath_get(pj, "#metadata#annotations#lsf.ibm.com/submissionUser", "", "#")

    row_dict["tag"] = get_pj_tag(pj)

    job_type = xpath_get(pj, "#metadata#labels#msd/job-type", "", sep="#")
    row_dict["type"] = job_type

    job_state = xpath_get(pj, "/status/jobStatus", "")
    row_dict["state"] = job_state.upper()

    row_dict["namespace"] = xpath_get(pj, "/metadata/namespace", "")

    create_time_str = xpath_get(pj, "/metadata/creationTimestamp", "")
    row_dict["create_time"] = create_time_str

    row_dict["age"] = get_age(parse_utc_time(create_time_str), datetime.datetime.utcnow())

    # get job_res
    task_res_gpu = xpath_get(pj, "#metadata#annotations#lsf.ibm.com/gpu", "")
    task_res_mem = ""
    task_res_cpu = ""
    task12n_num = ""

    job_taskgrps = xpath_get(pj, "/spec/taskGroups", None)
    if job_taskgrps:
        for task in job_taskgrps:
            task_name = xpath_get(task, "/metadata/name", "")
            if task_name == "task12n":
                task12n_num = xpath_get(task, "/spec/replica", "")
                task_containers = xpath_get(task, "/spec/template/spec/containers", None)
                if task_containers:
                    for task_container in task_containers:
                        if xpath_get(task_container, "/name", "") == "msdtask":
                            if not task_res_gpu:
                                task_res_gpu = xpath_get(task_container, "#resources#requests#nvidia.com/gpu", "", "#")

                            task_res_mem = xpath_get(task_container, "#resources#requests#memory", "", "#")
                            task_res_cpu = xpath_get(task_container, "#resources#requests#cpu", "", "#")
                            break

    task_res_gpu = task_res_gpu if task_res_gpu else "n/a"
    task_res_mem = task_res_mem if task_res_mem else "n/a"
    task_res_cpu = task_res_cpu if task_res_cpu else "n/a"
    task12n_num = task12n_num if task12n_num else "n/a"

    res_req = "num=%s,ngpus=%s,mem=%s,ncpus=%s" % (task12n_num, task_res_gpu, task_res_mem, task_res_cpu)
    row_dict["res_req"] = res_req

    # get nodes
    pj_uid = xpath_get(pj, "/metadata/uid")
    pod_nodenames = set([])
    if pod_items:
        for pod in pod_items:
            pod_ctl_id = xpath_get(pod, "/metadata/labels/controller-uid")
            pod_owner_id = xpath_get(pod, "/metadata/ownerReferences/uid")
            if (pj_uid == pod_ctl_id) or (pj_uid == pod_owner_id):
                pod_name = xpath_get(pod, "/metadata/name")
                pod_nodename = get_short_hostname(xpath_get(pod, "/spec/nodeName"))
                #print("%s is running on %s" % (pod_name, pod_nodename))
                if pod_nodename:
                    pod_nodenames.add(pod_nodename)

    row_dict["nodes"] = ','.join(pod_nodenames)

    return row_dict

def is_normal_job(job):
    is_packing = job.get("is_packing", False)
    pack_id = job.get("pack_id", "")

    return (not is_packing) and (not pack_id)

def is_pack_job_parent(job):
    is_packing = job.get("is_packing", False)
    return is_packing

def is_pack_job_child(job):
    is_packing = job.get("is_packing", False)
    pack_id = job.get("pack_id", "")
    if not is_packing and pack_id:
        return True
    return False

def is_job_active(job):
    state = job.get("state", "")
    return state in APP_ACTIVE_STATES

def slot_to_human_readable(num):
    if num is None:
        return "n/a"

    try:
        num = float(num)
        if num.is_integer():
            return str(int(num))
        else:
            return "%.1f" % float(num)
    except ValueError:
        return num

def mb_to_human_readable(num, suffix="iB"):
    if num is None:
        return "n/a"

    num = float(num) * 1024 * 1024
    return byte_to_human_readable(num, suffix)

def byte_to_human_readable(num, suffix="iB"):
    if num is None:
        return "n/a"

    try:
        num = float(num)
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(num) < 1024.0:
                if num.is_integer():
                    return "%d%s%s" % (int(num), unit, suffix)
                else:
                    return "%.1f%s%s" % (num, unit, suffix)
            num /= 1024.0

        if num.is_integer():
            return "%d%s%s" % (int(num), 'Yi', suffix)
        else:
            return "%.1f%s%s" % (num, 'Yi', suffix)

    except ValueError:
        return num

def get_job_tag(job):
    ret = ""

    consumer = job.get("consumer", "")
    submit_user = job.get("submit_user", "")
    tag = job.get("tag", "")

    if tag:
        ret = tag
    elif consumer and submit_user:
        ret = str(PurePath(consumer, submit_user))
    elif consumer:
        ret = consumer
    elif submit_user:
        ret = submit_user
    else:
        ret = ""

    return ret

def job_to_row_dict(job, pod_items=None):
    row_dict = {}

    job_name = job.get("name", "")

    job_used_gpu = job.get("used_gpu")
    job_used_mem = job.get("used_mem")
    job_used_cpu = job.get("used_cpu")

    task0_res = get_resource_dimension(job.get("task0_resreq", None))
    task12n_res = get_resource_dimension(job.get("task12n_resreq", None))

    if "task0_resreq" in job:
        task0_num = 1
    else:
        task0_num = 0

    task12n_num = job.get("task12n_num")
    if not task12n_num:
        task12n_num = 1

    if is_pack_job_parent(job):
        job_name = ""

        task0_gpu = task0_res.get("ngpus_limit")
        task0_mem = task0_res.get("mem_limit")
        task0_cpu = task0_res.get("ncpus_limit")

        task12n_gpu = task12n_res.get("ngpus_limit")
        task12n_mem = task12n_res.get("mem_limit")
        task12n_cpu = task12n_res.get("ncpus_limit")

    else:
        task0_gpu = task0_res.get("ngpus")
        task0_mem = task0_res.get("mem")
        task0_cpu = task0_res.get("ncpus")

        task12n_gpu = task12n_res.get("ngpus")
        task12n_mem = task12n_res.get("mem")
        task12n_cpu = task12n_res.get("ncpus")

    row_dict["res_req"] = "num=%s,ngpus=%s,mem=%s,ncpus=%s" % (task12n_num, slot_to_human_readable(task12n_gpu), mb_to_human_readable(task12n_mem, ""), slot_to_human_readable(task12n_cpu))

    row_dict["gpu"] = "%.1f/%.1f" % (float(job_used_gpu) if job_used_gpu else 0.0, float(task12n_gpu) * task12n_num if task12n_gpu else 0.0)
    row_dict["mem"] = "%.1f/%.1f" % (float(job_used_mem) if job_used_mem else 0.0, float(task12n_mem) * task12n_num if task12n_mem else 0.0)
    row_dict["cpu"] = "%.1f/%.1f" % (float(job_used_cpu) if job_used_cpu else 0.0, float(task12n_cpu) * task12n_num if task12n_cpu else 0.0)

    row_dict["id"] = job.get("id", "")
    row_dict["_type"] = "job"
    row_dict["tag"] = get_job_tag(job)

    row_dict["consumer"] = job.get("consumer", "")
    row_dict["submit_user"] = job.get("submit_user", "")

    row_dict["name"] = job_name
    row_dict["pri"] = job.get("priority", "")
    row_dict["type"] = job.get("type", "")
    
    job_state = job.get("state", "")
    job_state = "PENDING_PJ" if job_state == "PENDING_CRD_SCHEDULER" else job_state
    job_state = "PENDING_JOB" if job_state == "PENDING_JOB_SCHEDULER" else job_state
    row_dict["state"] = job_state
    
    row_dict["num"] = str(task12n_num)
    row_dict["submit_time"] = timestamp_to_string(job.get("submit_time"))
    row_dict["start_time"] = timestamp_to_string(job.get("start_time"))
    row_dict["end_time"] = timestamp_to_string(job.get("end_time"))

    row_dict["age"] = get_age(job.get("submit_time"), datetime.datetime.utcnow())

    row_dict["duration"] = get_age(job.get("start_time"), job.get("end_time"))

    kernel_ips = set([])
    kernels = job.get("kernels")
    if kernels:
        for kernel in kernels:
            kernel_ip = kernel.get("ip", "")
            if kernel_ip:
                kernel_ips.add(kernel_ip)

    pod_nodenames = set([])
    if kernel_ips and pod_items:
        for pod in pod_items:
            pod_nodename = get_short_hostname(xpath_get(pod, "/spec/nodeName"))
            if pod_nodename:
                pod_name = xpath_get(pod, "/metadata/name")
                pod_ip = xpath_get(pod, "/status/podIP")
                #print("%s is running on %s" % (pod_name, pod_nodename))
                if pod_ip and (pod_ip in kernel_ips):
                    pod_nodenames.add(pod_nodename)

                pod_ip_items = xpath_get(pod, "/status/podIPs", default=[])
                for pod_ip_item in pod_ip_items:
                    pod_ip = pod_ip_item.get("ip", "")
                    if pod_ip and (pod_ip in kernel_ips):
                        pod_nodenames.add(pod_nodename)

    row_dict["nodes"] = ','.join(pod_nodenames)

    return row_dict

def get_job_rows(body, state_list, pod_items=None):
    rows = []
    job_id_set = set([])

    if not body:
        return rows

    #print(body)
    for job in body:

        job_id = job.get("id", "")
        job_state = job.get("state", "")

        if job_id in job_id_set:
            continue

        if job_state not in state_list:
            continue

        if is_normal_job(job):
            rows.append(job_to_row_dict(job, pod_items))
            job_id_set.add(job_id)

        elif is_pack_job_parent(job):
            rows.append(job_to_row_dict(job, pod_items))
            job_id_set.add(job_id)

            parent_id = job.get("id", "")

            for sub_job in body:
                sub_job_id = sub_job.get("id", "")
                if not is_pack_job_child(sub_job):
                    continue
                if not sub_job.get("state", "") in state_list:
                    continue
                if parent_id != sub_job.get("pack_id", ""):
                    continue

                sub_job_row_dict = job_to_row_dict(sub_job, pod_items)
                sub_job_row_dict["id"] = "    " + sub_job_row_dict["id"]
                rows.append(sub_job_row_dict)
                job_id_set.add(sub_job_id)

    return rows

def list_app():
    req_args = get_request_args()

    url = flags.dli_rest_url + "/scheduler/engine/applications?start=%d&length=%d&sort=-submit_time&state=%s" % (flags.app_search_start, flags.app_search_length, ",".join(APP_ACTIVE_STATES))
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    pod_items = list_k8s_obj("pod")

    body = reversed(json.loads(resp.text))
    if flags.output_format == OutputFormat.JSON:
        job_list = []
        for job in body:
            if job.get("state") in APP_ACTIVE_STATES:
                job_list.append(job)
        pprint_json(job_list)
    else:
        columns = get_custom_columns()
        if not columns:
            columns = ["id", "consumer", "submit_user", "name", "pri", "type", "state", "num", "gpu", "mem", "cpu", "submit_time", "start_time", "age"] #"nodes", 

        rows = get_job_rows(body, APP_ACTIVE_STATES, pod_items)
        if rows:
            print_table(columns, rows)
        else:
            print("No result found")

def list_app_history():
    req_args = get_request_args()

    url = flags.dli_rest_url + "/scheduler/engine/applications?start=%d&length=%d&sort=-submit_time&state=%s" % (flags.app_search_start, flags.app_search_length, ",".join(APP_NONACTIVE_STATES))
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    pod_items = list_k8s_obj("pod")

    body = reversed(json.loads(resp.text))
    if flags.output_format == OutputFormat.JSON:
        job_list = []
        for job in body:
            if job.get("state") in APP_NONACTIVE_STATES:
                job_list.append(job)
        pprint_json(job_list)
    else:
        columns = get_custom_columns()
        if not columns:
            columns = ["id", "consumer", "submit_user", "name", "pri", "type", "state", "num", "gpu", "mem", "cpu", "submit_time", "start_time", "duration" ] # "end_time"

        rows = get_job_rows(body, APP_NONACTIVE_STATES, pod_items)
        if rows:
            print_table(columns, rows)
        else:
            print("No result found")

def get_app(app_id):
    req_args = get_request_args()

    url = flags.dli_rest_url + "/scheduler/engine/applications/%s" % app_id
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)
    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    #app_list = json.loads(resp.text)
    app_list = loads_json_with_order(resp.text)
    if app_list and len(app_list) > 0:
        pprint_json(app_list)

def wait_app(app_id):
    req_args = get_request_args()
    url = flags.dli_rest_url + "/scheduler/engine/applications/%s" % app_id
    my_print(DEBUG, url)
    
    while True:
        resp = requests.get(url=url, **req_args)
        if not resp.ok:
            pprint_resp(resp)
            sys.exit(1)

        app = None
        app_list = json.loads(resp.text)
        if app_list and len(app_list) > 0:
            app = app_list[0]

        if not app:
            print("Application %s does not exists" % app_id)
            sys.exit(1)

        app_state = app.get("state", "")
        app_age = get_age(app.get("submit_time"), datetime.datetime.utcnow())
        print(("%s    %s    %s") % (app_id, app_state, app_age))
        if app_state in APP_NONACTIVE_STATES:
            break

        time.sleep(10)

def stop_app(app_id_list):
    if not app_id_list:
        return 0

    req_args = get_request_args()

    ret = 0
    for app_id in app_id_list:
        url = flags.dli_rest_url + "/scheduler/applications/%s/stop" % app_id
        my_print(DEBUG, url)
        
        resp = requests.put(url=url, **req_args)
        resp_body = resp.content.decode('utf-8') if resp.content else ""
        if resp.ok:
            print("Stop application %s succeed. HTTP:%s %s" % (app_id, str(resp.status_code), resp_body))
        else:
            print("Stop application %s failed. HTTP:%s %s" % (app_id, str(resp.status_code), resp_body))
            ret = 1

    return ret

def stop_app_all():
    req_args = get_request_args()

    url = flags.dli_rest_url + "/scheduler/engine/applications?start=%d&length=%d&sort=submit_time&state=%s" % (flags.app_search_start, flags.app_search_length, ",".join(APP_ACTIVE_STATES))
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    app_id_list = []
    app_list = json.loads(resp.text)
    if app_list and len(app_list) > 0:
        for app in app_list:
            app_id = app.get("id")
            app_state = app.get("state")
            if app_state in APP_ACTIVE_STATES:
                app_id_list.append(app_id)

    return stop_app(app_id_list)

# List resource plan related functions
class ResoucePlanNode(object):

    def __init__(self, parent, name, is_root, is_rp):
        self.parent = parent
        self.name = name
        self.is_root = is_root
        self.is_rp = is_rp
        self.children = []

        if not is_root:
            self.parent.children.append(self)

    def is_root_node(self):
        return self.is_root

    def get_tag(self):
        if self.parent:
            return str(PurePath(self.parent.get_tag(), self.name))
        else:
            return "/"

    def is_resouceplan(self):
        return self.is_rp

    def is_user(self):
        return not self.is_rp

    def get_level(self):
        if self.parent:
            return self.parent.get_level() + 1
        else:
            return 0

    def __str__(self):
        node_name = str(self.name)
        return node_name

    def __repr__(self):
        return '<tree node representation>'

    def own_job(self, job):
        # node_tag: /<tetheredNameSpace>/projectid/training/user
        node_tag = self.get_tag()
        workload_tag = get_job_tag(job)

        #print("node_tag=%s, workload_tag=%s" % (node_tag, workload_tag))
        if node_tag and workload_tag:
            # workload_tag from job inforservice is relative path
            # workload_tag is same as we submitted. e.g /projectid/training/user
            # if node_tag.endswith(workload_tag):
            if node_tag == workload_tag:
                return True

        return False

    def own_pj(self, pj):
        # node_tag: /<tetheredNameSpace>/projectid/training/user
        node_tag = self.get_tag()
        workload_tag = get_pj_tag(pj)
        if node_tag and workload_tag:
            # tag from paralleljob is full path, muteated by cpd weebhook
            # workload_tag is different with we submitted. e.g /8d68a333-1cff-4b43-8061-4b6e489aeca2/Product A/tetheredNameSpace>/projectid/training/user
            if workload_tag.endswith(node_tag):
                return True

        return False

def list_k8s_obj(obj_type):
    req_args = get_request_args()

    url = flags.dli_rest_url + "/scheduler/kubernetesobjects?type=%s" % obj_type
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)
    if resp.ok:
        try:
            json_obj = json.loads(resp.text)
            # pprint_json(items)
            return json_obj.get("items")
        except:
            return []
    else:
        pprint_resp(resp)
        return []

def find_resource_plan(rp_items, path_pairs):
    # /metadata/name
    # /metadata/namespace
    # /spec/parent
    for item in rp_items:
        found = True
        for item_path, item_val in path_pairs.items():
            val = xpath_get(item, item_path, "")
            if val != item_val:
                found = False
                break
        if found:
            return item
    return None

def build_rp_tree(parent_node, rp_json, job_items):
    if not parent_node:
        return

    if not rp_json:
        return

    child_name = xpath_get(rp_json, "id")
    child_pid = xpath_get(rp_json, "pid")
    child_rp_node = ResoucePlanNode(parent=parent_node, name=child_name, is_root=False, is_rp=True)

    grand_children_json = xpath_get(rp_json, "childTreeDto")
    if grand_children_json:
        for grand_child_json in grand_children_json:
            build_rp_tree(child_rp_node, grand_child_json, job_items)

    grandchild_name_list = set([])
    for rp_node in child_rp_node.children:
        grandchild_name_list.add(rp_node.name)

    for job in job_items:
        consumer = job.get("consumer", "")
        submit_user = job.get("submit_user", "")
        #print("node_tag=%s, consumer=%s" % (child_rp_node.get_tag(), consumer))
        if is_job_active(job) and (consumer == child_rp_node.get_tag()) and submit_user and (submit_user not in grandchild_name_list):
            child_user_node = ResoucePlanNode(parent=child_rp_node, name=submit_user, is_root=False, is_rp=False)
            #print("adding child %s for job %s %s " %( submit_user, job.get("id"), job.get("state")))
            grandchild_name_list.add(submit_user)

def print_rp_workloads(rp_node, job_items, pod_items, ignore_ownership, printed_workloads):
    if not rp_node:
        return

    #print("print workload for node %s" % rp_node.name)
    job_padding = (RP_PADDING_SPACE * rp_node.get_level())
    sub_job_padding = job_padding + " " * len(RP_PADDING_SPACE)

    rows = []

    # print jobs
    for job in job_items:
        job_id = job.get("id", "")
        if (job_id not in printed_workloads) and is_job_active(job) and (ignore_ownership or rp_node.own_job(job)):
            #print("job_tag=%s, node_tag=%s" % (job_tag, rp_node.get_tag()))
            if is_normal_job(job):
                job_row_dict = job_to_row_dict(job, pod_items)
                job_row_dict["id"] = job_padding + job_row_dict["id"]
                rows.append(job_row_dict)
                printed_workloads.add(job_id)
            elif is_pack_job_parent(job):
                job_row_dict = job_to_row_dict(job, pod_items)
                job_row_dict["id"] = job_padding + job_row_dict["id"]
                rows.append(job_row_dict)
                printed_workloads.add(job_id)

                for sub_job in job_items:
                    sub_job_id = sub_job["id"]
                    if (sub_job_id not in printed_workloads) and is_job_active(sub_job) and is_pack_job_child(sub_job) and (sub_job.get("pack_id", "") == job_id):
                        sub_job_row_dict = job_to_row_dict(sub_job, pod_items)
                        sub_job_row_dict["id"] = sub_job_padding + sub_job_row_dict["id"]
                        rows.append(sub_job_row_dict)
                        printed_workloads.add(sub_job_id)

    columns = get_custom_columns()
    if not columns:
        #columns = ["id", "type", "state", "res_req", "age", "nodes", "_type"]
        columns = ["id", "type", "state", "res_req", "age", "nodes"]

    print_table(columns, rows, show_header=False, show_index=False, show_h_border=False)

def print_rp_node(rp_node, job_items, pod_items, ignore_ownership, printed_workloads):
    # print node itself
    if rp_node.get_level() > 0:
        node_padding = RP_PADDING_SPACE * (rp_node.get_level() - 1) + RP_PADDING_CHILD
    else:
        node_padding = ""

    if rp_node.is_root_node():
        node_name = TerminalColor.GREEN + "%s" % (str(rp_node.name)) + TerminalColor.ENDC
    elif rp_node.is_resouceplan():
        node_name = TerminalColor.GREEN + "%s" % (str(rp_node.name)) + TerminalColor.ENDC
    else:
        node_name = TerminalColor.RED + "%s" % (str(rp_node.name)) + TerminalColor.ENDC

    node_line = node_padding + node_name
    print(node_line)

    # print node workload
    print_rp_workloads(rp_node, job_items, pod_items, ignore_ownership, printed_workloads)

    # print children node
    for child in rp_node.children:
        print_rp_node(child, job_items, pod_items, ignore_ownership, printed_workloads)


def list_rp():
    req_args = get_request_args()

    # get all resource plan
    rp_json = None
    
    url = flags.dli_rest_url + "/resplans/resplantree"
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)
    if resp.ok:
        rp_json = json.loads(resp.text)
    else:
        pprint_resp(resp)
        sys.exit(1)

    if flags.output_format == OutputFormat.JSON:
        pprint_json(rp_json)
    else:
        # get all jobs
        job_items = []
        try:
            url = flags.dli_rest_url + "/scheduler/engine/applications?start=%d&length=%d&sort=submit_time&state=%s" % (flags.app_search_start, flags.app_search_length, ",".join(APP_ACTIVE_STATES))
            my_print(DEBUG, url)
            resp = requests.get(url=url, **req_args)
            if resp.ok:
                job_items = json.loads(resp.text)
        except:
            # print workload is optional
            pass

        # get all pods
        pod_items = list_k8s_obj("pod")

        # build root node
        root_node = ResoucePlanNode(parent=None, name="root", is_root=True, is_rp=True)
        build_rp_tree(root_node, rp_json, job_items)

        # print rp tree and workloads
        printed_workloads = set([])
        print_rp_node(rp_node=root_node, job_items=job_items, pod_items=pod_items, ignore_ownership=False, printed_workloads=printed_workloads)

        # handle unknown worloads
        has_unknown = False
        for job in job_items:
            job_id = job.get("id", "")
            if is_job_active(job) and job_id and (job_id not in printed_workloads):
                has_unknown = True
                break

        if has_unknown:
            unknown_node = ResoucePlanNode(parent=root_node, name="Unknown", is_root=False, is_rp=True)
            print_rp_node(rp_node=unknown_node, job_items=job_items, pod_items=pod_items, ignore_ownership=True, printed_workloads=printed_workloads)

def get_rp(rp_name):
    req_args = get_request_args()
    
    url = flags.dli_rest_url + "/resplans/resplan/%s" % rp_name
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)
    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    rp_json = loads_json_with_order(resp.text)
    pprint_json(rp_json)

def delete_rp(rp_name):
    req_args = get_request_args()
    
    url = flags.dli_rest_url + "/resplans/resplan/%s" % rp_name
    my_print(DEBUG, url)
    
    resp = requests.delete(url=url, **req_args)
    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    pprint_resp(resp)

def list_pj():
    my_print(DEBUG, "list_pj()")
    # get all parallel job
    pj_items = list_k8s_obj("paralleljob")

    if flags.output_format == OutputFormat.JSON:
        pprint_json(pj_items)
    else:
        # get all pods
        pod_items = list_k8s_obj("pod")

        columns = get_custom_columns()
        if not columns:
            columns = ["id", "namespace", "consumer", "submit_user", "type", "state", "res_req", "nodes", "create_time", "age"]

        rows = []
        for pj in pj_items:
            pj_row_dict = pj_to_row_dict(pj, pod_items)
            rows.append(pj_row_dict)

        if rows:
            print_table(columns, rows)
        else:
            print("No result found")

# analyze applications
class JobState:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    KILLED = "KILLED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

    STATES = [PENDING, RUNNING, FINISHED, KILLED, ERROR, UNKNOWN]

def translate_state(state):
    if state in ["NEW", "SUBMITTED", "PENDING", "PENDING_JOB_SCHEDULER", "PENDING_CRD_SCHEDULER", "RECOVERING"]:
        return JobState.PENDING

    if state in ["LAUNCHING", "RUNNING"]:
        return JobState.RUNNING

    if state in ["FINISHED"]:
        return JobState.FINISHED

    if state in ["KILLED"]:
        return JobState.KILLED

    if state in ["ERROR"]:
        return JobState.ERROR

    return JobState.UNKNOWN

class UserStat:

    def __init__(self, job_tag, job_type):
        self.job_tag = job_tag
        self.job_type = job_type

        # using resource
        self.curr_cpu = 0.0
        self.curr_gpu = 0.0
        self.curr_mem = 0.0

        # requested resource
        self.history_cpu = 0.0
        self.history_gpu = 0.0
        self.history_mem = 0.0

        self.job_count = OrderedDict()

        for state_name in JobState.STATES:
            self.job_count[state_name] = 0

    def append_job(self, job):
        # get resource in use
        job_used_gpu = job.get("used_gpu")
        job_used_mem = job.get("used_mem")
        job_used_cpu = job.get("used_cpu")

        self.curr_cpu += float(job_used_cpu) if job_used_cpu else 0.0
        self.curr_gpu += float(job_used_gpu) if job_used_gpu else 0.0
        self.curr_mem += float(job_used_mem) if job_used_mem else 0.0

        job_state = translate_state(job["state"])
        self.job_count[job_state] += 1

        if job_state in [JobState.ERROR, JobState.FINISHED, JobState.KILLED, JobState.UNKNOWN]:
            task0_res = get_resource_dimension(job.get("task0_resreq", None))
            task12n_res = get_resource_dimension(job.get("task12n_resreq", None))

            if "task0_resreq" in job:
                task0_num = 1
            else:
                task0_num = 0

            if "task12n_resreq" in job and "task12n" in job:
                task12n_num = len(job["task12n"])
            else:
                task12n_num = 0

            if is_normal_job(job):
                task0_gpu = task0_res.get("ngpus")
                task0_mem = task0_res.get("mem")
                task0_cpu = task0_res.get("ncpus")

                task12n_gpu = task12n_res.get("ngpus")
                task12n_mem = task12n_res.get("mem")
                task12n_cpu = task12n_res.get("ncpus")
            elif is_pack_job_parent(job):
                job_name = ""

                task0_gpu = task0_res.get("ngpus_limit")
                task0_mem = task0_res.get("mem_limit")
                task0_cpu = task0_res.get("ncpus_limit")

                task12n_gpu = task12n_res.get("ngpus_limit")
                task12n_mem = task12n_res.get("mem_limit")
                task12n_cpu = task12n_res.get("ncpus_limit")
            else:
                return

            self.history_cpu += task0_num * float(task0_cpu) if task0_cpu else 0.0
            self.history_gpu += task0_num * float(task0_gpu) if task0_gpu else 0.0
            self.history_mem += task0_num * float(task0_mem) if task0_mem else 0.0

            self.history_cpu += task12n_num * float(task12n_cpu) if task12n_cpu else 0.0
            self.history_gpu += task12n_num * float(task12n_gpu) if task12n_gpu else 0.0
            self.history_mem += task12n_num * float(task12n_mem) if task12n_mem else 0.0


def get_timestamp(dt):
    try:
        # python 3
        return dt.timestamp()
    except AttributeError:
        # python 2
        return time.mktime(dt.timetuple())


# noinspection PyDictCreation
def analyze_app(num_hours):
    num_hours = num_hours if num_hours else DEFAULT_ANALYZE_HOURS
    #print("num_hours=" + str(num_hours))
    to_timestamp = datetime.datetime.now()
    from_timestamp = to_timestamp - datetime.timedelta(hours=int(num_hours))
    print("From %s To %s" % (str(from_timestamp), str(to_timestamp)))

    req_args = get_request_args()
    
    url = flags.dli_rest_url + "/scheduler/engine/applications?start=%d&length=%d&sort=-submit_time" % (flags.app_search_start, flags.app_search_length)
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    job_items = json.loads(resp.text)

    user_stats = {}
    for job in job_items:
        job_tag = job["tag"]
        job_type = job["type"]
        job_submit_time = job["submit_time"]
        state = job["state"]
        if state in APP_NONACTIVE_STATES:
            # filter out job created before from_timestamp
            if from_timestamp and job_submit_time < get_timestamp(from_timestamp):
                # print("job_submit_time=%s, from_timestamp=%s" % (str(job_submit_time), str(from_timestamp.timestamp())))
                continue

            if to_timestamp and job_submit_time > get_timestamp(to_timestamp):
                # print("job_submit_time=%s, to_timestamp=%s" % (str(job_submit_time), str(to_timestamp.timestamp())))
                continue

        key = job_type + "_" + job_tag
        stat = None

        if key in user_stats:
            stat = user_stats[key]
        else:
            # print("key=" + key)
            stat = UserStat(job_tag, job_type)
            user_stats[key] = stat

        stat.append_job(job)

    # print tables
    columns = get_custom_columns()
    if not columns:
        columns = ["tag", "type", "curr_cpu", "curr_gpu", "curr_mem", "hist_cpu", "hist_gpu", "hist_mem"]
        for state_name in JobState.STATES:
            columns.append(state_name)

    rows = []
    for key, stat in sorted(user_stats.items()):
        row = {}
        row["tag"] = stat.job_tag
        row["type"] = stat.job_type
        row["curr_cpu"] = ("%.1f" % stat.curr_cpu)
        row["curr_gpu"] = ("%.1f" % stat.curr_gpu)
        row["curr_mem"] = mb_to_human_readable(stat.curr_mem, "")
        row["hist_cpu"] = ("%.1f" % stat.history_cpu)
        row["hist_gpu"] = ("%.1f" % stat.history_gpu)
        row["hist_mem"] = mb_to_human_readable(stat.history_mem, "")
        for state, cnt in stat.job_count.items():
            row[state] = cnt
        rows.append(row)

    print_table(columns, rows)

def get_k8s_obj_all(obj_type):
    req_args = get_request_args()

    url = flags.dli_rest_url + "/scheduler/kubernetesobjects?type=%s" % obj_type
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)
    if resp.ok:
        json_obj = json.loads(resp.text)
        pprint_json(json_obj)
    else:
        pprint_resp(resp)
        sys.exit(1)

def get_k8s_obj(obj_get_params):
    if (obj_get_params is None) or len(obj_get_params) != 2:
        my_print(ERROR, 'The command reuqire 2 paramters: obj type and obj name.')
        sys.exit(1)

    req_args = get_request_args()

    url = flags.dli_rest_url + "/scheduler/kubernetesobjects?type=%s&name=%s" % (obj_get_params[0], obj_get_params[1])
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)
    if resp.ok:
        json_obj = json.loads(resp.text)
        pprint_json(json_obj)
    else:
        pprint_resp(resp)
        sys.exit(1)

def hpo_to_row_dict(hpo):
    row = {}
    row["duration"] = hpo.get("duration")
    row["progress"] = hpo.get("progress")
    row["state"] = hpo.get("state")
    row["creator"] = hpo.get("creator")
    row["createtime"] = hpo.get("createtime")
    row["hpoName"] = hpo.get("hpoName")
    row["name"] = hpo.get("hpoName")
    row["running"] = hpo.get("running")
    row["complete"] = hpo.get("complete")
    row["failed"] = hpo.get("failed")

    exp_row_list = []
    if "experiments" in hpo:
        for exp in hpo.get("experiments"):
            exp_row = experiment_to_row_dict(exp)
            exp_row_list.append(exp_row)

    row["experiments"] = exp_row_list

    return row

def hpo_algo_to_row_dict(hpo_algo):
    row = {}
    row["name"] = hpo_algo.get("name")
    row["type"] = hpo_algo.get("type")
    row["remoteExec"] = hpo_algo.get("remoteExec")
    return row

def experiment_to_row_dict(exp):
    row = {}
    row["state"] = exp.get("state")
    row["endTime"] = exp.get("endTime", "")
    row["startTime"] = exp.get("startTime", "")
    row["id"] = exp.get("id")
    row["appId"] = exp.get("appId")
    row["driverId"] = exp.get("driverId")
    row["metricVal"] = exp.get("metricVal")
    return row

def print_hpo_callback(hpo_row):
    exp_rows = hpo_row.get("experiments")
    if exp_rows:
        for exp_row in exp_rows:
            exp_row["id"] = " " * 8 + str(exp_row["id"])
        columns = ["id", "appId", "state", "metricVal", "startTime", "endTime"]
        print_table(columns=columns, rows=exp_rows, show_header=False, show_index=False, show_h_border=False)

def start_hpo():
    my_print(DEBUG, "submit() hpo_input=%s, model_dir=%s" % (flags.hpo_input, flags.model_dir))

    if flags.hpo_input is None or not os.path.exists(flags.hpo_input):
        my_print(ERROR, "Invalid parameter for --hpo-start, a file in json format with search definition is required")
        sys.exit(1)

    if flags.model_dir is None or not os.path.exists(flags.model_dir):
        my_print(ERROR, "Invalid parameter for --model-dir, a directory with model files in it is required")
        sys.exit(1)

    body = {'data': json.dumps(json.load(open(flags.hpo_input)))}

    print("Copying files and directories ...")  # show progress ...
    temp_file = None
    multiple_files = []
    file_list = []

    # tar up the dir
    _, temp_file = tempfile.mkstemp(MODEL_DIR_SUFFIX)

    make_tarfile(temp_file, flags.model_dir)
    my_print(DEBUG, "modelFiles=%s" % temp_file)
    file_list.append(temp_file)

    totoal_size = get_files_size(file_list)
    print("Content size: %s" % (byte_to_human_readable(totoal_size, "")))

    for file in file_list:
        multiple_files.append(('file', open(file, 'rb')))

    req_args = get_request_args()
    try:
        resp = requests.post(flags.dli_rest_url + "/hypersearch", files=multiple_files, data=body, **req_args)
    except:
        if temp_file and os.path.isfile(temp_file):
            os.remove(temp_file)  # always remove first
        raise

    if temp_file and os.path.isfile(temp_file):
        os.remove(temp_file)  # always remove first

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    print("Start hypersearch with id: {}".format(resp.text))

def list_hpo():
    req_args = get_request_args()

    url = flags.dli_rest_url + "/hypersearch"
    if flags.query_args:
         url = url + "?" + flags.query_args
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    hpo_items = json.loads(resp.text)
    if flags.output_format == OutputFormat.JSON:
        pprint_json(hpo_items)
    else:
        columns = get_custom_columns()
        if not columns:
            columns = ["name", "state", "running", "progress", "creator", "createtime", "duration", "complete", "failed"]

        rows = []
        if hpo_items:
            for hpo in hpo_items:
                row_dict = hpo_to_row_dict(hpo)
                rows.append(row_dict)

        if rows:
            print_table(columns, rows, callback=print_hpo_callback)
        else:
            print("No result found")

def get_hpo(hpo_id):
    req_args = get_request_args()

    url = flags.dli_rest_url + "/hypersearch/%s" % hpo_id
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)
    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    hpo = json.loads(resp.text)
    pprint_json(hpo)

def stop_hpo(hpo_id_list):
    if not hpo_id_list:
        return 0

    req_args = get_request_args()

    ret = 0
    for hpo_id in hpo_id_list:
        url = flags.dli_rest_url + "/hypersearch/%s" % hpo_id
        my_print(DEBUG, url)
        
        resp = requests.put(url=url, **req_args)
        resp_body = resp.content.decode('utf-8') if resp.content else ""
        if resp.ok:
            print("Stop hpo %s succeed. HTTP:%s %s" % (hpo_id, str(resp.status_code), resp_body))
        else:
            print("Stop hpo %s failed. HTTP:%s %s" % (hpo_id, str(resp.status_code), resp_body))
            ret = 1

    return ret


def restart_hpo(hpo_id):
    req_args = get_request_args()

    url = flags.dli_rest_url + "/hypersearch/%s/restart" % hpo_id
    my_print(DEBUG, url)
    
    resp = requests.put(url=url, **req_args)
    pprint_resp(resp)
    if not resp.ok:
        sys.exit(1)

def delete_hpo(hpo_id_list):
    if not hpo_id_list:
        return 0

    req_args = get_request_args()

    ret = 0
    for hpo_id in hpo_id_list:
        url = flags.dli_rest_url + "/hypersearch/%s" % hpo_id
        my_print(DEBUG, url)
        
        resp = requests.delete(url=url, **req_args)
        resp_body = resp.content.decode('utf-8') if resp.content else ""
        if resp.ok:
            print("Delete hpo %s succeed. HTTP:%s %s" % (hpo_id, str(resp.status_code), resp_body))
        else:
            print("Delete hpo %s failed. HTTP:%s %s" % (hpo_id, str(resp.status_code), resp_body))
            ret = 1

    return ret

def delete_hpo_all():
    req_args = get_request_args()
    
    url = flags.dli_rest_url + "/hypersearch"
    my_print(DEBUG, url)
    
    resp = requests.delete(url=url, **req_args)
    pprint_resp(resp)
    if not resp.ok:
        sys.exit(1)

def list_hpo_algorithm():
    req_args = get_request_args()

    url = flags.dli_rest_url + "/hypersearch/algorithm"
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)

    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    hpo_algo_items = json.loads(resp.text)
    if flags.output_format == OutputFormat.JSON:
        pprint_json(hpo_algo_items)
    else:
        columns = get_custom_columns()
        if not columns:
            columns = ["name", "type", "remoteExec"]

        rows = []
        if hpo_algo_items:
            for hpo_algo in hpo_algo_items:
                row_dict = hpo_algo_to_row_dict(hpo_algo)
                rows.append(row_dict)

        if rows:
            print_table(columns, rows, callback=print_hpo_callback)
        else:
            print("No result found")

def get_hpo_algorithm(hpo_algorithm):
    req_args = get_request_args()

    url = flags.dli_rest_url + "/hypersearch/algorithm/%s" % hpo_algorithm
    my_print(DEBUG, url)
    
    resp = requests.get(url=url, **req_args)
    if not resp.ok:
        pprint_resp(resp)
        sys.exit(1)

    hpo_algorithm_json = json.loads(resp.text)
    pprint_json(hpo_algorithm_json)

def delete_hpo_algorithm(hpo_algorithm):
    req_args = get_request_args()

    url = flags.dli_rest_url + "/hypersearch/algorithm/%s" % hpo_algorithm
    my_print(DEBUG, url)
    
    resp = requests.delete(url=url, **req_args)
    pprint_resp(resp)
    if not resp.ok:
        sys.exit(1)

def print_help(module_name):
    print("Usage:")
    print("   python " + module_name + " --help")
    print("   python " + module_name + " --logon <connection-options> <logon-options>")
    print("   python " + module_name + " --dl-frameworks | --exec-get-all | --exec-delete-all <connection-options>")
    print("   python " + module_name + " --exec-start <framework-name> <connection-options> <datastore-meta> <submit-arguments>")
    print("   python " + module_name + " --exec-get | --exec-stop | --exec-delete <exec-id> <connection-options>")
    print("   python " + module_name + " --exec-deploy <connection-options> <deploy-arguments>")
    print("   python " + module_name + " --exec-launcherlogs | --exec-outlogs | --exec-errlogs <exec-id> <connection-options>")
    print("   python " + module_name + " --exec-trainlogs | --exec-trainoutlogs |--exec-trainerrlogs | --exec-trainresult <exec-id> <connection-options>")
    print("")

    print("Commands:")
    print("   --help               Print help message.")
    print("   --logon              Logs the user onto IBM Watson Machine Learning Accelerator")
    print("   --dl-frameworks      List available deep learning frameworks for exec.")
    print("   --exec-start         Submit a deep learning exec.")
    print("   --exec-get-all       Get all deep learning execs for current user.")
    print("   --exec-get           Show info for a deep learning exec")
    print("   --exec-stop          Stop a deep learning exec.")
    print("   --exec-deploy        Deploy a deep learning exec.")
    print("   --exec-delete        Delete a deep learning exec.")
    print("   --exec-delete-all    Delete all deep learning execs for current user.")
    print("   --exec-launcherlogs  Get launcher logs for a deep learning exec.")
    print("   --exec-outlogs       Get output logs for a deep learning exec.")
    print("   --exec-errlogs       Get error logs for a deep learning exec.")
    print("   --exec-trainlogs     Get train logs for a deep learning exec.")
    print("   --exec-trainoutlogs  Get train stdout log for a deep learning exec.")
    print("   --exec-trainerrlogs  Get train stderr log for a deep learning exec.")
    print("   --exec-trainresult   Get train result for a deep learning exec.")
    print("")

    print("Connect-options:")
    #print("   --master-host       Required. FQDN of IBM Watson Machine Learning Accelerator REST host.")
    #print("   --dli-rest-port     Required. IBM Watson Machine Learning Accelerator REST port. Enter -1 for OCP route.")
    print("   --rest-host          Required. FQDN of IBM Watson Machine Learning Accelerator REST host.")
    print("   --rest-port          Required. IBM Watson Machine Learning Accelerator REST port. Enter -1 for OCP route.")

    print("Logon-options:")
    print("   --username            Logon user name. Required for --logon command.")
    print("   --password            Logon user password. Required for --logon command.")
    print("")

    print("Datastore-meta:")
    print("   --cs-datastore-meta   Optional. Comma-separated string of name-value pairs. Acceptable names and values are:")
    print("                            type: 'fs'")
    print("                            data_path: Only needed for --exec-start option.")
    print("                                       For 'fs' type, this is relative path to data file system (DLI_DATA_FS)")
    print("   --data-source         Optional. Json string to describe a list of data sources. Refer to training API documentation for the format of a data source.")
    print("                             Use either this option or --cs-datastore-meta, and data-source will be used if both options are provided.")
    print("")

    print("deploy-arguments:")
    print("   --name           Optional. Deployed model name")
    print("   --runtime        Optional. Runtime to load the deployed model")
    print("   --kernel         Required. Deployed model kernel file")
    print("   --weight         Optional. Deployed model weight")
    print("   --tag            Optional. Tag the deployed model")
    print("   --attributes     Optional. Additional attributes required during model serving")
    print("   --envs           Optional. Additional environment variables required during model serving")
    print("")

    print("Submit-basic-arguments:")
    print("   <framework-name>      Required. Name of a deep learning framework returned by --dl-frameworks command.")
    print("   --model-main          Required. Main model file name.")
    print("   --model-dir           Optional. Path to a directory containing the deep learning model specified in --model-main.")
    print("   --pbmodel-name        Optional. Name of a prebuilt model such as AlexNet, VGG from TorchVision for PyTorch. You either specify")
    print("                             this option or --model-main option. You can get more info by running --dl-frameworks.")
    print("   --appName             Optional. Application name.")
    #print("   --submit-user         Optional. Submit user name.")
    print("   --consumer            Optional. Consumer path.")
    print("   --conda-env-name      Optional. Anaconda environment name to activate.")
    # print("   --conda-env-yaml    Optional. Anaconda environment yaml file.")
    # print("   --conda-package     Optional. Anaconda dependency packages.")
    print("   --numWorker           Optional. Worker number")
    print("   --workerDeviceType    Optional. Worker device type. cpu or gpu")
    print("   --workerDeviceNum     Optional. Worker device number")
    print("   --workerMemory        Optional. Worker memory")
    print("   --workerCPULimit      Optional. Worker CPU limit. For pack job only")
    #print("   --workerGPULimit      Optional. Worker GPU limit. For pack job only")
    print("   --workerMemoryLimit   Optional. Worker memory limit. For pack job only")

    #print("   --numDriver           Optional. Driver number")
    print("   --driverDeviceType    Optional. Driver device type. cpu or gpu")
    print("   --driverDeviceNum     Optional. Driver device number")
    print("   --driverMemory        Optional. Driver memory")
    print("   --driverCPULimit      Optional. Driver CPU limit. For pack job only")
    #print("   --driverGPULimit      Optional. Driver GPU limit. For pack job only")
    print("   --driverMemoryLimit   Optional. Driver memory limit. For pack job only")

    print("")

    print("Submit-metric-arguments:")
    print("   --cs-rmq-meta         Optional. RabbitMQ info for metric forwarding. Comma-separated string of name-value pairs.")
    print("   --cs-url-meta         Optional. Rest URL for metric forwarding.")
    print("   --cs-url-bearer       Optional. Bearer token for metric forwarding.")
    print("")

    print("Submit-advance-arguments:")
    print("   --msd-env                         Optional. Environment variable. --msd-env <name>=<value>")
    print("   --msd-attr                        Optional. Attribute variable. --msd-attr <name>=<value>")
    print("   --msd-image-name                  Optional. Docker image for woker pod.")
    print("   --msd-image-pull-secret           Optional. The secret name to pull docker image for woker pod.")
    print("   --msd-image-pull-policy           Optional. The policy to pull docker image for woker pod.")

    print("   --msd-priority                    Optional. Job priority, an valid integer greater than 0")
    print("   --msd-task0-node-selector         Optional. Node selector for task0 pod.")
    print("   --msd-task12n-node-selector       Optional. Node selector for task12n pod.")
    print("   --msd-pending-timeout             Optional. Job pending timeout in seconds.")
    print("   --lsf-gpu-syntax                  Optional. LSF gpu syntax to require gpu resource from LSF.")
    print("   --msd-podaffinity-rule            Optional. Pod affinity rule. preferred or required")
    print("   --msd-podaffinity-topology-key    Optional. Pod affinity topology key. which is the key for the node label that the system uses to denote such a topology domain")
    print("   --msd-pack-id                     Optional. Pack id for the job.")

    print("   [options]             Any model specific options.")
    print("")

    print("Other options:")
    print("   --jwt-token           Optional. JSON web token.")
    print("   --debug-level         Optional. Log level. Choices are: debug, info, warn, error")
    print("   --query-args          Optional. Rest query arguments. Only use with --exec-get-all and --hpo-get-all command.")
    print("")

    print("Examples:")
    print("   $ python " + module_name + " --logon --rest-host abc.ibm.com rest-port %s --username Admin --password Admin" % (str(DEFAULT_DLI_REST_PORT)))
    print("   $ python " + module_name + " --dl-frameworks --rest-host abc.ibm.com rest-port %s" % (str(DEFAULT_DLI_REST_PORT)))
    print("   $ python " + module_name + " --exec-start tensorflow --rest-host abc.ibm.com rest-port %s --cs-datastore-meta type=fs,data_path=mnist --model-main mnist.py" % (str(DEFAULT_DLI_REST_PORT)))
    print("   $ python " + module_name + " --exec-start tensorflow --rest-host abc.ibm.com rest-port %s --data-source '[{\"type\": \"fs\", \"location\": {\"paths\": \"mnist\"}}]' --model-main mnist.py" % (str(DEFAULT_DLI_REST_PORT)))
    print("   $ python " + module_name + " --exec-get job-12345 --rest-host abc.ibm.com rest-port %s" % (str(DEFAULT_DLI_REST_PORT)))
    print("   $ python " + module_name + " --exec-get-all --rest-host abc.ibm.com rest-port %s --query-args \"limit=10&state=FINISHED&sort_by=id:desc\"" % (str(DEFAULT_DLI_REST_PORT)))

def main():
    global flags, unknown_flags

    if len(sys.argv) < 2:
        print_help(os.path.basename(sys.argv[0]))
        sys.exit(1)

    # get the command
    cmd = sys.argv[1]

    if cmd in ["--help", "-h"]:
        print_help(os.path.basename(sys.argv[0]))
        sys.exit(0)

    parser = argparse.ArgumentParser(description='dlicmd options', usage=argparse.SUPPRESS)

    # logon commands
    parser.add_argument('--logon', action='store_true', default=False, dest="logon")
    parser.add_argument('--logoff', action='store_true', default=False, dest="logoff")

    # batch command
    parser.add_argument('--dl-frameworks', action='store_true', default=False, dest="frameworks_get_all")
    parser.add_argument("--exec-start", type=str, default=None, dest="framework")
    parser.add_argument('--exec-get-all', action='store_true', default=False, dest="exec_get_all")
    parser.add_argument("--exec-get", type=str, default=None, dest="batch_id")
    parser.add_argument("--exec-stop", type=str, default=None, dest="batch_id")
    parser.add_argument("--exec-delete", type=str, default=None, dest="batch_id")
    parser.add_argument('--exec-delete-all', action='store_true', default=False, dest="exec_delete_all")
    parser.add_argument("--exec-launcherlogs", type=str, default=None, dest="batch_id")
    parser.add_argument("--exec-outlogs", type=str, default=None, dest="batch_id")
    parser.add_argument("--exec-errlogs", type=str, default=None, dest="batch_id")
    parser.add_argument("--exec-trainlogs", type=str, default=None, dest="batch_id")
    parser.add_argument("--exec-trainoutlogs", type=str, default=None, dest="batch_id")
    parser.add_argument("--exec-trainerrlogs", type=str, default=None, dest="batch_id")
    parser.add_argument("--exec-trainresult", type=str, default=None, dest="batch_id")

    # resource plan command
    parser.add_argument('--rp-get-all', action='store_true', default=False, dest="rp_get_all")
    parser.add_argument("--rp-get", type=str, default=None, dest="rp_name")
    parser.add_argument("--rp-delete", type=str, default=None, dest="rp_name")

    # application command
    parser.add_argument('--app-get-all', action='store_true', default=False, dest="app_get_all")
    parser.add_argument("--app-get", type=str, default=None, dest="app_id")
    parser.add_argument("--app-wait", type=str, default=None, dest="app_id")
    parser.add_argument("--app-stop", type=str, default=None, dest="app_id_list", nargs='+')
    parser.add_argument('--app-stop-all', action='store_true', default=False, dest="app_stop_all")
    parser.add_argument("--app-analyze", type=int, default=DEFAULT_ANALYZE_HOURS, dest="num_hours")
    parser.add_argument("--app-launcherlogs", type=str, default=None, dest="app_id")
    parser.add_argument("--app-outlogs", type=str, default=None, dest="app_id")
    parser.add_argument("--app-errlogs", type=str, default=None, dest="app_id")

    parser.add_argument("--app-search-start", type=int, default=0, dest="app_search_start")
    parser.add_argument("--app-search-length", type=int, default=DEFAULT_HISTORY_LENGTH, dest="app_search_length")
    
    # hpo command
    parser.add_argument('--hpo-start', type=str, default=None, dest="hpo_input")
    parser.add_argument('--hpo-get-all', action='store_true', default=False, dest="hpo_get_all")
    parser.add_argument("--hpo-get", type=str, default=None, dest="hpo_id")
    parser.add_argument("--hpo-stop", type=str, default=None, dest="hpo_id_list", nargs='+')
    parser.add_argument("--hpo-restart", type=str, default=None, dest="hpo_id")
    parser.add_argument("--hpo-delete", type=str, default=None, dest="hpo_id_list", nargs='+')
    parser.add_argument('--hpo-algorithm-get-all', action='store_true', default=False, dest="hpo_algorithm_get_all")
    parser.add_argument("--hpo-algorithm-get", type=str, default=None, dest="hpo_algorithm")
    parser.add_argument("--hpo-algorithm-delete", type=str, default=None, dest="hpo_algorithm")

    # k8s object command
    parser.add_argument('--k8s-obj-get-all', type=str, default=None, dest="obj_type")
    parser.add_argument('--k8s-obj-get', type=str, default=None, dest="obj_get_params", nargs='+')

    # get/list output options
    parser.add_argument("-o", "--output", type=str, default=None, dest="output_format")

    # connect options
    parser.add_argument("--rest-host", "--master-host", type=str, default=None, dest="rest_host")
    parser.add_argument("--rest-port", "--dli-rest-port", type=int, default=DEFAULT_DLI_REST_PORT, dest="rest_port")

    # logon options
    parser.add_argument("--username", type=str, default=None, dest="username")
    parser.add_argument("--password", type=str, default=None, dest="password")

    # submit legacy command
    parser.add_argument("--ig", type=str, default=None, dest="instance_group")

    # submit basic arguments
    parser.add_argument("--model-main", type=str, default=None, dest="model_main")
    parser.add_argument("--model-dir", type=str, default=None, dest="model_dir")
    parser.add_argument('--user-cmd', action='store_true', default=False, dest="user_cmd")
    parser.add_argument("--pbmodel-name", type=str, default=None, dest="pb_model_name")
    parser.add_argument("--pbmodel-cmd", type=str, default=None, dest="pb_model_cmd")
    parser.add_argument("--appName", type=str, default=None, dest="app_name")
    parser.add_argument("--submit-user", type=str, default=None, dest="submit_user")
    parser.add_argument("--consumer", type=str, default=None, dest="consumer")
    parser.add_argument("--conda-home", type=str, default=None, dest="conda_home")
    parser.add_argument("--conda-env-yaml", type=str, default=None, dest="conda_env_yaml")
    parser.add_argument("--conda-package", type=str, default=None, dest="conda_package")
    parser.add_argument("--conda-env-name", type=str, default=None, dest="conda_env_name")
    parser.add_argument("--cs-datastore-meta", type=str, default=None, dest="cs_datastore_meta")
    parser.add_argument("--numWorker", type=int, default=None, dest="num_worker")
    parser.add_argument("--workerDeviceType", type=str, default="", dest="worker_device_type", choices=["cpu", "gpu"])
    parser.add_argument("--workerDeviceNum", type=str, default=None, dest="worker_device_num")
    parser.add_argument("--workerMemory", type=str, default=None, dest="worker_memory")
    parser.add_argument("--workerCPULimit", type=str, default=None, dest="worker_cpu_limit")
    parser.add_argument("--workerGPULimit", type=str, default=None, dest="worker_gpu_limit")
    parser.add_argument("--workerMemoryLimit", type=str, default=None, dest="worker_mem_limit")
    parser.add_argument("--numPs", type=int, default=None, dest="num_ps")
    parser.add_argument("--numDriver", type=int, default=None, dest="num_driver")
    parser.add_argument("--driverDeviceType", type=str, default="", dest="driver_device_type", choices=["cpu", "gpu"])
    parser.add_argument("--driverDeviceNum", type=str, default=None, dest="driver_device_num")
    parser.add_argument("--driverMemory", type=str, default=None, dest="driver_memory")
    parser.add_argument("--driverCPULimit", type=str, default=None, dest="driver_cpu_limit")
    parser.add_argument("--driverGPULimit", type=str, default=None, dest="driver_gpu_limit")
    parser.add_argument("--driverMemoryLimit", type=str, default=None, dest="driver_mem_limit")
    parser.add_argument("--sc-runasuser", type=int, default=None, dest="sc_runasuser")
    parser.add_argument("--sc-runasgroup", type=int, default=None, dest="sc_runasgroup")
    parser.add_argument("--sc-fsgroup", type=int, default=None, dest="sc_fsgroup")

    # submit metric arguments
    parser.add_argument("--cs-rmq-meta", type=str, default=None, dest="cs_rmq_meta")
    parser.add_argument("--cs-url-meta", type=str, default=None, dest="cs_url_meta")
    parser.add_argument("--cs-url-bearer", type=str, default=None, dest="cs_url_bearer")

    # submit advance arguments
    parser.add_argument("--msd-env", type=str, default=None, dest="msd_env", action='append')
    parser.add_argument("--msd-attr", type=str, default=None, dest="msd_attr", action='append')
    parser.add_argument("--msd-label", type=str, default=None, dest="msd_label", action='append')
    parser.add_argument("--msd-image-name", type=str, default=None, dest="msd_image_name")
    parser.add_argument("--msd-image-pull-secret", type=str, default=None, dest="msd_image_pull_secret")
    parser.add_argument("--msd-image-pull-policy", type=str, default=None, dest="msd_image_pull_policy", choices=["Always", "Never", "IfNotPresent"])
    parser.add_argument("--msd-priority", type=int, default=None, dest="msd_priority")
    parser.add_argument("--msd-task0-node-selector", type=str, default=None, dest="msd_task0_node_selector")
    parser.add_argument("--msd-task12n-node-selector", type=str, default=None, dest="msd_task12n_node_selector")
    parser.add_argument("--msd-pending-timeout", type=int, default=None, dest="msd_pending_timeout")
    parser.add_argument("--lsf-gpu-syntax", type=str, default=None, dest="lsf_gpu_syntax")
    parser.add_argument("--msd-podaffinity-rule", type=str, default=None, dest="msd_podaffinity_rule")
    parser.add_argument("--msd-podaffinity-topology-key", type=str, default=None, dest="msd_podaffinity_topology_key")
    parser.add_argument("--msd-pack-id", type=str, default=None, dest="msd_pack_id")

    # other options
    parser.add_argument("--jwt-token", type=str, default=None, dest="jwt_token")
    parser.add_argument("--debug-level", type=str, default="info", dest="debug_level", choices=["debug", "info", "warn", "error"])
    parser.add_argument("--query-args", type=str, default=None, dest="query_args")

    sep_index = -1
    try:
        sep_index = sys.argv.index("--")
    except ValueError:
        pass

    if sep_index != -1:
        #print(sys.argv[1:sep_index])
        #print(sys.argv[sep_index + 1: len(sys.argv)])
        flags, _ = parser.parse_known_args(args=sys.argv[1:sep_index])
        user_flags = sys.argv[sep_index + 1: len(sys.argv)]
    else:
        flags, user_flags = parser.parse_known_args()

    # drop options should not pass to dli rest api
    i = 1
    input_argv = []
    data_source = None
    # get options for commands
    while i < len(sys.argv):
        #print(sys.argv[i])
        opt_key = sys.argv[i]
        if opt_key in ["--model-dir"]:
            input_argv.append(opt_key)
            input_argv.append(os.path.basename(strip_path(sys.argv[i + 1])))
            i += 2
        elif opt_key in ["--rest-host", "--rest-port", "--master-host", "--dli-rest-port", "--username", "--password", "--jwt-token", "--query-args"]:
            i += 2
        elif opt_key in ["-o", "--output"]:
            i += 1
        elif opt_key in ["--data-source"]:
            data_source = sys.argv[i+1]
            i += 2
        else:
            input_argv.append(sys.argv[i])
            i += 1

    #print(input_argv)
    #print(user_flags)

    # check batch-get, etc... arguments
    if cmd == '--exec-start':
        if not flags.model_dir and not flags.model_main and not flags.pb_model_name and not flags.user_cmd:
            my_print(ERROR, 'model-main or pbmodel-name or cmd-str is required')
            sys.exit(1)

        if flags.pb_model_name:
            if not flags.pb_model_cmd:
                my_print(ERROR, 'pbmodel-cmd is required')
                sys.exit(1)

        if flags.user_cmd:
            if sep_index == -1 or not user_flags:
                my_print(ERROR, "-- COMMAND [args...] is required")
                sys.exit(1)

    if cmd == "--logon":
        logon()
        sys.exit(0)

    if cmd == "--logoff":
        logoff()
        sys.exit(0)

    #print(flags)
    #print(user_flags)

    check_cookies()
    get_info_from_dli_rest()

    if cmd == '--status':
        get_status()
    elif cmd == '--dl-frameworks':
        list_framework()
    elif cmd == '--exec-start':
        create_batch(input_argv, data_source)
    elif cmd == '--exec-get':
        get_batch_detail(flags.batch_id)
    elif cmd == '--exec-get-all':
        list_batch()
    elif cmd == '--exec-stop':
        stop_batch(flags.batch_id)
    elif cmd == '--exec-deploy':
        deploy_batch(input_argv)
    elif cmd == '--exec-delete':
        delete_batch(flags.batch_id)
    elif cmd == '--exec-delete-all':
        delete_batch_all()
    elif cmd in ['--exec-launcherlogs', '--exec-outlogs', '--exec-errlogs']:
        get_batch_exec_log(cmd, flags.batch_id)
    elif cmd in ['--exec-trainlogs', '--exec-trainoutlogs', '--exec-trainerrlogs']:
        get_batch_train_log(cmd, flags.batch_id, input_argv)
    elif cmd == '--exec-trainresult':
        get_train_result(flags.batch_id, input_argv)
    elif cmd == '--rp-get-all':
        list_rp()
    elif cmd == '--rp-get':
        get_rp(flags.rp_name)
    elif cmd == '--rp-delete':
        delete_rp(flags.rp_name)
    elif cmd == '--pj-get-all':
        list_pj()
    elif cmd == "--app-get":
        get_app(flags.app_id)
    elif cmd == "--app-wait":
        wait_app(flags.app_id)
    elif cmd == "--app-get-all":
        list_app()
    elif cmd == "--app-history-get-all":
        list_app_history()
    elif cmd == "--app-stop":
        stop_app(flags.app_id_list)
    elif cmd == "--app-stop-all":
        stop_app_all()
    elif cmd == "--app-analyze":
        analyze_app(flags.num_hours)
    elif cmd in ["--app-launcherlogs", "--app-outlogs", "--app-errlogs"]:
        get_app_logs(cmd, flags.app_id)
    elif cmd == "--k8s-obj-get-all":
        get_k8s_obj_all(flags.obj_type)
    elif cmd == "--k8s-obj-get":
        get_k8s_obj(flags.obj_get_params)
    elif cmd == "--hpo-start":
        start_hpo()
    elif cmd == "--hpo-get-all":
        list_hpo()
    elif cmd == "--hpo-get":
        get_hpo(flags.hpo_id)
    elif cmd == "--hpo-stop":
        stop_hpo(flags.hpo_id_list)
    elif cmd == "--hpo-restart":
        restart_hpo(flags.hpo_id)
    elif cmd == "--hpo-delete":
        delete_hpo(flags.hpo_id_list)
    elif cmd == "--hpo-delete-all":
        delete_hpo_all()
    elif cmd == "--hpo-algorithm-get-all":
        list_hpo_algorithm()
    elif cmd == "--hpo-algorithm-get":
        get_hpo_algorithm(flags.hpo_algorithm)
    elif cmd == "--hpo-algorithm-delete":
        delete_hpo_algorithm(flags.hpo_algorithm)
    else:
        print_help(os.path.basename(sys.argv[0]))
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
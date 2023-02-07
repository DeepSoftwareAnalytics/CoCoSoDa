import pickle
import os
import json
import prettytable as pt
import numpy as np
import math
import logging
logger = logging.getLogger(__name__)

def read_json_file(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    if len(data) == 1:
        data = json.loads(data[0])
    else:
        data = [json.loads(line) for line in data]
    return data 


def save_json_data(data_dir, filename, data):
    os.makedirs(data_dir, exist_ok=True)
    file_name = os.path.join(data_dir, filename)
    with open(file_name, 'w') as output:
        if type(data) == list:
            if type(data[0]) in [str, list,dict]:
                for item in data:
                    output.write(json.dumps(item))
                    output.write('\n')

            else:
                json.dump(data, output)
        elif type(data) == dict:
            json.dump(data, output)
        else:
            raise RuntimeError('Unsupported type: %s' % type(data))
    logger.info("saved dataset in " + file_name)

def save_pickle_data(path_dir, filename, data):
    full_path = path_dir + '/' + filename
    print("Save dataset to: %s" % full_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    with open(full_path, 'wb') as output:
        pickle.dump(data, output,protocol=4)


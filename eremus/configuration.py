import os
import json

class NotAllowedKey(Exception):
    pass

def config(key, value):
    """
    Changes the configuration file adding a key with a specific value.
    If key exists, it is updated.
    
    Parameters
    ------------
    key : str
        a key, indicating what we are configuring
    value : object
        corresponding value
    """
   
    # define configuration file
    conf_file = "configuration.txt"

    # check if key is allowed
    allowed_keys = {'path_to_eremus_data'}

    if key not in allowed_keys:
        raise NotAllowedKey(f"the key {key} is not allowed in this configuration file. Please use one of the following keys: {allowed_keys}")

    # check file exists
    if not os.path.exists(conf_file):
        with open(conf_file, 'x'):
            pass

    # check empty files
    empty = os.stat(conf_file).st_size == 0

    if empty:
        configuration = {}
    else:
        with open(conf_file, 'r') as json_file:
            configuration = json.load(json_file)

    with open(conf_file, 'w') as json_file:
        configuration[key] = value
        print(configuration)
        json.dump(configuration, json_file)
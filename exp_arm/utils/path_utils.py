import yaml
import os

# Load a yaml file (e.g. simu config file)
def load_yaml_file(yaml_file):
    '''
    Load config file (yaml)
    '''
    with open(yaml_file) as f:
        data = yaml.load(f)
    return data 

# Load config file
def load_config_file(config_name):
    '''
    Loads YAML config file in demos/config as a dict
    '''
    config_path = os.path.abspath(os.path.join(os.path.abspath(__file__ + "/../../../"), 'config/ocp_params'))
    config_file = config_path+"/"+config_name+".yml"
    config = load_yaml_file(config_file)
    return config



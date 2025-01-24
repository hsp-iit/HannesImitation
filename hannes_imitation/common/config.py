import yaml

def load_configuration(config_path='config.yaml'):

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config
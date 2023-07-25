import yaml

def experiment_config(conf_file, experiment):
    '''
    Load experiment settings from config YAML file
    '''
    with open(conf_file) as cf:
        conf = yaml.safe_load(cf)

    if experiment not in conf['experiments']:
        raise ValueError(f"Experiment '{experiment}' not in config file")
    return {**conf['default_experiment'], **conf['experiments'][experiment]}

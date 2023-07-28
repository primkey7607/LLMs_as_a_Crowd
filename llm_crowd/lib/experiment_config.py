import yaml

from llm_crowd.lib.utils import CONFDIR

def experiment_config(task: str, experiment: str):
    '''
    Load experiment settings from config YAML file
    '''
    with open(CONFDIR / f'{task}.yaml') as cf:
        conf = yaml.safe_load(cf)

    if experiment not in conf['experiments']:
        raise ValueError(f"Experiment '{experiment}' not in config file")
    return {**conf['default_experiment'], **conf['experiments'][experiment]}

def all_experiment_configs(task: str):
    with open(CONFDIR / f'{task}.yaml') as cf:
        conf = yaml.safe_load(cf)

    return {
        exp: {**conf['default_experiment'], **exp_conf}
        for exp, exp_conf in conf['experiments'].items()}

def custom_crowds(task: str):
    with open(CONFDIR / f'{task}.yaml') as cf:
        conf = yaml.safe_load(cf)

    if 'custom_crowds' in conf:
        return conf['custom_crowds']
    return {}

import yaml

from llm_crowd.lib.utils import CONFDIR

def experiment_config(task, experiment):
    '''
    Load experiment settings from config YAML file
    '''
    with open(CONFDIR / f'{task}.yaml') as cf:
        conf = yaml.safe_load(cf)

    if experiment not in conf['experiments']:
        raise ValueError(f"Experiment '{experiment}' not in config file")
    return {**conf['default_experiment'], **conf['experiments'][experiment]}

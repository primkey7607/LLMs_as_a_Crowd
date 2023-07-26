import argparse
from pathlib import Path

import pandas as pd

from llm_crowd.lib.utils import ROOTDIR, DATADIR
from llm_crowd.lib.experiment_config import experiment_config
from llm_crowd.lib.experiment import write_responses, combine_responses
from llm_crowd.entity_resolution.prompt_template import ERPromptTemplate

NUM_SHOTS = 2

ERDIR = DATADIR / 'entity_resolution'

def load_testsets(exp_conf):
    return {d: pd.read_csv(ERDIR / 'test' / f"{d}.csv") for d in exp_conf['datasets']}

def load_shots_datasets(exp_conf, size='small'):
    if exp_conf['shots'] == 'none':
        return {}

    if exp_conf['shots_dataset'] is None:
        shots_datasets = exp_conf['datasets']
    else:
        shots_datasets = [exp_conf['shots_dataset']]

    return {d: load_shots_dataset(d) for d in shots_datasets}

def load_shots_dataset(dataset, size='small'):
    df = pd.read_csv(ERDIR / 'train' / f'{dataset}-{size}.csv')

    shot_yes = df[df['label']].sample(n=len(df)*shots, replace=True, random_state=100).reset_index(drop=True)
    shot_no = df[~df['label']].sample(n=len(df)*shots, replace=True, random_state=100).reset_index(drop=True)
    return (shot_yes, shot_no)

def init_templates(exp_conf):
    return [
        ERPromptTemplate(t, exp_conf['cot'], exp_conf['temperature'])
        for t in exp_conf['templates']]

def examples_funcs(shots_datasets):
    def examples_none(dataset, row_idx, rep):
        return []

    def examples_regular(dataset, row_idx, rep):
        '''
        Shots vary by training set row
        '''
        examples = []
        yes_df, no_df = shots_datasets[dataset]
        for i in range(NUM_SHOTS):
            shot_idx = row_idx * NUM_SHOTS + i
            yes_example = yes_df.loc[shot_idx]
            no_example = no_df.loc[shot_idx]
            examples.append((yes_example['left'], yes_example['right'], yes_example['label']))
            examples.append((no_example['left'], no_example['right'], no_example['label']))
        return examples

    def examples_uniform(dataset, row_idx, rep):
        '''
        Shots are uniform across training set row but vary by rep
        '''
        examples = []
        yes_df, no_df = shots_datasets[dataset]
        for i in range(NUM_SHOTS):
            shot_idx = rep * NUM_SHOTS + i
            yes_example = yes_df.loc[shot_idx]
            no_example = no_df.loc[shot_idx]
            examples.append((yes_example['left'], yes_example['right'], yes_example['label']))
            examples.append((no_example['left'], no_example['right'], no_example['label']))
        return examples

    return {'none': examples_none, 'regular': examples_regular, 'uniform': examples_uniform}

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("experiment")
    args = parser.parse_args()

    exp_conf = experiment_config(args.config, args.experiment)

    exp_dir = Path(ROOTDIR / 'out' / 'entity_resolution' / args.experiment)
    raw_dir = exp_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_testsets(exp_conf)

    prompt_templates = init_templates(exp_conf)

    shots_datasets = load_shots_datasets(exp_conf)
    examples_func = examples_funcs(shots_datasets)[exp_conf['shots']]

    write_responses(raw_dir, datasets, prompt_templates, examples_func, exp_conf['reps'])
    combine_responses(raw_dir, exp_dir / 'responses.csv')

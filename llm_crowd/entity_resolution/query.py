import argparse

from llm_crowd.lib.utils import ROOTDIR, DATADIR
from llm_crowd.lib.experiment_config import experiment_config
from llm_crowd.lib.experiment import write_responses, combine_responses, train_df, test_df
from llm_crowd.entity_resolution.prompt_template import ERPromptTemplate

NUM_SHOTS = 2

def main(experiment):
    exp_conf = experiment_config('entity_resolution', experiment)

    datasets = load_testsets(exp_conf)
    datasets = {d: test_df('entity_resolution', d) for d in exp_conf['datasets']} 

    prompt_templates = init_templates(exp_conf)

    shots_datasets = load_shots_datasets(exp_conf)
    examples_func = examples_funcs(shots_datasets)[exp_conf['shots']]

    write_responses(
        'entity_resolution',
        experiment,
        datasets,
        prompt_templates,
        examples_func,
        exp_conf['reps'])
    combine_responses('entity_resolution', experiment)

def load_testsets(exp_conf):
    return {d: test_df('entity_resolution', d) for d in exp_conf['datasets']}

def load_shots_datasets(exp_conf, size='small'):
    if exp_conf['shots'] == 'none':
        return {}

    if exp_conf['shots_dataset'] is None:
        shots_datasets = exp_conf['datasets']
    else:
        shots_datasets = [exp_conf['shots_dataset']]

    return {d: load_shots_dataset(d) for d in shots_datasets}

def load_shots_dataset(dataset, size='small'):
    df = train_df('entity_resolution', dataset, size)

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
    parser.add_argument("experiment")
    args = parser.parse_args()

    main(args.experiment)

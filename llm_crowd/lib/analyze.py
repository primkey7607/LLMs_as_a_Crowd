import numpy as np
import pandas as pd
import sklearn
import yaml
import argparse
from pathlib import Path

from llm_crowd.lib.crowd_methods import CROWD_METHODS
from llm_crowd.lib.experiment import task_dir, experiment_dir
from llm_crowd.lib.experiment_config import all_experiment_configs


def main(task: str):
    template_dfs = []
    crowd_dfs = []

    finished_experiments = set()
    for exp, conf in all_experiment_configs(task).items():
        exp_dir = experiment_dir(task, exp)
        infile = exp_dir / 'responses.csv'
        if not infile.exists():
            continue
        print(f"Experiment {exp}...")
        finished_experiments.add(exp)

        df = pd.read_csv(infile, usecols=['template', 'dataset', 'row', 'rep', 'answer', 'truth'])
        truth = df.groupby(['dataset', 'row'])['truth'].mean().astype(int).reset_index()
        df = df.drop(columns=['truth'])
        df.loc[df['answer'] == -1, 'answer'] = 0

        if conf['reps_separate']:
            dfs = {}
            for rep, rep_df in df.groupby(['rep']):
                dfs[f'{exp}_{rep}'] = rep_df
        else:
            dfs = {exp: df}
        for sub_exp, df in dfs.items():
            template_f1s = calculate_template_f1s(df, truth)
            template_f1s['experiment'] = sub_exp
            template_dfs.append(template_f1s)

            crowd_f1s = calculate_crowd_f1s(df, truth, conf['crowd_methods'])
            crowd_f1s['experiment'] = sub_exp
            crowd_dfs.append(crowd_f1s)

    template_df = pd.concat(template_dfs)[['experiment', 'dataset', 'template', 'F1']]
    crowd_df = pd.concat(crowd_dfs)[['experiment', 'dataset', 'method', 'F1']]

    tdir = task_dir(task)
    template_df.to_csv(tdir / 'templates.csv', index=False)
    crowd_df.to_csv(tdir / 'crowds.csv', index=False)
            

def calculate_template_f1s(df, truth):
    # Handle duplicates
    df = df.groupby(['dataset', 'template', 'row']).mean() > 0.5
    df = df.reset_index()

    out = df[['dataset', 'template']].drop_duplicates()
    out['F1'] = -100.0
    out = out.set_index(['dataset', 'template'])
    df = df.merge(truth, on=['dataset', 'row'])
    for (dataset, template), df_group in df.groupby(['dataset', 'template']):
        f1 = sklearn.metrics.f1_score(df_group['truth'], df_group['answer']) * 100
        out.loc[(dataset, template)] = f1
    out = out.reset_index()
    return out


def calculate_crowd_f1s(df, truth, methods):
    out = {'dataset': [], 'method': [], 'F1': []}

    df = df.rename(columns={'template': 'worker', 'row': 'task', 'answer': 'label'})

    for dataset, df_dataset in df.groupby('dataset'):
        truth_dataset = truth[truth['dataset'] == dataset]
        for method in methods:
            args = [df_dataset]
            if method == 'GoldStandard':
                args.append(truth_dataset)
            prediction = CROWD_METHODS[method]().fit_predict(*args)
            prediction = truth_dataset.join(prediction)
            f1 = sklearn.metrics.f1_score(prediction['truth'], prediction['agg_label']) * 100
            out['dataset'].append(dataset)
            out['method'].append(method)
            out['F1'].append(f1)

    out = pd.DataFrame(out)
    return out


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    args = parser.parse_args()

    main(args.task)

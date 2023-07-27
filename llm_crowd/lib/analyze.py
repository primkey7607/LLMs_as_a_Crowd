import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin
from scipy import stats
import sklearn
import yaml

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

        exp_template_dfs = []
        exp_crowd_dfs = []
        for sub_exp, sub_df in dfs.items():
            template_f1s = calculate_template_f1s(sub_df, truth)
            template_f1s['experiment'] = sub_exp
            exp_template_dfs.append(template_f1s)

            crowd_f1s = calculate_crowd_f1s(sub_df, truth, conf['crowd_methods'])
            crowd_f1s['experiment'] = sub_exp
            exp_crowd_dfs.append(crowd_f1s)
        template_dfs += exp_template_dfs
        crowd_dfs += exp_crowd_dfs

        if 'correlations' in conf['extra_analysis']:
            df_corrs = independence_partial_correlations(df, truth)
            df_corrs.to_csv(exp_dir / 'corrs.csv', index=False)
        if 'stdev' in conf['extra_analysis']:
            exp_template_df = pd.concat(exp_template_dfs)
            exp_crowd_df = pd.concat(exp_crowd_dfs)
            stdev_df = standard_deviations(exp_template_df, exp_crowd_df)
            stdev_df.to_csv(exp_dir / 'stdevs.csv', index=False)
            

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


def independence_partial_correlations(df, truth):
    '''
    Return DF with a row for each worker pair, with the conditional independence and partial
    correlations (controlling for true label) reported
    '''
    df2 = df.copy()
    df2['template'] = 'vp2'
    df = pd.concat([df, df2])
    templates = df['template'].unique()
    
    df_pivot = df.pivot(columns='template', index=['dataset', 'row'], values='answer').reset_index()
    df_pivot = df_pivot.merge(truth, on=['dataset', 'row'])

    df_true = df_pivot[df_pivot['truth'] == 1]
    df_false = df_pivot[df_pivot['truth'] == 0]

    out = {'template1': [], 'template2': [], 'chi2_stat': [], 'partial_corr': []}

    for i in range(len(templates)):
        t1 = templates[i]
        for j in range(i+1, len(templates)):
            t2 = templates[j]

            test_stat = min(independence_test(df_true, t1, t2), independence_test(df_false, t1, t2))
            corr = df_pivot.partial_corr(x=t1, y=t2, covar='truth').loc['pearson', 'r']
            out['template1'].append(t1)
            out['template2'].append(t2)
            out['chi2_stat'].append(test_stat)
            out['partial_corr'].append(corr)

    return pd.DataFrame(out)

def independence_test(df, col1, col2):
    res = stats.chi2_contingency(pd.crosstab(df[col1], df[col2]))
    return res[1]

def standard_deviations(template_df, crowd_df):
    df = pd.concat(template_df.rename(columns={'template': 'method'}), crowd_df)
    df = df.groupby(['dataset', 'method'])['F1'].std().reset_index()
    return df.rename(columns={'F1': 'F1_stdev'})

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    args = parser.parse_args()

    main(args.task)

from crowdkit.aggregation import MajorityVote, Wawa, DawidSkene, GLAD

from llm_crowd.lib.crowd_bwa import bwa
from llm_crowd.lib.crowd_ebcc import ebcc_vb

class EBCC:
    def fit_predict(self, df):
        df['worker_codes'], _ = pd.factorize(df['worker'])
        elbos = []
        results = []
        for _ in range(40):
            seed = np.random.randint(1e8)
            prediction, elbo = ebcc_vb(df[['task', 'worker_codes', 'label']].values, num_groups=10, seed=seed, empirical_prior=True)
            elbos.append(elbo)
            results.append((prediction, seed, elbo))
        predictions, seed, elbo = results[np.argmax(elbos)]
        out = pd.Series(predictions.argmax(axis=1), name='agg_label')
        out.index.name = 'task'
        return out

class BWA:
    def fit_predict(self, df):
        df['worker_codes'], _ = pd.factorize(df['worker'])
        predictions = bwa(df[['task', 'worker_codes', 'label']].values)
        out = pd.Series(predictions.argmax(axis=1), name='agg_label')
        out.index.name = 'task'
        return out

class GoldStandard:
    def fit_predict(self, df, truth):
        workers = list(df['worker'].unique())
        df = df.groupby(['worker', 'task']).mean().reset_index()
        df_pivot = df.pivot(columns='worker', index='task', values='label').join(truth)
        grouped = df_pivot.groupby(workers).mean().reset_index()
        grouped['agg_label'] = 0
        grouped.loc[grouped['truth'] > 0.5, 'agg_label'] = 1
        predictions = df_pivot.reset_index().merge(grouped[workers + ['agg_label']], on=workers)
        return predictions.set_index('task')['agg_label']

CROWD_METHODS = {
    'DawidSkene': DawidSkene,
    'MajorityVote': MajorityVote,
    'Wawa': Wawa,
    'GLAD': GLAD,
    'EBCC': EBCC,
    'BWA': BWA,
    'GoldStandard': GoldStandard}

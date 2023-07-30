import pandas as pd
import random
import itertools
import os

#NOTE: upon manually reviewing the 100 non-FK table pairs
#generated, we find that they are, indeed, incorrect
#with the given random seed.
def generate_sample(n=100):
    out_schema = ['Input Table', 'FK Table', 'Ground Truth']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    truedf = pd.read_csv('../chembl_groundtruth.csv')
    tbl_lst = os.listdir('../chembl_csvs')
    
    pairs = list(itertools.combinations(tbl_lst, 2))
    true_pairs = [pair[0] for pair in truedf.groupby(['Input Table', 'FK Table'])]
    bad_pairs = [pair for pair in pairs if pair not in true_pairs]
    
    random.seed(3)
    bad_sample = random.sample(bad_pairs, n)
    for b in bad_sample:
        out_dct['Input Table'].append(b[0])
        out_dct['FK Table'].append(b[1])
        out_dct['Ground Truth'].append(False)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv('chembl_half_half.csv', index=False)

if __name__=='__main__':
    generate_sample()
    
    
    
    
    


import pickle
import pandas as pd

'''
this function combines santos's generated results
and the ground truth for the SANTOS small benchmark.
'''
def santos_candidates_tocsv(outfname):
    out_schema = ['Input Table', 'Lake Table', 'SANTOS Unionable', 'Ground Truth']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    santos_candfile = 'stats/santos_benchmark_result_by_santos_full.pickle'
    gt_file = 'groundtruth/santosUnionBenchmark.pickle'
    
    with open(santos_candfile, 'rb') as fh:
        santos_dct = pickle.load(fh)
    
    with open(gt_file, 'rb') as fh:
        gt_dct = pickle.load(fh)
    
    for k in gt_dct:
        true_lst = gt_dct[k]
        santos_lst = santos_dct[k]
        fullset = set(true_lst).union(set(santos_lst))
        
        for cand in fullset:
            out_dct['Input Table'].append(k)
            out_dct['Lake Table'].append(cand)
            if cand in santos_lst:
                out_dct['SANTOS Unionable'].append(True)
            else:
                out_dct['SANTOS Unionable'].append(False)
            
            if cand in true_lst:
                out_dct['Ground Truth'].append(True)
            else:
                out_dct['Ground Truth'].append(False)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(outfname, index=False)

'''
this function combines santos's generated results
and the ground truth for the TUS benchmark.
'''
def tus_candidates_tocsv(outfname):
    out_schema = ['Input Table', 'Lake Table', 'SANTOS Unionable', 'Ground Truth']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    santos_candfile = 'stats/tus_benchmark_result_by_santos_full.pickle'
    gt_file = 'supplementary_materials/TUS_benchmark_relabeled_groundtruth.csv'
    
    with open(santos_candfile, 'rb') as fh:
        santos_dct = pickle.load(fh)
    
    gtdf = pd.read_csv(gt_file)
            
    
    for k in santos_dct:
        santos_lst = santos_dct[k]
        
        for cand in santos_lst:
            out_dct['Input Table'].append(k)
            out_dct['Lake Table'].append(cand)
            if cand in santos_lst:
                out_dct['SANTOS Unionable'].append(True)
            else:
                out_dct['SANTOS Unionable'].append(False)
            
            if ((gtdf['query_table'] == k) & (gtdf['data_lake_table'] == cand)).any():
                out_dct['Ground Truth'].append(True)
            else:
                out_dct['Ground Truth'].append(False)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(outfname, index=False)

def santos_topkcand_tocsv(outpref, topk=10):
    out_schema = ['Input Table', 'Lake Table', 'SANTOS Unionable', 'Ground Truth']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    santos_candfile = 'stats/santos_benchmark_result_by_santos_full.pickle'
    gt_file = 'groundtruth/santosUnionBenchmark.pickle'
    
    with open(santos_candfile, 'rb') as fh:
        santos_dct = pickle.load(fh)
    
    with open(gt_file, 'rb') as fh:
        gt_dct = pickle.load(fh)
    
    for k in gt_dct:
        true_lst = gt_dct[k]
        santos_lst = santos_dct[k][:topk]
        fullset = set(true_lst).union(set(santos_lst))
        
        for cand in fullset:
            out_dct['Input Table'].append(k)
            out_dct['Lake Table'].append(cand)
            if cand in santos_lst:
                out_dct['SANTOS Unionable'].append(True)
            else:
                out_dct['SANTOS Unionable'].append(False)
            
            if cand in true_lst:
                out_dct['Ground Truth'].append(True)
            else:
                out_dct['Ground Truth'].append(False)
    
    out_df = pd.DataFrame(out_dct)
    outfname = outpref + 'k' + str(topk) + '.csv'
    out_df.to_csv(outfname, index=False)

if __name__=='__main__':
    santos_candidates_tocsv('santos_small_gtcomparek10.csv')
    tus_candidates_tocsv('tus_gtcomparek10.csv')


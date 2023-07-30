from submodules.aurum_datadiscovery.ddapi import API
from submodules.aurum_datadiscovery.modelstore.elasticstore import StoreHandler
from submodules.aurum_datadiscovery.knowledgerepr.fieldnetwork import deserialize_network
import pandas as pd

def write_aurummhmatches(gt_file : str):
    out_schema = ['Input Table', 'Input Column', 'FK Table', 'FK Column', 'Aurum Score']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    miss_schema = ['Input Table', 'Input Column']
    miss_dct = {}
    for m in miss_schema:
        miss_dct[m] = []
    
    
    gtdf = pd.read_csv(gt_file)
    inpfiles = gtdf['Input Table'].unique().tolist()
    
    store_client = StoreHandler()
    path = 'submodules/aurum-datadiscovery/test/testmodel/'
    network = deserialize_network(path)
    api = API(network)
    api.init_store()
    
    # #first, test basic keyword search as a sanity check
    # res = api.keyword_search('activities', max_results=10)
    # for el in res:
    #     print(str(el))
    # print("Keyword Search Complete.")
    
    #sanity check passed. now testing joinability.
    for tbl_name in inpfiles:
        tbl_path = tbl_name + '.csv'
        fullpath = '../chembl_csvs/' + tbl_path
        tbl_cols = pd.read_csv(fullpath, nrows=0).columns.tolist()
        for c in tbl_cols:
            cur_field = ('chembl_repo', tbl_path, c)
            try:
                res = api.pkfk_field(cur_field)
            except KeyError:
                miss_dct['Input Table'].append(tbl_path)
                miss_dct['Input Column'].append(c)
                continue
            sortres = sorted(res, key=lambda el: el.score, reverse=True)
            print("RES size: " + str(res.size()))
            # print(type(res))
            # val_cnt = 0
            for el in sortres:
                valid = False
                if el.source_name != tbl_path:
                    valid = True
                    # val_cnt += 1
                # if val_cnt < 10 and valid:
                if valid:
                    out_dct['Input Table'].append(tbl_path)
                    out_dct['Input Column'].append(c)
                    out_dct['FK Table'].append(el.source_name)
                    out_dct['FK Column'].append(el.field_name)
                    out_dct['Aurum Score'].append(el.score)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv('aurum_fullmatches.csv', index=False)
    miss_df = pd.DataFrame(miss_dct)
    miss_df.to_csv('aurum_missingmatches.csv', index=False)

if __name__=='__main__':
    write_aurummhmatches('../chembl_groundtruth.csv')

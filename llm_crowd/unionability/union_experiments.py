import pandas as pd
import pickle
import openai
import signal
import os
import shutil
import argparse
from crowdkit.aggregation import MajorityVote, Wawa, DawidSkene
from googletrans import Translator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

'''
Purpose: Run multiple prompts with multiple temperatures,
and figure out which configuration is best for unionability. We also have functions
to analyze the results and figure out which configuration will give
the best performance.
    
'''
    
    
#we will compute P@10 and R@10, as described in the paper
def santos_performance(fname, top_k=10):
    out_schema = ['Input Table', 'Precision at k', 'Recall at k', 'Precision', 'Recall']
    outfname = 'santos_small_gtcompare_stats_k' + str(top_k) + '.csv'
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    df = pd.read_csv(fname)
    #precision at 10--number of unionable items included in the top-10
    inpfs = df['Input Table'].unique().tolist()
    for inp in inpfs:
        laketbls = df[df['Input Table'] == inp]
        tp_num = laketbls[(laketbls['SANTOS Unionable'] == True) & (laketbls['Ground Truth'] == True)].shape[0]
        fp_num = laketbls[(laketbls['SANTOS Unionable'] == True) & (laketbls['Ground Truth'] == False)].shape[0]
        fn_num = laketbls[(laketbls['SANTOS Unionable'] == False) & (laketbls['Ground Truth'] == True)].shape[0]
        tn_num = laketbls[(laketbls['SANTOS Unionable'] == False) & (laketbls['Ground Truth'] == False)].shape[0]
        
        prec_k = min(tp_num, top_k) / min(top_k, (tp_num + fp_num))
        full_prec = tp_num / (tp_num + fp_num)
        
        rec_k = min(tp_num, top_k) / min(top_k, (tp_num + fn_num))
        full_rec = tp_num / (tp_num + fn_num)
        
        out_dct['Input Table'].append(inp)
        out_dct['Precision at k'].append(prec_k)
        out_dct['Precision'].append(full_prec)
        out_dct['Recall at k'].append(rec_k)
        out_dct['Recall'].append(full_rec)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(outfname, index=False)
    
    avg_preck = out_df['Precision at k'].mean()
    avg_reck = out_df['Recall at k'].mean()
    avg_prec = out_df['Precision'].mean()
    avg_rec = out_df['Recall'].mean()
    
    print("Average Precision at " + str(top_k) + " " + str(avg_preck))
    print("Average Recall at " + str(top_k) + " " + str(avg_reck))
    print("Average Precision " + str(avg_prec))
    print("Average Recall " + str(avg_rec))

def get_candidate(r_ind, dfrow, num_samples=5):
    name1 = dfrow['Input Table']
    name2 = dfrow['Lake Table']
    
    path1 = '/home/cc/union_data/santos/benchmark/santos_benchmark/query/' + name1
    path2 = '/home/cc/union_data/santos/benchmark/santos_benchmark/datalake/' + name2
    
    exists1 = os.path.exists(path1)
    exists2 = os.path.exists(path2)
    
    if not exists1 or not exists2:
        out_schema = ['Missing Index', 'Missing Input', 'Missing Lake Table']
        out_dct = {}
        for o in out_schema:
            out_dct[o] = []
        
        out_dct['Missing Index'].append(r_ind)
        if not exists1:
            out_dct['Missing Input'].append(path1)
        else:
            out_dct['Missing Input'].append('Present.')
        
        if not exists2:
            out_dct['Missing Lake Table'].append(path2)
        else:
            out_dct['Missing Lake Table'].append('Present.')
        
        out_df = pd.DataFrame(out_dct)
        out_df.to_csv('missingcandidate_' + str(r_ind) + '.csv', index=False)
        return None
    
    df1 = pd.read_csv(path1, nrows=num_samples)
    df2 = pd.read_csv(path2, nrows=num_samples)
    st1 = df1.to_csv(index=False)
    st2 = df2.to_csv(index=False)
    return (st1, st2)

def get_candidate_fast(r_ind, dfrow):
    #i think we can trust that if r_ind is not in the dictionary,
    #it's because it's missing from the lake.
    if r_ind in all_inpstrs:
        return all_inpstrs[r_ind]
    return None

def get_allcandstrs(inp_tbls, match_file, num_samples=5):
    
    df = pd.read_csv(match_file)
    res_dct = {}
    for tbl in inp_tbls:
        dfmatches = df[df['Input Table'] == tbl]
        dfdct = dfmatches.to_dict(orient='index')
        for r_ind in dfdct:
            dfrow = dfdct[r_ind]
            name1 = dfrow['Input Table']
            name2 = dfrow['Lake Table']
            
            path1 = '/home/cc/union_data/santos/benchmark/santos_benchmark/query/' + name1
            path2 = '/home/cc/union_data/santos/benchmark/santos_benchmark/datalake/' + name2
            
            exists1 = os.path.exists(path1)
            exists2 = os.path.exists(path2)
            
            if not exists1 or not exists2:
                out_schema = ['Missing Index', 'Missing Input', 'Missing Lake Table']
                out_dct = {}
                for o in out_schema:
                    out_dct[o] = []
                
                out_dct['Missing Index'].append(r_ind)
                if not exists1:
                    out_dct['Missing Input'].append(path1)
                else:
                    out_dct['Missing Input'].append('Present.')
                
                if not exists2:
                    out_dct['Missing Lake Table'].append(path2)
                else:
                    out_dct['Missing Lake Table'].append('Present.')
                
                out_df = pd.DataFrame(out_dct)
                out_df.to_csv('missingcandidate_' + str(r_ind) + '.csv', index=False)
                continue
            
            df1 = pd.read_csv(path1, nrows=num_samples)
            df2 = pd.read_csv(path2, nrows=num_samples)
            st1 = df1.to_csv(index=False)
            st2 = df2.to_csv(index=False)
            res_dct[r_ind] = (st1, st2)
    
    return res_dct

class MyTimeoutException(Exception):
    pass

#register a handler for the timeout
def handler(signum, frame):
    print("Waited long enough!")
    raise MyTimeoutException("STOP")

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def response_suffwtemp(prompt_tmp, row1, row2, temp_val, timeout=30):
    fullprompt = prompt_tmp.get_prompt_nospaces(row1, row2)
    
    followup = 'Begin your answer with YES or NO.'
    if prompt_tmp.lang != 'english':
        translator = Translator()
        followup = translator.translate(followup, src='english', dest=fullprompt.lang).text
    
    fullprompt = fullprompt + ' ' + followup
    
    chat = [{"role": "system", "content": "You are a helpful assistant who can only answer YES or NO and then explain your reasoning."}]
    fullmsg = {"role": "user", "content": fullprompt }
    chat.append(fullmsg)
    print("Sending: {}".format(chat))
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val,
        max_tokens=30
    )
    chat_response = response["choices"][0]["message"]["content"]
    
    if prompt_tmp.lang != 'english':
        translator = Translator()
        chat_response = translator.translate(chat_response, src=prompt_tmp.lang, dest='english').text
    
    return chat_response

def parse_enresponse(response):
    if response.lower().startswith('yes'):
        return 1
    elif response.lower().startswith('no'):
        return 0
    else:
        return -1

def storysuff(inp_tbls, match_file, story_name, samp_range : list, samp_type, num_reps=10):
    story_fname = 'all_prompts/' + story_name + '_unionprompt_english.pkl'
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    
    df = pd.read_csv(match_file)
    
    for tbl in inp_tbls:
        dfmatches = df[df['Input Table'] == tbl]
        dfdct = dfmatches.to_dict(orient='index')
        for r_ind in dfdct:
            dfrow = dfdct[r_ind]
            row_gt = dfrow['Ground Truth']
            match = get_candidate_fast(r_ind, dfrow)
            if match == None:
                continue
            for i in range(num_reps):
                for sval in samp_range:
                    outname = 'unionmatchwsuff' + story_name + '-' + str(r_ind) + '-' + 'rep' + str(i) + '-' + samp_type + str(sval).replace('.', '_') + '.csv'
                    if os.path.exists(outname):
                        continue
                    
                    if samp_type == 'temperature':
                        story_response = response_suffwtemp(story_tmp, match[0], match[1], sval)
                    else:
                        raise Exception("Sampling Type not supported: {}".format(samp_type))
                    
                    story_answer = parse_enresponse(story_response)
                    outdct = {}
                    outdct['Match File'] = [match_file]
                    outdct['Row No'] = [r_ind]
                    outdct['Rep No'] = [i]
                    outdct['Sampling Type'] = [samp_type]
                    outdct['Sampling Param'] = [sval]
                    outdct['Story Name'] = [story_name]
                    outdct['Story Response'] = [story_response]
                    outdct['Story Answer'] = [story_answer]
                    outdct['Ground Truth'] = [row_gt]
                    outdf = pd.DataFrame(outdct)
                    outdf.to_csv(outname)

def extract_storyprompt(story_name, row0, row1):
    story_fname = 'all_prompts/' + story_name + '_unionprompt_english.pkl'
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    story_prompt = story_tmp.get_prompt_nospaces(row0, row1)
    return story_prompt

def combine_storiesresults(match_file, folder):
    out_schema = ['Match File', 'Row No', 'Rep No', 'Sampling Type',
                  'Sampling Param', 'Story Name', 'Story Prompt', 'Story Response', 'Story Answer', 'Ground Truth']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    matchdf = pd.read_csv(match_file)
    for partf in os.listdir(folder):
        f = os.path.join(folder, partf)
        if f.endswith('.csv'):
            df = pd.read_csv(f)
            for row in df.to_dict(orient='records'):
                r_ind = row['Row No']
                dfrow = matchdf.loc[r_ind]
                match = get_candidate(r_ind, dfrow)
                story_prompt = extract_storyprompt(row['Story Name'], match[0], match[1])
                out_dct['Story Prompt'].append(story_prompt)
                
                for c in df.columns:
                    if c in out_dct:
                        out_dct[c].append(row[c])
    
    outdf = pd.DataFrame(out_dct)
    outdf.to_csv(folder + '_full.csv', index=False)
    
def crowd_gather(fullfname, base_file, temp):
    raw_df = pd.read_csv(fullfname)
    basedf = pd.read_csv(base_file)
    df = raw_df[raw_df['Sampling Param'] == temp]
    out_schema = ['worker', 'task', 'label']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    raw_labels = df['Story Answer'].tolist()
    new_labels = [max(x, 0) for x in raw_labels] #change -1s to 0s.
    
    out_dct['worker'] = df['Story Name'].tolist()
    out_dct['label'] = new_labels
    matchfiles = df['Match File'].tolist()
    rownos = df['Row No'].tolist()
    tasklst = []
    pairslst = []
    
    for i in range(len(matchfiles)):
        new_el = matchfiles[i] + ':' + str(rownos[i])
        tasklst.append(new_el)
        pairslst.append((matchfiles[i], rownos[i]))
    
    pairsset = set(pairslst)
    
    out_dct['task'] = tasklst
    
    out_df = pd.DataFrame(out_dct)
    
    agg_mv = MajorityVote().fit_predict(out_df)
    agg_wawa = Wawa().fit_predict(out_df)
    agg_ds = DawidSkene(n_iter=10).fit_predict(out_df)
    
    mv_dct = agg_mv.to_dict()
    wawa_dct = agg_wawa.to_dict()
    ds_dct = agg_ds.to_dict()
    
    res_schema = ['Match File', 'Row No', 'Vote', 'SANTOS Answer', 'Ground Truth']
    mv_res = {}
    wawa_res = {}
    ds_res = {}
    
    for rs in res_schema:
        mv_res[rs] = []
        wawa_res[rs] = []
        ds_res[rs] = []
    
    for pair in pairsset:
        mv_res['Match File'].append(pair[0])
        mv_res['Row No'].append(pair[1])
        wawa_res['Match File'].append(pair[0])
        wawa_res['Row No'].append(pair[1])
        ds_res['Match File'].append(pair[0])
        ds_res['Row No'].append(pair[1])
        
        task_ind = pair[0] + ':' + str(pair[1])
        mv_res['Vote'].append(mv_dct[task_ind])
        wawa_res['Vote'].append(wawa_dct[task_ind])
        ds_res['Vote'].append(ds_dct[task_ind])
        
        pair_df = df[(df['Match File'] == pair[0]) & (df['Row No'] == pair[1])]
        pair_gt = pair_df['Ground Truth'].unique().tolist()[0]
        mv_res['Ground Truth'].append(pair_gt)
        wawa_res['Ground Truth'].append(pair_gt)
        ds_res['Ground Truth'].append(pair_gt)
        
        mv_res['SANTOS Answer'].append(basedf.loc[pair[1]]['SANTOS Unionable'])
        wawa_res['SANTOS Answer'].append(basedf.loc[pair[1]]['SANTOS Unionable'])
        ds_res['SANTOS Answer'].append(basedf.loc[pair[1]]['SANTOS Unionable'])
    
    mv_df = pd.DataFrame(mv_res)
    wawa_df = pd.DataFrame(wawa_res)
    ds_df = pd.DataFrame(ds_res)
    
    mv_df.to_csv('MajorityVote_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    wawa_df.to_csv('Wawa_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    ds_df.to_csv('DawidSkene_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)

def perprompt_majorities(fullfname, temp):
    df = pd.read_csv(fullfname)
    tmpdf = df[df['Sampling Param'] == temp]
    out_schema = ['Match File', 'Row No', 'Prompt', 'Yes Votes', 'No Votes', 'Majority']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    for prompt in tmpdf['Story Name'].unique().tolist():
        promptdf = tmpdf[tmpdf['Story Name'] == prompt]
        for match_file in promptdf['Match File'].unique().tolist():
            for rowno in promptdf['Row No'].unique().tolist():
                cand_df = promptdf[(promptdf['Match File'] == match_file) & (promptdf['Row No'] == rowno)]
                vote_dct = cand_df['Story Answer'].value_counts().to_dict()
                if 1 in vote_dct:
                    yesvotes = vote_dct[1]
                else:
                    yesvotes = 0
                
                if 0 in vote_dct:
                    novotes = vote_dct[0]
                else:
                    novotes = 0
                othervotes = sum([vote_dct[k] for k in vote_dct if k != 0 and k != 1])
                out_dct['Match File'].append(match_file)
                out_dct['Row No'].append(rowno)
                out_dct['Prompt'].append(prompt)
                out_dct['Yes Votes'].append(yesvotes)
                out_dct['No Votes'].append(novotes + othervotes)
                if yesvotes > (novotes + othervotes):
                    out_dct['Majority'].append(True)
                else:
                    out_dct['Majority'].append(False)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv('per_promptresults_tmperature' + str(temp).replace('.', '_') + '.csv', index=False)

def perprompt_single(fullfname, temp):
    df = pd.read_csv(fullfname)
    tmpdf = df[df['Sampling Param'] == temp]
    out_schema = ['Match File', 'Row No', 'Prompt', 'Yes Votes', 'No Votes', 'Majority']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    for prompt in tmpdf['Story Name'].unique().tolist():
        promptdf = tmpdf[tmpdf['Story Name'] == prompt]
        for match_file in promptdf['Match File'].unique().tolist():
            for rowno in promptdf['Row No'].unique().tolist():
                cand_df = promptdf[(promptdf['Match File'] == match_file) & (promptdf['Row No'] == rowno) & (promptdf['Rep No'] == 0)]
                vote_dct = cand_df['Story Answer'].value_counts().to_dict()
                if 1 in vote_dct:
                    yesvotes = vote_dct[1]
                else:
                    yesvotes = 0
                
                if 0 in vote_dct:
                    novotes = vote_dct[0]
                else:
                    novotes = 0
                othervotes = sum([vote_dct[k] for k in vote_dct if k != 0 and k != 1])
                out_dct['Match File'].append(match_file)
                out_dct['Row No'].append(rowno)
                out_dct['Prompt'].append(prompt)
                out_dct['Yes Votes'].append(yesvotes)
                out_dct['No Votes'].append(novotes + othervotes)
                if yesvotes > (novotes + othervotes):
                    out_dct['Majority'].append(True)
                else:
                    out_dct['Majority'].append(False)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv('per_promptresults_tmperature' + str(temp).replace('.', '_') + '.csv', index=False)

def get_stats(method_names, temps, story_names):
    stat_schema = ['Method Name', 'Temperature', 'SANTOS Precision', 'SANTOS Recall',
                   'SANTOS f1', 'Crowd Precision', 'Crowd Recall', 'Crowd f1']
    
    for sn in story_names:
        sprec = sn + ' Precision'
        srec = sn + ' Recall'
        sf = sn + ' f1'
        stat_schema += [sprec, srec, sf]
    
    stats_dct = {}
    for o in stat_schema:
        stats_dct[o] = []
    
    for mn in method_names:
        for tmp in temps:
            vote_file = mn + '_results-temperature' + str(tmp).replace('.', '_') + '.csv'
            df = pd.read_csv(vote_file)
            df['Vote_bool'] = (df['Vote'] == 1)
            
            ditto_tps = df[(df['SANTOS Answer'] == df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            ditto_tns = df[(df['SANTOS Answer'] == df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
            ditto_fps = df[(df['SANTOS Answer'] != df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
            ditto_fns = df[(df['SANTOS Answer'] != df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            
            our_tps = df[(df['Vote_bool'] == df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            our_tns = df[(df['Vote_bool'] == df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
            our_fps = df[(df['Vote_bool'] != df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
            our_fns = df[(df['Vote_bool'] != df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            
            ditto_precision = ditto_tps / (ditto_tps + ditto_fps)
            ditto_recall = ditto_tps / (ditto_tps + ditto_fns)
            ditto_f1 = 2 * (ditto_precision * ditto_recall) / (ditto_precision + ditto_recall)
            
            our_precision = our_tps / (our_tps + our_fps)
            our_recall = our_tps / (our_tps + our_fns)
            our_f1 = 2 * (our_precision * our_recall) / (our_precision + our_recall)
            
            stats_dct['Method Name'].append(mn)
            stats_dct['Temperature'].append(tmp)
            stats_dct['SANTOS Precision'].append(ditto_precision)
            stats_dct['SANTOS Recall'].append(ditto_recall)
            stats_dct['SANTOS f1'].append(ditto_f1)
            stats_dct['Crowd Precision'].append(our_precision)
            stats_dct['Crowd Recall'].append(our_recall)
            stats_dct['Crowd f1'].append(our_f1)
            
            story_df = pd.read_csv('per_promptresults_tmperature' + str(tmp).replace('.', '_') + '.csv')
            for sn in story_names:
                sndf = story_df[story_df['Prompt'] == sn]
                ans_dct = {}
                for row in sndf.to_dict(orient='records'):
                    rowno = row['Row No']
                    snvote = row['Majority']
                    ans_dct[rowno] = snvote
                
                sn_tps = 0
                sn_tns = 0
                sn_fps = 0
                sn_fns = 0
                for rowno in ans_dct:
                    gt = df[df['Row No'] == rowno]['Ground Truth'].tolist()[0]
                    if gt == ans_dct[rowno] and gt == True:
                        sn_tps += 1
                    elif gt == ans_dct[rowno] and gt == False:
                        sn_tns += 1
                    elif gt != ans_dct[rowno] and gt == True:
                        sn_fns += 1
                    elif gt != ans_dct[rowno] and gt == False:
                        sn_fps += 1
                
                sn_precision = sn_tps / (sn_tps + sn_fps)
                sn_recall = sn_tps / (sn_tps + sn_fns)
                sn_f1 = 2 * (sn_precision * sn_recall) / (sn_precision + sn_recall)
                stats_dct[sn + ' Precision'].append(sn_precision)
                stats_dct[sn + ' Recall'].append(sn_recall)
                stats_dct[sn + ' f1'].append(sn_f1)
                
    
    stats_df = pd.DataFrame(stats_dct)
    stats_df.to_csv('santosvsmultiprompt_stats.csv', index=False)

#just return the crowd results
def crowd_only(fulldf, maindf, temp, itr_num=None):
    
    df = fulldf[fulldf['Sampling Param'] == temp]
    out_schema = ['worker', 'task', 'label']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    raw_labels = df['Story Answer'].tolist()
    new_labels = [max(x, 0) for x in raw_labels] #change -1s to 0s.
    
    out_dct['worker'] = df['Story Name'].tolist()
    out_dct['label'] = new_labels
    matchfiles = df['Match File'].tolist()
    rownos = df['Row No'].tolist()
    tasklst = []
    pairslst = []
    
    for i in range(len(matchfiles)):
        new_el = matchfiles[i] + ':' + str(rownos[i])
        tasklst.append(new_el)
        pairslst.append((matchfiles[i], rownos[i]))
    
    pairsset = set(pairslst)
    
    out_dct['task'] = tasklst
    
    out_df = pd.DataFrame(out_dct)
    
    agg_mv = MajorityVote().fit_predict(out_df)
    agg_wawa = Wawa().fit_predict(out_df)
    agg_ds = DawidSkene(n_iter=10).fit_predict(out_df)
    
    mv_dct = agg_mv.to_dict()
    wawa_dct = agg_wawa.to_dict()
    ds_dct = agg_ds.to_dict()
    
    res_schema = ['Match File', 'Row No', 'Vote', 'SANTOS Answer', 'Ground Truth']
    mv_res = {}
    wawa_res = {}
    ds_res = {}
    
    for rs in res_schema:
        mv_res[rs] = []
        wawa_res[rs] = []
        ds_res[rs] = []
    
    for pair in pairsset:
        mv_res['Match File'].append(pair[0])
        mv_res['Row No'].append(pair[1])
        wawa_res['Match File'].append(pair[0])
        wawa_res['Row No'].append(pair[1])
        ds_res['Match File'].append(pair[0])
        ds_res['Row No'].append(pair[1])
        
        task_ind = pair[0] + ':' + str(pair[1])
        mv_res['Vote'].append(mv_dct[task_ind])
        wawa_res['Vote'].append(wawa_dct[task_ind])
        ds_res['Vote'].append(ds_dct[task_ind])
        
        pair_df = df[(df['Match File'] == pair[0]) & (df['Row No'] == pair[1])]
        pair_gt = pair_df['Ground Truth'].unique().tolist()[0]
        mv_res['Ground Truth'].append(pair_gt)
        wawa_res['Ground Truth'].append(pair_gt)
        ds_res['Ground Truth'].append(pair_gt)
        
        pair_santos = maindf.loc[pair[1]]['SANTOS Unionable']
        mv_res['SANTOS Answer'].append(pair_santos)
        wawa_res['SANTOS Answer'].append(pair_santos)
        ds_res['SANTOS Answer'].append(pair_santos)
    
    mv_df = pd.DataFrame(mv_res)
    wawa_df = pd.DataFrame(wawa_res)
    ds_df = pd.DataFrame(ds_res)
    mv_name = 'MajorityVote_results' + '-temperature' + str(temp).replace('.', '_')
    if itr_num != None:
        mv_name += '-itr' + str(itr_num)
    mv_name += '.csv'
    
    wa_name = 'Wawa_results' + '-temperature' + str(temp).replace('.', '_')
    if itr_num != None:
        wa_name += '-itr' + str(itr_num)
    wa_name += '.csv'
    
    ds_name = 'DawidSkene_results' + '-temperature' + str(temp).replace('.', '_')
    if itr_num != None:
        ds_name += '-itr' + str(itr_num)
    ds_name += '.csv'
    
    mv_df.to_csv(mv_name, index=False)
    wawa_df.to_csv(wa_name, index=False)
    ds_df.to_csv(ds_name, index=False)
    return {'MajorityVote' : mv_df, 'Wawa' : wawa_df, 'DawidSkene' : ds_df}

def move_results(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    for f in os.listdir():
        if f.startswith('unionmatchwsuff') and f.endswith('.csv'):
            shutil.move(f, os.path.join(dirname, f))


if __name__=='__main__':
    # santos_candidates_tocsv('santos_small_gtcompare.csv')
    # santos_topkcand_tocsv('santos_small_gtcompare')
    # santos_performance('santos_small_gtcomparek10.csv')
    # inp_tblsample = ['new_york_city_restaurant_inspection_results_a.csv', 'practice_reference_a.csv',
    #                  'minister_meetings_a.csv', 'practice_reference_b.csv', 'contributors_parties_b.csv',
    #                  'contributors_parties_a.csv', 'tuition_assistance_program_tap_recipients_a.csv',
    #                  'workforce_management_information_b.csv', 'workforce_management_information_a.csv',
    #                  'deaths_2012_2018_a.csv', 'immigration_records_a.csv']
    #We will try a very small sample first, and see what improvement we get.
    # inp_tblsample = ['new_york_city_restaurant_inspection_results_a.csv']
    # inp_tblsample = ['minister_meetings_a.csv', 'practice_reference_a.csv', 'deaths_2012_2018_a.csv']
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots')
    parser.add_argument('--dataset', default='santos_small')
    args = parser.parse_args()
    few_shot = True
    if args.shots is None:
        few_shot = False
    
    if args.dataset == 'santos_small':
        candfile = 'santos_small_gtcomparek10.csv'
    else:
        candfile = 'tus_gtcomparek10.csv'
    inp_tblsample = pd.read_csv(candfile)['Input Table'].unique().tolist()
    #the below two tables were actually not part of the original benchmark, and should be excluded
    #from analysis
    exclude_tbls = ['albums.csv', 'new_york_city_restaurant_inspection_results.csv']
    
    all_inpstrs = get_allcandstrs(inp_tblsample, candfile)
    # stories = ['veryplain', 'accountant', 'cheat', 'mleng', 'theft', 'likelyyes', 'likelyno']
    stories = ['veryplain', 'accountant', 'cheat', 'mleng', 'theft']
    
    for s in stories:
        #we can rename the below, but important to note that the default is few-shot
        storysuff(inp_tblsample, candfile, s, [0.0], 'temperature', num_reps=1)
    
    outdir = 'unioncomplete_' + args.dataset
    move_results(outdir)
    combine_storiesresults(candfile, outdir)
    # crowd_gather('unionlarge_res_full.csv', candfile, 0.0)
    # crowd_gather('unionlarge_res_full.csv', candfile, 1.0)
    crowd_gather(outdir + '_full.csv', candfile, 0.0)
    
    # perprompt_single('unionlarge_res_full.csv', 0.0)
    # perprompt_single('unionlarge_res_full.csv', 1.0)
    perprompt_single(outdir + '_full.csv', 0.0)
    
    method_names = ['MajorityVote', 'Wawa', 'DawidSkene']
    temps = [0.0]
    get_stats(method_names, temps, stories)
    
    
    

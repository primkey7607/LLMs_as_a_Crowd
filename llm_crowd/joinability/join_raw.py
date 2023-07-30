import pandas as pd
import pickle
import openai
import signal
import argparse
import shutil
import os
from tablesonly_prompts import JoinRawPrompt
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
and figure out which configuration is best for joinability. We also have functions
to analyze the results and figure out which configuration will give
the best performance
'''
        

class MyTimeoutException(Exception):
    pass

#register a handler for the timeout
def handler(signum, frame):
    print("Waited long enough!")
    raise MyTimeoutException("STOP")

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def response_suffwtemp(prompt_tmp, tbl1, tbl2, col1, col2, temp_val, few_shot=None, timeout=30):
    followup = 'Begin your answer with YES or NO.'
    chat = [{"role": "system", "content": "You are a helpful assistant who can only answer YES or NO and then explain your reasoning."}]
    if few_shot != None:
        prompt = prompt_tmp.get_chat(tbl1, tbl2, few_shot, followup)
        chat += prompt
    else:
        fullprompt = prompt_tmp.get_prompt_nospaces(tbl1, tbl2)
        if prompt_tmp.lang != 'english':
            translator = Translator()
            followup = translator.translate(followup, src='english', dest=fullprompt.lang).text
        
        fullprompt = fullprompt + ' ' + followup
        
        
        fullmsg = {"role": "user", "content": fullprompt }
        chat.append(fullmsg)
    
    print("Sending: {}".format(chat))
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=chat,
        temperature=temp_val,
        max_tokens=20
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

def get_candidate(dfrow):
    tbl1name = dfrow['Input Table']
    tbl2name = dfrow['FK Table']
    # tbl1 = pd.read_csv('/home/cc/join_data/chembl_csvs/' + tbl1name, nrows=3).to_csv()
    # tbl2 = pd.read_csv('/home/cc/join_data/chembl_csvs/' + tbl2name, nrows=3).to_csv()
    tbl1 = chembl_strs[tbl1name]
    tbl2 = chembl_strs[tbl2name]
    col1 = dfrow['Input Column']
    col2 = dfrow['FK Column']
    
    return (tbl1, tbl2, col1, col2)

def storysuff(inp_tbls, match_file, story_name, samp_range : list, samp_type, num_reps=10):
    story_fname = 'all_prompts/' + story_name + '_joinrawprompt_english.pkl'
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    
    df = pd.read_csv(match_file)
    
    for tbl in inp_tbls:
        dfmatches = df[df['Input Table'] == tbl]
        dfdct = dfmatches.to_dict(orient='index')
        for r_ind in dfdct:
            dfrow = dfdct[r_ind]
            row_gt = dfrow['Ground Truth']
            match = get_candidate(dfrow)
            if match == None:
                continue
            for i in range(num_reps):
                for sval in samp_range:
                    outname = 'joinmatchwsuff' + story_name + '-' + str(r_ind) + '-' + 'rep' + str(i) + '-' + samp_type + str(sval).replace('.', '_') + '.csv'
                    if os.path.exists(outname):
                        continue
                    if samp_type == 'temperature':
                        story_response = response_suffwtemp(story_tmp, match[0], match[1], match[2], match[3], sval)
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

def construct_fewshot(row_num=None):
    synth_oswrong1 = {'Country' : ['Japan', 'China', 'India', 'Brazil'],
                    'City' : ['Tokyo', 'Shanghai', 'Delhi', 'Sao Paulo'], 
                    'Population' : [37435191, 26317104, 29399141, 21846507]}
    swost1 = pd.DataFrame(synth_oswrong1).to_csv(index=False)
    synth_oswrong2 = {'Serial No' : [53584747, 19673444, 29399141, 26317104],
                    'Product Name' : ['GoPro HDMI Cable Cable AHDMC-301 - Filmtools',
                                      'Mens Nixon The Crew Watch A1186-001 Men\'s Watches',
                                      'Nike Metcon 2 - Black/White/Wolf Grey/Volt  Nike Mens Shoes Regular Training Grey/Volt',
                                      'PowerLine HD Day Night Cloud Camera Kit DCS-6045LKT PowerLine Day/Night Kit'],
                    'Price' : [59.99, 354.59, 87.34, 55.45]}
    swost2 = pd.DataFrame(synth_oswrong2).to_csv(index=False)
    os_ex = (swost1, swost2, 'Population', 'Serial No', 'NO.')
    
    wrong1 = {'Employee ID' : [253, 125, 946],
              'Position' : ['Senior Manager', 'Sales Representative', 'Software Engineer'],
              'Salary' : [78123, 48423, 66135]}
    wrong2 = {'Date' : ['02-21-2021', '02-22-2021', '02-23-2021'], 
              'Time' : ['18:00', '22:00', '8:00'], 
              'Code' : ['More-than-usual meal ingestion',
                        'Pre-snack blood glucose measurement',
                        'NPH insulin dose']}
    swst1 = pd.DataFrame(wrong1).to_csv(index=False)
    swst2 = pd.DataFrame(wrong2).to_csv(index=False)
    wrong_ex = (swst1, swst2, 'Employee ID', 'Date', 'NO.')
    
    simtext1 = {'Median Age' : [42.0, 32.7, 31.7, 29.7],
                'Race' : ['White', 'Black', 'American Indian/Alaska Native', 'Hawaiian/Guamanian/Samoan' ],
                'Population' : [234370202, 40610815, 2632102, 570116]}
    simt1 = pd.DataFrame(simtext1).to_csv(index=False)
    simtext2 = {'Col_1' : ['Mainland Indigenous', 'Black', 'Pacific Islander', 'White'],
                'Col_2' : [44772, 41511, 61911, 65902]}
    simt2 = pd.DataFrame(simtext2).to_csv(index=False)
    simtext_ex = (simt1, simt2, 'Race', 'Col_1', 'YES.')
    
    simcol1 = {'Administrative Region' : ['Ohio', 'Alberta', 'Tottori', 'Durango'],
               'Area (sq km)' : [116096, 255541, 3510, 123364],
               'Population' : [11780000, 4371000, 570569, 1846000]}
    simcol2 = {'State' : ['Illinois', 'Nevada', 'Arkansas'],
               'Per Capita GDP' : [82125, 62656, 39107]}
    simc1 = pd.DataFrame(simcol1).to_csv(index=False)
    simc2 = pd.DataFrame(simcol2).to_csv(index=False)
    simcol_ex = (simc1, simc2, 'Administrative Region', 'State', 'YES.')
    
    badnum1 = {'Bus Company' : ['OC Transpo', 'Chicago Transit Authority', 'Greyhound'],
               'Revenue per Month' : [11800000, 280150000, 1800000000],
               'Year' : [2020, 2022, 2022] }
    badnum2 = {'Hospital' : ['Mayo Clinic', 'Johns Hopkins Hospital', 'University of Nebraska Medical Center'],
               'Annual Revenue (in millions)' : [15600, 2100, 639.5],
               'Year Founded' : [1864, 1889, 1869]}
    bnum1 = pd.DataFrame(badnum1).to_csv(index=False)
    bnum2 = pd.DataFrame(badnum2).to_csv(index=False)
    badnum_ex = (bnum1, bnum2, 'Year', 'Year Founded', 'NO.')
    
    goodnum1 = {'Name' : ['Komodo', 'Alinea', 'Buddakan', 'Top of the World'],
                'City' : ['Miami FL', 'Chicago IL', 'New York City NY', 'Las Vegas NV'],
                'Annual Revenue' : [41000000, 27072500, 21800593, 25672308],
                'Meals Served' : [285000, 41650, 177000, 218586]}
    goodnum2 = {'Location' : ['(25.761681, -80.191788)', '(41.881832, -87.623177)', '(36.188110, -115.176468)'],
                'Population' : [439890, 2697000, 646790],
                'GDP Per Capita' : [48140, 80398, 43584]}
    gnum1 = pd.DataFrame(goodnum1).to_csv(index=False)
    gnum2 = pd.DataFrame(goodnum2).to_csv(index=False)
    goodnum_ex = (gnum1, gnum2, 'City', 'Location', 'YES.')
    
    examples = [wrong_ex, simtext_ex, os_ex, simcol_ex, badnum_ex, goodnum_ex]
    return examples

def storysuff_fewshot(inp_tbls, match_file, story_name, samp_range : list, samp_type, examples, num_reps=10):
    story_fname = 'all_prompts/' + story_name + '_joinrawprompt_english.pkl'
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    
    df = pd.read_csv(match_file)
    
    for tbl in inp_tbls:
        dfmatches = df[df['Input Table'] == tbl]
        dfdct = dfmatches.to_dict(orient='index')
        for r_ind in dfdct:
            dfrow = dfdct[r_ind]
            row_gt = dfrow['Ground Truth']
            match = get_candidate(dfrow)
            if match == None:
                continue
            for i in range(num_reps):
                for sval in samp_range:
                    outname = 'joinmatchwsuff_fewshot' + story_name + '-' + str(r_ind) + '-' + 'rep' + str(i) + '-' + samp_type + str(sval).replace('.', '_') + '.csv'
                    if os.path.exists(outname):
                        continue
                    if samp_type == 'temperature':
                        story_response = response_suffwtemp(story_tmp, match[0], match[1], match[2], match[3], sval, few_shot=examples)
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

def storysuff_ptfirst(inp_tbls, match_file, story_name, samp_range : list, samp_type, num_reps=10):
    story_fname = 'all_prompts/' + story_name + '_joinprompt_english.pkl'
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    
    df = pd.read_csv(match_file)
    for i in range(num_reps):
        for sval in samp_range:
            for tbl in inp_tbls:
                dfmatches = df[df['Input Table'] == tbl]
                dfdct = dfmatches.to_dict(orient='index')
                for r_ind in dfdct:
                    dfrow = dfdct[r_ind]
                    row_gt = dfrow['Ground Truth']
                    match = get_candidate(dfrow)
                    if match == None:
                        continue
                    
                    outname = 'joinmatchwsuff' + story_name + '-' + str(r_ind) + '-' + 'rep' + str(i) + '-' + samp_type + str(sval).replace('.', '_') + '.csv'
                    if os.path.exists(outname):
                        continue
                    if samp_type == 'temperature':
                        story_response = response_suffwtemp(story_tmp, match[0], match[1], match[2], match[3], sval)
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

def storyfirst(inp_tbls, match_file, story_names, samp_range : list, samp_type, num_reps=10):
    story_tmps = []
    for story_name in story_names:
        story_fname = 'all_prompts/' + story_name + '_joinprompt_english.pkl'
        with open(story_fname, 'rb') as fh:
            story_temp = pickle.load(fh)
        story_tmps.append((story_name, story_temp))
    
    df = pd.read_csv(match_file)
    for i in range(num_reps):
        for sval in samp_range:
            for tbl in inp_tbls:
                dfmatches = df[df['Input Table'] == tbl]
                dfdct = dfmatches.to_dict(orient='index')
                for r_ind in dfdct:
                    dfrow = dfdct[r_ind]
                    row_gt = dfrow['Ground Truth']
                    match = get_candidate(dfrow)
                    if match == None:
                        continue
                    
                    for pair in story_tmps:
                        story_name = pair[0]
                        story_tmp = pair[1]
                        outname = 'joinmatchwsuff' + story_name + '-' + str(r_ind) + '-' + 'rep' + str(i) + '-' + samp_type + str(sval).replace('.', '_') + '.csv'
                        if os.path.exists(outname):
                            continue
                        if samp_type == 'temperature':
                            story_response = response_suffwtemp(story_tmp, match[0], match[1], match[2], match[3], sval)
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

def extract_storyprompt(story_name, tbl1, tbl2, col1, col2):
    story_fname = 'all_prompts/' + story_name + '_joinrawprompt_english.pkl'
    with open(story_fname, 'rb') as fh:
        story_tmp = pickle.load(fh)
    story_prompt = story_tmp.get_prompt_nospaces(tbl1, tbl2, col1, col2)
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
                match = get_candidate(dfrow)
                story_prompt = extract_storyprompt(row['Story Name'], match[0], match[1], match[2], match[3])
                out_dct['Story Prompt'].append(story_prompt)
                
                for c in df.columns:
                    if c in out_dct:
                        out_dct[c].append(row[c])
    
    outdf = pd.DataFrame(out_dct)
    outdf.to_csv(folder + '_full.csv', index=False)

def crowd_gather(raw_df, base_file, temp):
    basedf = pd.read_csv(base_file)
    df = raw_df[raw_df['Sampling Param'] == temp]
    out_schema = ['worker', 'task', 'label']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    raw_labels = df['Story Answer'].tolist()
    new_labels = [max(x, 0) for x in raw_labels] #change -1s to 0s. in this case, that makes sense.
    
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
    
    res_schema = ['Match File', 'Row No', 'Vote', 'Aurum Answer', 'Ground Truth']
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
        
        #the fact that this is even in Aurum's match file means it is true.
        mv_res['Aurum Answer'].append(True)
        wawa_res['Aurum Answer'].append(True)
        ds_res['Aurum Answer'].append(True)
    
    mv_df = pd.DataFrame(mv_res)
    wawa_df = pd.DataFrame(wawa_res)
    ds_df = pd.DataFrame(ds_res)
    
    mv_df.to_csv('MajorityVote_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    wawa_df.to_csv('Wawa_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)
    ds_df.to_csv('DawidSkene_results' + '-temperature' + str(temp).replace('.', '_') + '.csv', index=False)

def perprompt_majorities(df, temp):
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

def perprompt_single(df, temp):
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
    stat_schema = ['Method Name', 'Temperature', 'Aurum Precision', 'Crowd Precision']
    
    for sn in story_names:
        sprec = sn + ' Precision'
        stat_schema += [sprec]
    
    stats_dct = {}
    for o in stat_schema:
        stats_dct[o] = []
    
    for mn in method_names:
        for tmp in temps:
            vote_file = mn + '_results-temperature' + str(tmp).replace('.', '_') + '.csv'
            df = pd.read_csv(vote_file)
            df['Vote_bool'] = (df['Vote'] == 1)
            
            ditto_tps = df[(df['Aurum Answer'] == df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            ditto_tns = df[(df['Aurum Answer'] == df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
            ditto_fps = df[(df['Aurum Answer'] != df['Ground Truth']) & (df['Ground Truth'] == False)].shape[0]
            ditto_fns = df[(df['Aurum Answer'] != df['Ground Truth']) & (df['Ground Truth'] == True)].shape[0]
            
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
            stats_dct['Aurum Precision'].append(ditto_precision)
            # stats_dct['Aurum Recall'].append(ditto_recall)
            # stats_dct['Aurum f1'].append(ditto_f1)
            stats_dct['Crowd Precision'].append(our_precision)
            # stats_dct['Crowd Recall'].append(our_recall)
            # stats_dct['Crowd f1'].append(our_f1)
            
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
                # stats_dct[sn + ' Recall'].append(sn_recall)
                # stats_dct[sn + ' f1'].append(sn_f1)
                
    
    stats_df = pd.DataFrame(stats_dct)
    stats_df.to_csv('aurumvsmultiprompt_stats.csv', index=False)

def analyze(fullfname, base_file, temps, story_names, method_names, min_reps=None, rep_cutoff=None):
    result_file = fullfname
    df = pd.read_csv(result_file)
    df = df[~df['Sampling Param'].isna()]
    df['Rep No'] = df['Rep No'].astype(float).astype(int)
    df['Row No'] = df['Row No'].astype(float).astype(int)
    if rep_cutoff is not None:
        df = df[df['Rep No'] < rep_cutoff]
    elif min_reps is not None:
        df = df[df['Rep No'] >= min_reps]
    for temp in temps:
        print(f"Temp {temp}:")
        crowd_gather(df, base_file, temp)
        perprompt_single(df, temp)
    
    get_stats(method_names, temps, story_names)

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
    
    res_schema = ['Match File', 'Row No', 'Vote', 'Aurum Answer', 'Ground Truth']
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
        
        mv_res['Aurum Answer'].append(True)
        wawa_res['Aurum Answer'].append(True)
        ds_res['Aurum Answer'].append(True)
    
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
        if f.startswith('joinmatchwsuff') and f.endswith('.csv'):
            shutil.move(f, os.path.join(dirname, f))

if __name__=='__main__':
    chembl_strs = {}
    for f in os.listdir('../chembl_csvs'):
        fpath = os.path.join('../chembl_csvs', f)
        if fpath.endswith('.csv'):
            tbl_str = pd.read_csv(fpath, nrows=2).to_csv()
            chembl_strs[f] = tbl_str
    print(chembl_strs.keys())
    # inp_tblsample = ['assays.csv']
    # inp_tbls = os.listdir('../join_data/chembl_csvs')
    # inp_tbls = ['assays.csv', 'formulations.csv', 'drug_mechanism.csv', 'activities.csv', 'compound_properties.csv']
    # inp_tbls = ['assays.csv', 'drug_mechanism.csv']
    candfile = 'chembl_half_half.csv'
    inp_tbls = pd.read_csv(candfile)['Input Table'].unique().tolist()
    stories = ['plain', 'accountant', 'mleng', 'lost', 'fk']
    join_examples = construct_fewshot()
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots')
    args = parser.parse_args()
    few_shot = True
    if args.shots is None:
        few_shot = False
    for s in stories:
        if few_shot:
            storysuff_fewshot(inp_tbls, candfile, s, [0.0], 'temperature', join_examples, num_reps=1)
        else:
            storysuff(inp_tbls, candfile, s, [0.0], 'temperature', join_examples, num_reps=1)
    # storyfirst(inp_tbls, candfile, stories, [2.0], 'temperature')
    move_results('joinraw_complete')
    combine_storiesresults(candfile, 'joinraw_complete')
    
    # crowd_gather('joincomplete_full.csv', candfile, 0.0)
    # crowd_gather('joincomplete_full.csv', candfile, 1.0)
    # crowd_gather('joincomplete_full.csv', candfile, 2.0)
    
    # perprompt_majorities('chemblresults_sample.csv', 0.0)
    # perprompt_majorities('chemblresults_sample.csv', 1.0)
    # perprompt_majorities('joincomplete_full.csv', 2.0)
    
    method_names = ['MajorityVote', 'Wawa', 'DawidSkene']
    # temps = [0.0, 1.0, 2.0]
    
    # # get_stats(method_names, temps, stories)
    analyze('joinraw_complete_full.csv', candfile, [0.0], stories, method_names)
    

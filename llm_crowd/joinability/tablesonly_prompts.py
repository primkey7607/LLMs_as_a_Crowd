import pickle

class JoinRawPrompt:
    def __init__(self, preamble, c1sentence, c2sentence, question):
        self.preamble = preamble
        self.c1sentence = c1sentence
        self.c2sentence = c2sentence
        self.question = question
        self.lang = 'english'
    
    def get_prompt_nospaces(self, candidate1, candidate2, include_preamble=True):
        full_st = self.c1sentence + candidate1 + self.c2sentence + candidate2 + self.question
        if include_preamble:
            full_st = self.preamble + full_st
        return full_st
    
    def get_chat(self, candidate1, candidate2, examples, followup):
        chat = []
        include_preamble = True
        for example in examples:
            message = self.get_prompt_nospaces(example[0], example[1], include_preamble=include_preamble) + ' ' + followup
            chat.append({"role": "user", "content": message})
            include_preamble = False
            chat.append({"role": "assistant", "content": example[4]})
        message = self.get_prompt_nospaces(candidate1, candidate2, include_preamble=include_preamble) + ' ' + followup
        chat.append({"role": "user", "content": message})
        return chat

def gen_plain():
    preamble = 'Consider samples of the following two tables:\n\n'
    c1sentence = 'Table 1:\n'
    c2sentence = '\n\nTable 2:\n'
    question = '\n\n Is there a Table 1 column and Table 2 column that capture the same information? '
    layprompt = JoinRawPrompt(preamble, c1sentence, c2sentence, question)
    return layprompt

def gen_accountant():
    preamble = 'I\'m an accountant trying to understand whether I can combine a few spreadsheets. Consider samples of the following two spreadsheets:\n\n'
    c1sentence = 'Spreadsheet 1:\n'
    c2sentence = '\n\nSpreadsheet 2:\n'
    question = '\n\n Based on these spreadsheet samples, is there a Spreadsheet 1 column and a Spreadsheet 2 column that capture the same information?'
    
    layprompt = JoinRawPrompt(preamble, c1sentence, c2sentence, question)
    return layprompt

def gen_mleng():
    preamble = 'I\'m a machine learning engineer looking for training data in our company\'s data lake.\n\n'
    c1sentence = 'I have the following training dataset so far:\n\n'
    c2sentence = '\n\nAnd I\'m looking for datasets containing new features I can add to this. So far, I found the following dataset:\n\n'
    question = '\n\nDoes it make sense to add columns from this dataset to my current training dataset by joining the two datasets on some column '
    question += 'from my training dataset, and some column '
    question += 'from the dataset I found?'
    csprompt = JoinRawPrompt(preamble, c1sentence, c2sentence, question)
    return csprompt

def gen_lost():
    preamble = 'I\'m a DB Admin who recently recovered a table that was lost. '
    c1sentence = 'Here is the table I recovered (call it Table 1):\n'
    c2sentence = '\n\nTo verify that I recovered the right table, I know this table '
    c2sentence += 'is supposed to have a foreign key dependency with the following table (call it Table 2):\n'
    question = '\n\nIs it likely that some column '
    question += 'from Table 1 and some column '
    question += 'from Table 2 capture the same type of information?'
    layprompt = JoinRawPrompt(preamble, c1sentence, c2sentence, question)
    return layprompt

def gen_fk():
    preamble = 'I\'m a data engineer who found two tables in my company\'s data lake, and is attempting to ingest them into the company\'s DBMS.\n'
    c1sentence = 'Sample of Table 1:\n'
    c2sentence = '\n\nSample of Table 2:\n'
    question = '\n\nBased on the given table samples, would it make sense to declare a foreign key dependency between Table 1 and Table 2 on some column '
    question += 'from Table 1 and some column '
    question += 'from Table 2?'
    layprompt = JoinRawPrompt(preamble, c1sentence, c2sentence, question)
    return layprompt

if __name__=='__main__':
    plain_tmp = gen_plain()
    acc_tmp = gen_accountant()
    mleng_tmp = gen_mleng()
    lost_tmp = gen_lost()
    fk_tmp = gen_fk()
    lst = [plain_tmp, acc_tmp, mleng_tmp, lost_tmp, fk_tmp]
    names = ['plain', 'accountant', 'mleng', 'lost', 'fk']
    for i,l in enumerate(lst):
        with open('all_prompts/' + names[i] + '_joinrawprompt_english.pkl', 'wb') as fh:
            pickle.dump(l, fh)
    
    cand1 = 'tid,target_type,pref_name,tax_id,organism,chembl_id,species_group_flag'
    cand1 += '\n1,SINGLE PROTEIN,Maltase-glucoamylase,9606.0,Homo sapiens,CHEMBL2074,0'
    cand1 += '\n2,SINGLE PROTEIN,Sulfonylurea receptor 2,9606.0,Homo sapiens,CHEMBL1971,0'
    
    cand2 = 'doc_id,journal,year,volume,issue,first_page,last_page,pubmed_id,doi,chembl_id,title,doc_type,authors,abstract,patent_id,ridx,src_id'
    cand2 += '\n-1,,,,,,,,,CHEMBL1158643,Unpublished dataset,DATASET,,,,CLD0,0'
    cand2 += '\n1,J Med Chem,2004.0,47,1,1,9,14695813.0,10.1021/jm030283g,CHEMBL1139451,The discovery of ezetimibe: a view from outside the receptor.,PUBLICATION,Clader JW.,,,CLD0,1'
    
    col1 = 'tid'
    col2 = 'doc_id'
    
    print(plain_tmp.get_prompt_nospaces(cand1, cand2))
    print(acc_tmp.get_prompt_nospaces(cand1, cand2))
    print(mleng_tmp.get_prompt_nospaces(cand1, cand2))
    print(lost_tmp.get_prompt_nospaces(cand1, cand2))
    print(fk_tmp.get_prompt_nospaces(cand1, cand2))
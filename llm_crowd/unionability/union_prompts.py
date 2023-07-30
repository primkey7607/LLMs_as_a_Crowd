from prompt_generatorv2 import IntegrationPrompt
import pickle

'''
Purpose: This file contains the union prompt generation functions.
'''

def gen_accountant():
    preamble = 'I\'m an accountant trying to understand a few spreadsheets.'
    c1sentence = ' One looks like this:\n\n'
    c2sentence = '\n\nAnd the other looks like this:\n\n'
    question = '\n\nIs it possible that these spreadsheets are describing the same type of information?'
    layprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return layprompt

def gen_theft():
    preamble = 'It is common for data thieves to hide their theft by creating a similar dataset instead of using the original one outright by creating a relational table with a similar schema, but different data instances.'
    c1sentence = 'A company that was recently hacked tells you they have reason to believe the following relational table was transferred during the hack:\n\n'
    c2sentence = '\n\nDuring your investigation, you find someone in possession of a relational table:\n\n'
    # question = 'Would you charge the storefront owner for selling an illegal copy of the product?'
    question = '\n\nIs it likely that this is a similar dataset?'
    detprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return detprompt

def gen_mleng():
    preamble = 'I\'m a machine learning engineer looking for training data in our company\'s data lake.'
    c1sentence = 'I have the following training dataset so far:\n\n'
    c2sentence = '\n\nAnd I\'m looking for data containing rows I can add to this. So far, I found the following dataset:\n\n'
    question = '\n\nModulo some data transformation, is it possible to add rows from this dataset to my current training dataset?'
    csprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return csprompt

def gen_veryplain():
    preamble = ''
    c1sentence = ''
    c2sentence = '\n\n###\n\n'
    question = '\n\n###\n\nAre these relations unionable?'
    csprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return csprompt

def gen_cheat():
    preamble = 'Suppose you are a database design professor. '
    preamble += 'You have tasked your students with designing database schemas for their own web applications, and providing some plausible example rows. '
    preamble += 'Unfortunately, it is common for students to cheat by copying each others\' schemas and '
    preamble += 'modifying the column names slightly, and simply using different example rows.'
    c1sentence = 'While grading these assignments, you notice the following relations that two students provided:\n\n'
    c2sentence = '\n\n###\n\n'
    question = '\n\n###\n\nBased on this information, is it possible that one student copied the other\'s table?'
    csprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return csprompt

def gen_likelyyes():
    preamble = ''
    c1sentence = ''
    c2sentence = '\n\n###\n\n'
    question = '\n\n###\n\nAre these relations unionable? You are inclined to say \'Yes\'.'
    csprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return csprompt

def gen_likelyno():
    preamble = ''
    c1sentence = ''
    c2sentence = '\n\n###\n\n'
    question = '\n\n###\n\nAre these relations unionable? You are inclined to say \'No\'.'
    csprompt = IntegrationPrompt(preamble, c1sentence, c2sentence, question)
    return csprompt

if __name__=='__main__':
    accprompt = gen_accountant()
    theftprompt = gen_theft()
    mlprompt = gen_mleng()
    vp_prompt = gen_veryplain()
    cheat_prompt = gen_cheat()
    yes_prompt = gen_likelyyes()
    no_prompt = gen_likelyno()
    
    lst = [accprompt, theftprompt, mlprompt, vp_prompt, cheat_prompt, yes_prompt, no_prompt]
    names = ['accountant', 'theft', 'mleng', 'veryplain', 'cheat', 'likelyyes', 'likelyno']
    for i,l in enumerate(lst):
        with open('all_prompts/' + names[i] + '_unionprompt_english.pkl', 'wb') as fh:
            pickle.dump(l, fh)
    
    example_totry1 = 'row1,row2\n1,USA'
    example_totry2 = 'row1,row2\n3,Canada'
    
    print(accprompt.get_prompt_nospaces(example_totry1, example_totry2))
    print(theftprompt.get_prompt_nospaces(example_totry1, example_totry2))
    print(mlprompt.get_prompt_nospaces(example_totry1, example_totry2))
    print(vp_prompt.get_prompt_nospaces(example_totry1, example_totry2))
    print(cheat_prompt.get_prompt_nospaces(example_totry1, example_totry2))
    print(yes_prompt.get_prompt_nospaces(example_totry1, example_totry2))
    print(no_prompt.get_prompt_nospaces(example_totry1, example_totry2))




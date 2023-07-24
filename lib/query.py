import pandas as pd
import openai
import signal
import argparse
from dotenv import load_dotenv
import os
from pathlib import Path
from googletrans import Translator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff
from prompt_generator import TEMPLATES

STORIES = ['baseline', 'plain', 'veryplain', 'customer', 'journalist', 'security', 'layperson', 'detective']
DATASETS = ['cameras', 'computers', 'shoes', 'watches']
ALL_DATASETS = DATASETS + ['Amazon-Google', 'wdc_seen', 'wdc_half', 'wdc_unseen', 'wdc']

'''
Purpose: Run multiple prompts with multiple temperatures,
and figure out which configuration is best for entity resolution. We also have functions
to analyze the results and figure out which configuration will give
the best performance.
    
'''

class MyTimeoutException(Exception):
    pass

#register a handler for the timeout
def handler(signum, frame):
    print("Waited long enough!")
    raise MyTimeoutException("STOP")

def write_responses(
        out_dir: Path,
        datasets: Dict[str, pd.DataFrame],
        prompt_templates: List[PromptTemplate],
        examples_func: Callable[[str, int], List[Tuple[Any, Any, Any]],
        num_reps: int = 1):
    '''
    Write a csv file in out_dir for each (prompt template, df, row in df, rep) combination. We write
    separate files so experiments can be parallelized and so errors don't make us lose progress.
    '''
    os.makedirs(out_dir, exist_ok=True)

    for dataset_name, df in datasets.items():
        for idx, row in df.iterrows():
            examples = examples_func(dataset_name, idx)

            for template in prompt_templates:
                for rep in range(num_reps):
                    fname = out_dir / f'{dataset_name}-{template}-{idx}-{rep}.csv'
                    if fname.exists():
                        print(f"Already exists: {fname} exists...")
                        continue
                    print(f"Querying: {fname}")
                    response, answer = template.get_response(row['left'], row['right'], examples)

                    out_df = pd.DataFrame({
                        'template': [template.name],
                        'dataset': [dataset_name],
                        'row': [idx],
                        'rep': [rep],
                        'response': [response],
                        'answer': [answer],
                        'truth': row['label']})
                    out_df.to_csv(fname, index=False)

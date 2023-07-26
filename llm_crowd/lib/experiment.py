import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import openai
import pandas as pd

from llm_crowd.lib.prompt_template import PromptTemplate

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
        examples_func: Callable[[str, int, int], List[Tuple[Any, Any, Any]]],
        num_reps: int = 1):
    '''
    Write a csv file in out_dir for each (prompt template, df, row in df, rep) combination. We write
    separate files so experiments can be parallelized and so errors don't make us lose progress.
    '''
    os.makedirs(out_dir, exist_ok=True)

    for rep in range(num_reps):
        for dataset_name, df in datasets.items():
            for idx, row in df.iterrows():
                examples = examples_func(dataset_name, idx, rep)

                for template in prompt_templates:
                    fname = out_dir / f'{dataset_name}-{template.name}-{idx}-{rep}.csv'
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

def combine_responses(in_dir: Path, out_file: Path):
    '''
    Combine csv files from write_responses into a single file
    '''
    small_dfs = []
    big_dfs = []
    counter = 0
    print("Fixing newlines...")
    os.system(f"bash ./fix_newline.sh {in_dir}")
    for f in Path(in_dir).glob(f'*.csv'):
        df = pd.read_csv(f)
        small_dfs.append(df)
        if len(df) > 1:
            raise ValueError(f"Invalid file: {f}")
        counter += 1
        if counter % 1000 == 0:
            print(f'{counter}...')
            big_dfs.append(pd.concat(small_dfs))
            small_dfs = []
    print("Concatting...")
    df = pd.concat(big_dfs + small_dfs)
    print("Writing...")
    df.to_csv(out_file, index=False)

    

import argparse

import pandas as pd

ATTRIBUTES = ['brand', 'title', 'description', 'price', 'priceCurrency']

def convert_df(df):
    df['left'] = df.apply(left_col, axis=1)
    df['right'] = df.apply(right_col, axis=1)
    df['label'] = df['label'].astype(str)
    return df[['left', 'right', 'label']]

def left_col(row):
    return side_col('left', row)

def right_col(row):
    return side_col('right', row)

def side_col(side, row):
    out_example = []
    for attribute in ATTRIBUTES:
        col = f'{attribute}_{side}'
        if row[col] is None:
            continue
        out_example.append(f"{attribute}: {row[col]}")
    return '\n'.join(out_example)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outfile")
    args = parser.parse_args()

    df = pd.read_json(args.infile, compression='gzip', lines=True)
    df = convert_df(df)
    df.to_csv(args.outfile, index=False)

import argparse
import re

import pandas as pd

LANGUAGE_TAG = r'"@[a-z][a-z](-[a-zA-Z]*)?'
COLVAL = r'COL ([a-zA-Z]+) VAL +'

def convert_lines(lines):
    lefts = []
    rights = []
    labels = []
    for line in lines:
        line = line.strip()
        line = re.sub(LANGUAGE_TAG, '', line)
        line = line.replace('"', '')
        line = re.sub(COLVAL, r'\n\1: ', line)
        left, right, label = line.split('\t')

        left = left.strip()
        left = left.split('\n')
        left = [l[:1000] for l in left]
        left = '\n'.join(left)

        right = right.strip()
        right = right.split('\n')
        right = [r[:1000] for r in right]
        right = '\n'.join(right)

        lefts.append(left)
        rights.append(right)
        labels.append(label)
    return pd.DataFrame({'left': lefts, 'right': rights, 'label': labels})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outfile")
    args = parser.parse_args()

    with open(args.infile) as f:
        lines = f.readlines()
    df = convert_lines(lines)
    df.to_csv(args.outfile, index=False)

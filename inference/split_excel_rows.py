import pandas as pd
import numpy as np
import sys

def split(file, N):
    
    if file.endswith('.jsonl'):
        df = pd.read_json(file, lines=True)
    elif file.endswith('.xlsx'):
        df = pd.read_excel(file)
    dfs = np.array_split(df, int(N))

    base_name = file.split('/')[-1]
    for idx, subdf in enumerate(dfs):
        subdf.to_excel(f'tmp/{base_name.split(".")[0]}_sub{idx}.xlsx', index=False)

if __name__ == '__main__':
    file=sys.argv[1]
    N = sys.argv[2]
    split(file, N)
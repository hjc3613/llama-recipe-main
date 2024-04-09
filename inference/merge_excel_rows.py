import pandas as pd
import os
import sys

def merge(dir, output, pattern=None):
    result = []
    files = sorted(os.listdir(dir))
    for file in files:
        if pattern and pattern not in file:
            continue
        print('to merge file: ', os.path.join(dir, file))
        df = pd.read_excel(os.path.join(dir, file))
        result.append(df)
    result: pd.DataFrame = pd.concat(result, ignore_index=True)
    result.to_excel(output, index=False)
    print(f'merged file save to {output}')

if __name__ == '__main__':
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
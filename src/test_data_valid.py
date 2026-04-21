import pandas as pd
import glob

files = glob.glob('data/raw/*.csv')
for f in files:
    df = pd.read_csv(f)
    falls = df['event_flag'].sum()
    print(f'{f.split(chr(92))[-1]}: {len(df)} rows, {falls} fall flags')
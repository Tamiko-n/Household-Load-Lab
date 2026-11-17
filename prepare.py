from pathlib import Path
import argparse
import numpy as np
import pandas as pd

APPLIANCES = {"air1": "Air conditioner", "refrigerator1": "Refrigerator", "dishwasher1": "Dishwasher"}
REGIONS = {"Austin": "1minute_data_austin.csv", "California": "1minute_data_california.csv",
           "New York": "1minute_data_newyork.csv", "Puerto Rico": "pr_realpower_*.csv"}
NEURAL = ['MLP', '1D CNN', 'GRU', 'LSTM']
WINDOW = 30
SEED = 42

def prepare(root, output):
    for region, pattern in REGIONS.items():
        paths = sorted(p for p in root.rglob(pattern) if p.is_file())
        if not paths:
            raise FileNotFoundError(f"No {region} files found under {root}")
        parts = []
        for path in paths:
            print(path.name, flush=True)
            source = ['solar', 'solar2', 'battery1']
            cols = ['dataid', 'localminute', 'grid', *source, *APPLIANCES]
            for chunk in pd.read_csv(path, usecols=cols, chunksize=250_000):
                # Deterministic sampling of whole calendar days to preserve short sequences.
                date = chunk.localminute.str.slice(0, 10)
                keep = pd.util.hash_pandas_object(date, index=False).to_numpy() % 10 == 0
                chunk = chunk.loc[keep].copy()
                chunk['aggregate'] = chunk.grid + chunk[source].fillna(0).sum(axis=1)
                chunk = chunk.drop(columns=['grid', *source])
                parts.append(chunk.dropna(subset=['dataid', 'localminute', 'aggregate']))
        data = pd.concat(parts, ignore_index=True)
        data.localminute = pd.to_datetime(data.localminute, utc=True, errors='coerce', format='mixed')
        data = data.dropna(subset=['localminute']).sort_values(['dataid', 'localminute'])
        data = data.drop_duplicates(['dataid', 'localminute'])
        frames = []
        for home, rows in data.groupby('dataid'):
            times = rows.localminute.to_numpy(dtype='datetime64[m]').astype('int64')
            power = rows['aggregate'].to_numpy(dtype='float32')
            targets = rows[list(APPLIANCES)].to_numpy(dtype='float32')
            if len(rows) < WINDOW:
                continue
            spans = np.lib.stride_tricks.sliding_window_view(times, WINDOW)
            ends = np.arange(WINDOW - 1, len(rows))
            valid = (spans[:, -1] - spans[:, 0] == WINDOW - 1) & (times[ends] % 5 == 0)
            valid &= np.isfinite(targets[ends]).any(axis=1)
            if not valid.any():
                continue
            frames.append((np.lib.stride_tricks.sliding_window_view(power, WINDOW)[valid],
                           targets[ends[valid]].clip(0), times[ends[valid]],
                           np.full(valid.sum(), str(home))))
        x, y, t, h = [np.concatenate(values) for values in zip(*frames)]
        np.savez_compressed(output / f'{region}.npz', x=x, y=y, time=t, home=h)
        print(region, len(x), len(set(h)), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('E:/ann'))
    parser.add_argument('--cache', type=Path, default=Path('cache'))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    prepare(args.data, args.cache)


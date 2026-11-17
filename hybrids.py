"""Train every two-, three-, and four-encoder feature fusion on the pooled split."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from prepare import APPLIANCES, NEURAL, SEED
from train import Network, cutoff, features, load, metrics, sample, splits

COMBINATIONS = {f"Hybrid: {' + '.join(names)}": names
                for size in range(2, len(NEURAL) + 1) for names in combinations(NEURAL, size)}

def embeddings(config, x, names):
    blocks = {}
    with torch.inference_mode():
        data = torch.from_numpy(x)
        for name in names:
            encoder = Network(name)
            encoder.load_state_dict(config['models'][name][0])
            encoder.eval()
            blocks[name] = np.concatenate([encoder.encode(batch).numpy() for batch in data.split(1024)])
    return blocks

def head(size):
    return nn.Sequential(nn.Linear(size, 48), nn.ReLU(), nn.Dropout(.1), nn.Linear(48, 2))

def fuse(blocks, names):
    return np.concatenate([blocks[name] for name in names], axis=1)

def predict_head(model, x, power_scale):
    with torch.no_grad():
        out = torch.cat([model(batch) for batch in torch.from_numpy(x).split(2048)])
        return out[:, 0].sigmoid().numpy(), nn.functional.softplus(out[:, 1]).numpy() * power_scale

def infer(config, x, name, state):
    names = COMBINATIONS[name]
    model = head(32 * len(names))
    model.load_state_dict(state['weights'])
    model.eval()
    return predict_head(model, fuse(embeddings(config, x, names), names), state['scale'])

def fit(x, xv, p, pv, y, yv, epochs, rng):
    torch.manual_seed(SEED)
    model = head(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(15, (1-y.mean())/y.mean()), dtype=torch.float32))
    scale = max(.1, float(np.quantile(p[y], .9)))
    def loss(data, power, state):
        output = model(torch.from_numpy(data))
        return bce(output[:, 0], torch.from_numpy(state.astype('float32'))) + nn.functional.smooth_l1_loss(
            nn.functional.softplus(output[:, 1]), torch.from_numpy(power / scale))
    best, history = float('inf'), []
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(x))
        for ids in np.array_split(order, max(1, int(np.ceil(len(x)/512)))):
            optimizer.zero_grad()
            loss(x[ids], p[ids], y[ids]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(loss(xv, pv, yv))
        history.append(value)
        if value < best:
            best = value
            weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(weights)
    return model, weights, scale, history

def run(cache, output, epochs, cap):
    data = load(cache)
    home, region = data['home'], data['region']
    folds = list(splits(data))
    previous = json.loads((output/'splits.json').read_text(encoding='utf-8'))
    manifest = []
    for protocol, train, validation, test in folds:
        ids = {name: sorted(set(home[mask])) for name, mask in
               [('train', train), ('validation', validation), ('test', test)]}
        prior = next(f for f in previous if f['protocol'] == protocol)
        assert set(ids['test']) == set(prior['test'])
        assert set(ids['train']) | set(ids['validation']) == set(prior['development'])
        assert not set(ids['test']).intersection(ids['train'] + ids['validation'])
        manifest.append(dict(protocol=protocol, development=prior['development'], **ids))
    (output/'splits.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    protocol, train, validation, test = folds[0]
    base = torch.load(output/'models.pt', map_location='cpu', weights_only=False)
    rows, traces, logs, bundle = [], [], [], {}
    rng = np.random.default_rng(SEED)
    for j, circuit in enumerate(APPLIANCES):
        cfg = base[circuit]
        power = data['y'][:, j]
        valid = np.isfinite(power) & np.isfinite(data['x']).all(axis=1)
        it = sample(train & valid, home, cap, rng)
        iv = sample(validation & valid, home, max(250, cap//3), rng)
        ie = np.flatnonzero(test & valid)
        truth = power > cfg['threshold']
        print(circuit, len(it), len(iv), len(ie), flush=True)
        encoded = [embeddings(cfg, features(data['x'][ids], cfg['center'], cfg['scale']), NEURAL)
                   for ids in (it, iv, ie)]
        bundle[circuit] = {}
        for name, names in COMBINATIONS.items():
            x, xv, xe = [fuse(blocks, names) for blocks in encoded]
            model, weights, scale, history = fit(x, xv, power[it], power[iv], truth[it], truth[iv], epochs, rng)
            sv, _ = predict_head(model, xv, scale)
            ct = cutoff(truth[iv], sv)
            se, estimates = predict_head(model, xe, scale)
            bundle[circuit][name] = dict(weights=weights, scale=scale, cutoff=ct)
            identity = dict(protocol=protocol, appliance=circuit, model=name)
            logs.extend(dict(**identity, epoch=k+1, validation_loss=v) for k,v in enumerate(history))
            for reg in np.unique(region[ie]):
                in_region = region[ie] == reg
                for household in ['All homes', *np.unique(home[ie[in_region]])]:
                    pick = in_region if household == 'All homes' else in_region & (home[ie] == household)
                    ids = ie[pick]
                    result = metrics(truth[ids], se[pick] >= ct, power[ids], estimates[pick])
                    group = dict(**identity, region=reg, home=household)
                    rows.append(dict(**group, label_threshold_w=cfg['threshold']*1000, score_threshold=ct,
                                     train_on=int(truth[it].sum()), train_off=int((~truth[it]).sum()),
                                     validation_on=int(truth[iv].sum()), validation_off=int((~truth[iv]).sum()),
                                     **result))
                    if household != 'All homes':
                        days, inv = np.unique(data['time'][ids]//1440, return_inverse=True)
                        chosen = np.bincount(inv, weights=power[ids]).argmax()
                        plot = inv == chosen
                        traces.append(pd.DataFrame(dict(**group, time=data['time'][ids][plot],
                            actual_w=power[ids][plot]*1000, predicted_w=estimates[pick][plot]*1000)))
            print(name, flush=True)
    torch.save(bundle, output/'hybrids.pt')
    pd.DataFrame(rows).to_csv(output/'hybrid_metrics.csv', index=False)
    pd.concat(traces, ignore_index=True).to_csv(output/'hybrid_traces.csv.gz', index=False)
    pd.DataFrame(logs).to_csv(output/'hybrid_training_log.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path('cache'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--cap', type=int, default=1100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.cache, args.output, args.epochs, args.cap)

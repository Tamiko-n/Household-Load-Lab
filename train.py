import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from prepare import APPLIANCES, REGIONS, WINDOW, SEED, NEURAL
MODELS = ['Always off', 'Linear', 'Gradient boosting', *NEURAL, 'Neural ensemble']

class Network(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == 'MLP':
            self.encoder = nn.Sequential(nn.Linear(2 * WINDOW, 64), nn.ReLU(),
                                         nn.Dropout(.1), nn.Linear(64, 32), nn.ReLU())
        elif name == '1D CNN':
            self.encoder = nn.Sequential(nn.Conv1d(2, 24, 5, padding=2), nn.ReLU(),
                                         nn.Conv1d(24, 32, 5, padding=2), nn.ReLU())
        else:
            self.encoder = {'GRU': nn.GRU, 'LSTM': nn.LSTM}[name](2, 32, batch_first=True)
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        if self.name == 'MLP':
            encoded = self.encoder(x.flatten(1))
        elif self.name == '1D CNN':
            encoded = self.encoder(x.transpose(1, 2))[:, :, -1]
        else:
            encoded = self.encoder(x)[0][:, -1]
        out = self.head(encoded)
        return out[:, 0], nn.functional.softplus(out[:, 1])

def features(x, center, scale):
    delta = np.diff(x, axis=1, prepend=x[:, :1])
    return np.stack(((x - center) / scale, delta / scale), axis=-1).clip(-25, 25)

def metrics(truth, above, watts, estimate):
    tn, fp, fn, tp = confusion_matrix(truth, above, labels=[False, True]).ravel()
    error = (watts - estimate) * 1000
    return dict(accuracy=(tp + tn) / len(truth),
                balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) and (tn + fp) else np.nan,
                macro_f1=(2 * tp / max(1, 2 * tp + fp + fn) + 2 * tn / max(1, 2 * tn + fp + fn)) / 2,
                mae_w=np.abs(error).mean(), rmse_w=np.sqrt(np.square(error).mean()),
                on=int(tp+fn), off=int(tn+fp), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def cutoff(truth, score):
    def rank(t):
        tn, fp, fn, tp = confusion_matrix(truth, score >= t, labels=[False, True]).ravel()
        balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2
        f1 = tp / max(1, 2 * tp + fp + fn) + tn / max(1, 2 * tn + fp + fn)
        return (balanced + f1, -abs(t - .5))
    return float(max(np.linspace(.05, .95, 19), key=rank))

def fit_neural(name, x, p, y, xv, pv, yv, epochs, rng, log):
    torch.manual_seed(SEED)
    model = Network(name)
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    pos_weight = min(15., (1-y.mean()) / y.mean())
    loss_state = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    power_scale = max(.1, float(np.quantile(p[y], .9)))
    def loss(data, power, state):
        logit, estimate = model(torch.from_numpy(data))
        return loss_state(logit, torch.from_numpy(state.astype('float32'))) + nn.functional.smooth_l1_loss(
            estimate, torch.from_numpy(power / power_scale))
    best = float('inf')
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/256)))):
            batch = x[ids].copy()
            batch[:, :, 0] += rng.normal(0, .10, (len(ids), 1))
            opt.zero_grad()
            loss(batch, p[ids], y[ids]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            value = float(loss(xv, pv, yv))
        log.append(value)
        if value < best:
            best = value
            weights = {key: val.detach().clone() for key, val in model.state_dict().items()}
    model.load_state_dict(weights)
    def predict(array):
        with torch.no_grad():
            batches = [model(chunk) for chunk in torch.from_numpy(array).split(1024)]
        return (torch.cat([a for a, _ in batches]).sigmoid().numpy(),
                torch.cat([b for _, b in batches]).numpy() * power_scale)
    return predict, weights, power_scale

def load(cache):
    regions = []
    for region in REGIONS:
        with np.load(cache / f'{region}.npz') as data:
            regions.append(dict(**{name: data[name] for name in data.files},
                                region=np.full(len(data['home']), region)))
    return {key: np.concatenate([r[key] for r in regions]) for key in regions[0]}

def splits(data):
    h, r = data['home'], data['region']
    held = set()
    for region in REGIONS:
        homes = np.unique(h[r == region])
        np.random.default_rng(SEED).shuffle(homes)
        held.update(homes[:max(1, len(homes)//4)])
    for label, test in [('Unseen households', np.isin(h, sorted(held))), *[
        (f'Unseen region / {region}', np.isin(h, np.unique(h[r == region]))) for region in REGIONS]]:
        train = np.zeros(len(h), dtype=bool)
        val = train.copy()
        for home in np.unique(h[~test]):
            indices = np.flatnonzero((h == home) & ~test)
            times = data['time'][indices]
            boundary = np.quantile(times, .8)
            train[indices[times < boundary]] = True
            val[indices[times >= boundary + WINDOW]] = True
        assert not (set(h[test]) & set(h[train | val]))
        yield label, train, val, test

def sample(mask, homes, cap, rng):
    indices = []
    for home in np.unique(homes[mask]):
        pool = np.flatnonzero(mask & (homes == home))
        indices.append(rng.choice(pool, min(len(pool), cap), replace=False))
    return np.concatenate(indices) if indices else np.empty(0, dtype=int)

def run(cache, output, epochs, cap):
    data = load(cache)
    rng = np.random.default_rng(SEED)
    rows, traces, logs, manifest = [], [], [], []
    bundle = {}
    for fold, train, val, test in splits(data):
        print(fold, flush=True)
        manifest.append(dict(protocol=fold, development=sorted(set(data['home'][train | val])),
                             test=sorted(set(data['home'][test]))))
        for j, circuit in enumerate(APPLIANCES):
            watts = data['y'][:, j]
            valid = np.isfinite(watts) & np.isfinite(data['x']).all(axis=1)
            it = sample(train & valid, data['home'], cap, rng)
            iv = sample(val & valid, data['home'], max(250, cap//3), rng)
            ie = np.flatnonzero(test & valid)
            if min(len(it), len(iv), len(ie)) < 2:
                print('Insufficient data:', circuit, flush=True)
                continue
            positives = watts[it][watts[it] >= .015]
            if len(positives) < 2:
                continue
            # Learn the midpoint between two power regimes on the log scale.
            levels = np.log1p(watts[it] * 1000).reshape(-1, 1)
            centers = KMeans(n_clusters=2, random_state=SEED, n_init=5).fit(levels).cluster_centers_
            threshold = float(np.expm1(centers.mean()) / 1000)
            states = watts > threshold
            if min(len(np.unique(states[it])), len(np.unique(states[iv]))) < 2:
                print('One class in development:', circuit, flush=True)
                continue
            center = float(np.median(data['x'][it]))
            scale = max(.1, float(np.quantile(data['x'][it], .75) - np.quantile(data['x'][it], .25)))
            x, xv, xe = (features(data['x'][ids], center, scale) for ids in (it, iv, ie))
            print(circuit, len(it), len(iv), len(ie), flush=True)
            scores = {}
            saved = {}
            zero = lambda a: (np.zeros(len(a)), np.zeros(len(a)))
            predictors = {'Always off': zero}
            for name, classifier, regressor in [
                ('Linear', LogisticRegression(max_iter=200, class_weight='balanced'), Ridge()),
                ('Gradient boosting', HistGradientBoostingClassifier(max_iter=85, class_weight='balanced',
                    early_stopping=False, random_state=SEED),
                 HistGradientBoostingRegressor(max_iter=85, early_stopping=False, random_state=SEED))]:
                xf = x.reshape(len(x), -1)
                classifier.fit(xf, states[it])
                regressor.fit(xf, watts[it])
                def predict(a, clf=classifier, reg=regressor):
                    flat = a.reshape(len(a), -1)
                    return clf.predict_proba(flat)[:, 1], reg.predict(flat).clip(0)
                predictors[name] = predict
                saved[name] = (classifier, regressor)
            for name in NEURAL:
                history = []
                predictors[name], weights, power_scale = fit_neural(
                    name, x, watts[it], states[it], xv, watts[iv], states[iv], epochs, rng, history)
                logs.extend(dict(protocol=fold, appliance=circuit, model=name, epoch=k+1,
                                 validation_loss=v) for k,v in enumerate(history))
                saved[name] = (weights, power_scale)
            for name, predict in predictors.items():
                scores[name] = (*predict(xv), *predict(xe))
            scores['Neural ensemble'] = tuple(np.mean([scores[name][k] for name in NEURAL], axis=0)
                                                for k in range(4))
            cutoffs = {}
            for name, (sv, _, se, estimate) in scores.items():
                cutoffs[name] = ct = cutoff(states[iv], sv) if name != 'Always off' else .5
                identity = dict(protocol=fold, appliance=circuit, model=name)
                for region in np.unique(data['region'][ie]):
                    mask = data['region'][ie] == region
                    ids = ie[mask]
                    result = metrics(states[ids], se[mask] >= ct, watts[ids], estimate[mask])
                    for home in ['All homes', *np.unique(data['home'][ids])]:
                        if home == 'All homes':
                            subset, sub_est, sub_scores = ids, estimate[mask], se[mask]
                            detail = result
                        else:
                            pick = data['home'][ids] == home
                            subset, sub_est, sub_scores = ids[pick], estimate[mask][pick], se[mask][pick]
                            detail = metrics(states[subset], sub_scores >= ct, watts[subset], sub_est)
                        rows.append(dict(**identity, region=region, home=home, label_threshold_w=threshold*1000,
                                         score_threshold=ct, train_on=int(states[it].sum()),
                                         train_off=int((~states[it]).sum()), validation_on=int(states[iv].sum()),
                                         validation_off=int((~states[iv]).sum()), **detail))
                        if home != 'All homes':
                            # Keep the measured interval with the most observed activity for inspection.
                            days = data['time'][subset] // 1440
                            day, inv = np.unique(days, return_inverse=True)
                            chosen = day[np.bincount(inv, weights=watts[subset]).argmax()]
                            plot = days == chosen
                            traces.append(pd.DataFrame(dict(**identity, region=region, home=home,
                                time=data['time'][subset][plot], actual_w=watts[subset][plot]*1000,
                                predicted_w=sub_est[plot]*1000)))
            bundle[circuit] = dict(center=center, scale=scale, threshold=threshold,
                                   models=saved, cutoffs=cutoffs)
        if fold == 'Unseen households':
            torch.save(bundle, output / 'models.pt')
        pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
        pd.concat(traces, ignore_index=True).to_csv(output / 'traces.csv.gz', index=False)
        pd.DataFrame(logs).to_csv(output / 'training_log.csv', index=False)
        (output / 'splits.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path('cache'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--cap', type=int, default=1800, help='Training windows per home and appliance')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run(args.cache, args.output, args.epochs, args.cap)


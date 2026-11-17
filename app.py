from pathlib import Path
import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from prepare import APPLIANCES, WINDOW, NEURAL

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
st.set_page_config(page_title='Household Load Lab', page_icon='⚡', layout='wide')
st.markdown("""<style>
[data-testid=stHeader] {background:#fff}
.block-container {max-width:1320px;padding-top:2rem}
.hero {background:linear-gradient(120deg,#32125e,#7130c8);color:white;
border-radius:22px;padding:2rem 2.4rem;margin-bottom:1.4rem}
.hero h1 {color:white;margin:0;padding:0;font-size:2.5rem}
.hero p {color:#efe1ff;margin:.5rem 0 0}
[data-testid=stMetric] {background:#f5efff;border:1px solid #e8d8ff;border-radius:12px;padding:14px}
</style><div class="hero"><h1>Household Load Lab</h1>
<p>Appliance detection and power estimation on unseen households</p></div>""", unsafe_allow_html=True)

@st.cache_data
def read(path, updated):
    return pd.read_csv(path, dtype={'home': str})

def csv(name):
    path = RESULTS / name
    return read(path, path.stat().st_mtime_ns)

def combined(base, hybrid):
    original = csv(base)
    if (RESULTS / hybrid).exists():
        return pd.concat([original, csv(hybrid)], ignore_index=True)
    return original

def split_manifest():
    return json.loads((RESULTS / 'splits.json').read_text(encoding='utf-8'))

@st.cache_resource
def fitted():
    import torch

    torch.set_num_threads(2)
    return torch.load(RESULTS / 'models.pt', weights_only=False, map_location='cpu')

@st.cache_resource
def fitted_hybrids():
    import torch

    return torch.load(RESULTS / 'hybrids.pt', weights_only=True, map_location='cpu')

def predict(upload, circuit, model_name):
    import torch
    from train import Network, features

    data = pd.read_csv(upload, dtype={'dataid': str})
    required = {'dataid', 'localminute', 'aggregate_kw'}
    if missing := required - set(data):
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    data.localminute = pd.to_datetime(data.localminute, format='mixed', errors='raise', utc=True)
    data.aggregate_kw = pd.to_numeric(data.aggregate_kw, errors='raise')
    if data.dataid.isna().any() or data.localminute.isna().any() or not np.isfinite(data.aggregate_kw).all():
        raise ValueError('Every row needs a valid household ID, timestamp and aggregate power.')
    data = data.sort_values(['dataid', 'localminute']).drop_duplicates(['dataid', 'localminute'])
    config = fitted()[circuit]
    hybrid = None
    if model_name.startswith('Hybrid:'):
        from hybrids import infer

        hybrid = fitted_hybrids()[circuit][model_name]
    frames = []
    for _, home in data.groupby('dataid'):
        if len(home) < WINDOW:
            continue
        utc = home.localminute.to_numpy(dtype='datetime64[s]').astype('int64')
        times = np.lib.stride_tricks.sliding_window_view(utc, WINDOW)
        contiguous = (np.diff(times, axis=1) == 60).all(axis=1)
        if not contiguous.any():
            continue
        windows = np.lib.stride_tricks.sliding_window_view(home.aggregate_kw.to_numpy('float32'), WINDOW)
        x = features(windows[contiguous], config['center'], config['scale'])
        chosen = home.iloc[np.flatnonzero(contiguous) + WINDOW - 1].copy()
        if hybrid is not None:
            score, kw = infer(config, x, model_name, hybrid)
        elif model_name in NEURAL or model_name == 'Neural ensemble':
            outputs = []
            for name in NEURAL if model_name == 'Neural ensemble' else [model_name]:
                weights, scale = config['models'][name]
                network = Network(name)
                network.load_state_dict(weights)
                network.eval()
                with torch.inference_mode():
                    batches = [network(batch) for batch in torch.from_numpy(x).split(1024)]
                    outputs.append((torch.cat([a for a, _ in batches]).sigmoid().numpy(),
                                    torch.cat([b for _, b in batches]).numpy() * scale))
            score, kw = np.mean(outputs, axis=0)
        else:
            classifier, regressor = config['models'][model_name]
            flat = x.reshape(len(x), -1)
            score, kw = classifier.predict_proba(flat)[:, 1], regressor.predict(flat).clip(0)
        threshold = hybrid['cutoff'] if hybrid is not None else config['cutoffs'][model_name]
        chosen['predicted_on'] = score >= threshold
        chosen['predicted_w'] = kw * 1000
        frames.append(chosen)
    if not frames:
        raise ValueError('The CSV needs at least one uninterrupted 30-minute household window.')
    return pd.concat(frames, ignore_index=True)

if not (RESULTS / 'metrics.csv').exists():
    st.info('Run train.py to generate the benchmark and model files.')
    st.stop()

data = combined('metrics.csv', 'hybrid_metrics.csv')
households, comparison, transfer, upload = st.tabs(
    ['Household splits', 'Model comparison', 'Dataset shift', 'Evaluate new CSV'])

with households:
    st.subheader('Training, validation and test households')
    manifest = {fold['protocol']: fold for fold in split_manifest()}
    split_protocol = st.selectbox('Benchmark protocol', list(manifest), key='household_protocol')
    fold = manifest[split_protocol]
    st.caption('Training and validation can contain the same household ID because they use earlier '
               'and later readings, respectively. The test homes are excluded from both.')
    groups = {'Training': fold['train'], 'Validation': fold['validation'], 'Testing': fold['test']}
    panels = st.columns(3)
    for panel, (name, ids) in zip(panels, groups.items()):
        with panel:
            st.metric(name, len(ids))
            st.markdown(', '.join(ids))
    rows = [{'Split': name, 'Household ID': home} for name, ids in groups.items() for home in ids]
    st.download_button('Download household IDs', pd.DataFrame(rows).to_csv(index=False),
                       'household-splits.csv', key='download_household_splits')
with comparison:
    cols = st.columns(4)
    protocol = cols[0].selectbox('Test protocol', data.protocol.unique())
    pool = data[data.protocol == protocol]
    region = cols[1].selectbox('Test region', pool.region.unique())
    pool = pool[pool.region == region]
    circuit = cols[2].selectbox('Appliance', pool.appliance.unique(), format_func=APPLIANCES.get)
    pool = pool[pool.appliance == circuit]
    home = cols[3].selectbox('Household', pool.home.unique())
    pool = pool[pool.home == home].sort_values('macro_f1', ascending=False)
    chosen = st.selectbox('Model details', pool.model)
    row = pool[pool.model == chosen].iloc[0]
    cells = st.columns(5)
    for cell, label, value in zip(cells, ['Accuracy','Balanced accuracy','Macro F1','MAE','RMSE'],
        [f'{row.accuracy:.1%}', f'{row.balanced_accuracy:.1%}' if pd.notna(row.balanced_accuracy) else 'One class',
         f'{row.macro_f1:.3f}', f'{row.mae_w:.1f} W', f'{row.rmse_w:.1f} W']):
        cell.metric(label, value)
    passed = row.accuracy >= .85 and row.balanced_accuracy >= .75 and row.macro_f1 >= .70
    st.caption(f'{row.on:,} ON · {row.off:,} OFF · {row.label_threshold_w:.1f} W label threshold · '
               + ('Meets minimum benchmark' if passed else 'Minimum benchmark unmet or undefined'))
    main, side = st.columns([2,1])
    with main:
        st.subheader(f'Model leaderboard · {len(pool)} models')
        keys = ['model','accuracy','balanced_accuracy','macro_f1','mae_w','rmse_w']
        st.dataframe(pool[keys], hide_index=True, width='stretch')
    with side:
        st.subheader('Confusion matrix')
        st.dataframe(pd.DataFrame([[row.tn,row.fp],[row.fn,row.tp]],
            columns=['Predicted OFF','Predicted ON'], index=['Actual OFF','Actual ON']))
    st.subheader('Measured and estimated power')
    trace_file = 'hybrid_traces.csv.gz' if chosen.startswith('Hybrid:') else 'traces.csv.gz'
    traces = csv(trace_file)
    trace = traces[(traces.protocol == protocol) & (traces.region == region)
                   & (traces.appliance == circuit) & (traces.model == chosen)]
    plot_home = st.selectbox('Plot household', trace.home.unique() if home == 'All homes' else [home])
    trace = trace[trace.home == plot_home].copy()
    trace['UTC'] = pd.to_datetime(trace.time, unit='m', utc=True)
    plot = trace.melt(id_vars='UTC', value_vars=['actual_w','predicted_w'], var_name='Series', value_name='Power (W)')
    plot['Series'] = plot.Series.map({'actual_w':'Measured','predicted_w':'Estimated'})
    st.altair_chart(alt.Chart(plot).mark_line().encode(x='UTC:T', y='Power (W):Q',
        color=alt.Color('Series:N', scale=alt.Scale(domain=['Measured','Estimated'], range=['#281045','#a368f5'])),
        tooltip=['UTC:T','Series:N','Power (W):Q']).properties(height=320).interactive(), width='stretch')
    st.caption('The plot shows the sampled UTC day with the most measured appliance energy for this household.')
    st.download_button('Download full benchmark CSV', data.to_csv(index=False), 'metrics.csv')

with transfer:
    st.subheader('How performance changes across regions')
    st.caption('Unseen households uses one fixed model per appliance across all test regions. '
               'Unseen region excludes every household from the named region during development.')
    s1, s2 = st.columns(2)
    shift_appliance = s1.selectbox('Appliance', data.appliance.unique(), format_func=APPLIANCES.get, key='shift_appliance')
    shift_model = s2.selectbox('Model', data.model.unique(), key='shift_model')
    if shift_model.startswith('Hybrid:'):
        st.caption('Hybrid scores use the fixed-model unseen-household test across available regions.')
    shift = data[(data.home == 'All homes') & (data.appliance == shift_appliance) & (data.model == shift_model)].copy()
    shift['Protocol'] = np.where(shift.protocol == 'Unseen households', 'Unseen households', 'Unseen region')
    st.altair_chart(alt.Chart(shift).mark_bar().encode(x='region:N', xOffset='Protocol:N',
        y=alt.Y('macro_f1:Q', title='Macro F1', scale=alt.Scale(domain=[0,1])),
        color=alt.Color('Protocol:N', scale=alt.Scale(range=['#5d20ad','#b58bee'])),
        tooltip=['region:N','Protocol:N','macro_f1:Q','balanced_accuracy:Q','mae_w:Q']).properties(height=300), width='stretch')
    st.caption('A missing bar means this test cohort has no usable labels for the selected appliance.')
    st.dataframe(shift[['region','Protocol','accuracy','balanced_accuracy','macro_f1','mae_w','rmse_w','on','off']],
                 hide_index=True, width='stretch')

with upload:
    st.subheader('Evaluate a new household')
    st.markdown('Upload a 1-minute CSV with **dataid**, **localminute**, and **aggregate_kw**. '
                'Add the selected appliance circuit in kW to score the predictions.')
    appliance = st.selectbox('Appliance circuit', data.loc[data.protocol == 'Unseen households', 'appliance'].unique(), format_func=APPLIANCES.get)
    choices = ['Neural ensemble', *NEURAL, 'Gradient boosting', 'Linear']
    if (RESULTS/'hybrids.pt').exists():
        choices += list(data.loc[(data.appliance == appliance) &
                                 data.model.str.startswith('Hybrid:'), 'model'].unique())
    model = st.selectbox('Model', choices)
    file = st.file_uploader('Household CSV', type=['csv'])
    if file:
        try:
            prediction = predict(file, appliance, model)
            st.dataframe(prediction.head(200), hide_index=True)
            st.download_button('Download predictions', prediction.to_csv(index=False), 'predictions.csv', key='pred')
            if appliance in prediction:
                from train import metrics

                manifest = split_manifest()[0]
                actual = pd.to_numeric(prediction[appliance], errors='coerce')
                valid = np.isfinite(actual) & ~prediction.dataid.isin(manifest['development'])
                if valid.any():
                    actual = actual[valid].clip(lower=0).to_numpy()
                    score = metrics(actual > fitted()[appliance]['threshold'], prediction.predicted_on[valid].to_numpy(),
                                    actual, prediction.predicted_w[valid].to_numpy()/1000)
                    st.caption('Metrics include labeled rows whose IDs are absent from the development split.')
                    st.dataframe(pd.DataFrame([score]), hide_index=True)
                else:
                    st.info('Scores require valid appliance labels from a household outside the development split.')
        except (ValueError, KeyError) as error:
            st.error(str(error))



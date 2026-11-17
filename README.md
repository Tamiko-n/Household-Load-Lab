# Household Load Lab

Predict appliance ON/OFF state and power from 30 consecutive one-minute aggregate meter readings.

## Run the dashboard

Open PowerShell in this folder. The bundled results can be used immediately with the project dependencies installed.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The dashboard includes model comparisons, region-transfer results, confusion matrices, power traces and CSV downloads. For inference with saved weights, upload one-minute readings with columns `dataid,localminute,aggregate_kw`. Use stable Pecan Street household IDs for overlap checks, or identifiers from a genuinely separate dataset. Include the selected appliance column in kW for scored evaluation. Uploaded power must use the same aggregate definition as the training data. Only 30-minute continuous windows are scored.

## Reproduce the benchmark

```powershell
python prepare.py --data E:/ann --cache ../work/cache
python train.py --cache ../work/cache --output results --epochs 5 --cap 1100
```

Use Python 3.11 and the supplied regional directory layout. The bundled weights were saved with scikit-learn 1.9.1 and PyTorch 2.15.0 CPU. Preparation reads the three city power files and the three Puerto Rico real-power files in chunks. Training seed is 42. The result files contain scores, validation losses, household IDs, representative prediction traces and pooled model weights. Load only the project-generated model bundle from a trusted source.

## Evaluation design

- **Unseen households.** Globally withhold household IDs sampled from each region. A single set of trained models scores all four test regions. This measures variation across datasets with the model held fixed.
- **Unseen region.** Withhold all household IDs found in the target region, including any matching IDs elsewhere. Train on the other regions and repeat for each target region.
- Split development data chronologically within each household, using the earlier 80% of its time span for training and the later span for validation. Leave a 30-minute boundary buffer so no input window overlaps the two sets.
- Center and scale aggregate readings using development-training medians and IQRs. Derive input features from aggregate level and first difference. Choose ON/OFF label thresholds using two-cluster log-power partitioning on training appliance readings, with the midpoint between cluster centers converted to watts. Choose prediction cutoffs and neural checkpoints on validation.
- Compare always-OFF, linear logistic/Ridge, histogram gradient boosting, MLP, 1D CNN, GRU, LSTM and a mean neural ensemble. Neural models use weighted state loss, a power-estimation head and aggregate level jitter during training. Cap training windows per household and appliance to reduce dominance by homes with many readings.

## Source columns and cleaning

`dataid` is the household identifier. `localminute` is parsed into UTC. The aggregate estimate assumes signed battery generation and uses `grid + solar + solar2 + battery1` in kW, with absent generation channels treated as zero. This is an aggregate reconstruction assumption for the supplied metering scheme. The three targets are `air1`, `refrigerator1` and `dishwasher1` in kW. Keep missing appliance labels missing, ignore missing aggregate rows, clip negative appliance power to zero, and deduplicate home-timestamp pairs. Sample about one in ten calendar days deterministically, and score every fifth minute with a full contiguous input window. Puerto Rico has no usable dishwasher labels. MAE and RMSE are shown in watts. Balanced accuracy is undefined when a test slice contains only one class.

The brief's minimum per-appliance classification targets are 85% accuracy, 75% balanced accuracy and 0.70 macro F1. The higher targets are 90%, 85% and 0.80. Scores can change with metering, appliance mix and regional behavior. The dataset-shift comparisons quantify that variation.


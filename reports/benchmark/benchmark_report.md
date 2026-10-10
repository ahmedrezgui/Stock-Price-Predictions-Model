# 5-stock forecasting benchmark

## Scope and decision

Stocks: **BIAT, ATB, ARTES, NAKL, CIL**. Neural training and prediction device: **cuda**; GPU: **NVIDIA GeForce RTX 3050 Ti Laptop GPU**. ARIMA runs on CPU.

The lower equal-stock mean validation relative RMSE among neural return models is **LSTM_return**. The lowest validation score across return models/baselines is **LSTM_return**. This is a retrospective comparison, not deployment approval or a profitability claim. Test results do not choose the winner.

## Model-choice rationale

**Provisional return-model choice: LSTM_return.** The selection rule is the lowest equal-stock mean validation relative RMSE, after averaging seeds within each stock. Direction models are evaluated separately because probability prediction is a different task.

The candidate scores **0.997116** on validation (historical-mean baseline = 1), an error reduction of **0.288%** relative to that baseline. It beats the baseline on **2/5** stocks and has the lowest validation return error on **1/5** stocks.

On test, this candidate scores **1.000522**; the lowest observed test error belongs to **ARIMA** (**0.996132**). Test rankings describe generalization and do not change the validation-based choice. A score above 1 means worse error than the baseline.

This supports a candidate for further validation, not an unconditional production winner. Small average differences and inconsistent stock-level results do not establish a robust advantage; no statistical significance claim is made. Keep ARIMA and the simple baselines as challengers. Before deployment, freeze the candidate settings and evaluate newly collected, unseen observations against those challengers with a predefined acceptance rule. If the candidate does not improve reliably, keep the stronger baseline rather than selecting LSTM because it was preferred.

## Fixed protocol

- Start requested: 2010-01-01; as-of cutoff: 2026-10-10; each stock uses all available completed rows within that range.
- Horizon: 1 observed bar(s). Unverified calendar means these are not guaranteed exchange sessions.
- Maximum epochs: 50; patience: 10; seeds: [7, 19, 43].
- Window: 20; hidden units: 32; layers: 1; dropout: 0.1.
- Batch size: 64; learning rate: 0.001; weight decay: 0.0001.
- Return objective: huber on training-standardized targets; direction objective: binary cross entropy.
- Same public LSTMModel / GRUModel and trainer as routine training. Benchmark overrides history, epoch budget and device; it never refits on test or promotes models.
- Training before 2022; validation in 2022–2023; test from 2024 onward. Future labels crossing boundaries are purged. Scalers use training inputs only.
- Both families run each seed for return prediction and separately for direction probability. Validation restores the best epoch; no test-based hyperparameter, seed or trading-threshold selection.
- Model tables average metrics across seeds (not ensemble predictions). Stocks receive equal weight. Relative RMSE uses each stock's historical-mean baseline; relative Brier uses its training-frequency baseline. Values below 1 beat that baseline.
- Forecast, confusion, probability and strategy figures use seed 7 fixed in advance. Learning curves include all seeds.
- Neural latency includes input transfer and CPU result transfer, with warmup excluded and CUDA synchronization. ARIMA latency includes sequential state updates; these workloads differ. GPU speedup over CPU was not measured.
- Stocks were selected for history coverage and valid local OHLCV, not benchmark performance. Actual coverage appears in each stock section. Invalid stocks were not silently repaired.
- Existing historical data and some test periods have been inspected in prior work; this is not a newly unseen prospective test. Unverified corporate actions, data revisions and missing-session evidence limit conclusions.

## Overall results

| phase | model | relative_rmse | relative_brier | mae | accuracy | training_seconds | prediction_ms_per_sample |
|---|---|---|---|---|---|---|---|
| validation | zero_return | 1.000021 | — | 0.010347 | 0.642588 | — | — |
| validation | historical_mean | 1.000000 | — | 0.010387 | 0.624406 | — | — |
| validation | historical_probability | — | 1.000000 | — | 0.642588 | — | — |
| validation | majority_direction | — | — | — | 0.642588 | — | — |
| test | zero_return | 0.999859 | — | 0.009992 | 0.611236 | — | — |
| test | historical_mean | 1.000000 | — | 0.010029 | 0.582915 | — | — |
| test | historical_probability | — | 1.000000 | — | 0.611236 | — | — |
| test | majority_direction | — | — | — | 0.611236 | — | — |
| validation | ARIMA | 1.002528 | — | 0.010507 | 0.583484 | 1.031812 | 9.960360 |
| test | ARIMA | 0.996132 | — | 0.010084 | 0.549538 | 1.031812 | 10.946123 |
| validation | LSTM_return | 0.997116 | — | 0.010657 | 0.517888 | 2.243571 | 0.011492 |
| test | LSTM_return | 1.000522 | — | 0.010279 | 0.533037 | 2.243571 | 0.010791 |
| validation | GRU_return | 1.000840 | — | 0.010664 | 0.522724 | 2.127111 | 0.010915 |
| test | GRU_return | 0.997867 | — | 0.010243 | 0.542034 | 2.127111 | 0.010352 |
| validation | LSTM_direction | — | 0.992823 | — | 0.635391 | 2.253139 | 0.010741 |
| test | LSTM_direction | — | 0.992681 | — | 0.614925 | 2.253139 | 0.010850 |
| validation | GRU_direction | — | 0.994404 | — | 0.636752 | 2.369171 | 0.010925 |
| test | GRU_direction | — | 0.989199 | — | 0.613658 | 2.369171 | 0.011150 |

![Across-stock comparison](figures/00_stock_comparison.png)

## Reading the metrics

RMSE and MAE measure errors in decimal log returns (lower is better). RMSE emphasizes large errors. Accuracy is the fraction of correct positive/nonpositive signs; compare it with the majority baseline. Brier measures squared probability error (lower is better); ROC-AUC measures ranking (higher is better). Probabilities are uncalibrated sigmoid outputs from trained direction heads, not signs disguised as confidence. Unavailable metrics appear as —. Seed standard deviation describes training variability, not a confidence interval.

Strategies form signals after close, enter at the next open, and hold a nonoverlapping fixed horizon. Costs are 10 bps on each side. Probability signals use 0.5. Buy-and-hold is gross. Zero-volume execution bars make this simple simulation unavailable. Charts are research diagnostics, not deployable trading evidence.

## BIAT

4127 rows: 2010-01-04 to 2026-10-07. ARIMA order: (0, 0, 0). Input SHA-256: `d1405d91963facbf83693f402fbf269019b670f3d1b44f3deeace8b9d46c3174`.

| split | n | first_origin | last_origin | last_label |
|---|---|---|---|---|
| train | 2894 | 2010-03-01 00:00:00 | 2021-12-30 00:00:00 | 2021-12-31 00:00:00 |
| validation | 506 | 2022-01-03 00:00:00 | 2023-12-28 00:00:00 | 2023-12-29 00:00:00 |
| test | 685 | 2024-01-02 00:00:00 | 2026-10-06 00:00:00 | 2026-10-07 00:00:00 |

| phase | model | rmse | rmse_std | mae | accuracy | brier | auc | training_seconds | prediction_ms_per_sample | relative_rmse | relative_brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | zero_return | 0.010701 | — | 0.006319 | 0.545455 | — | — | — | — | 1.001174 | — |
| validation | historical_mean | 0.010689 | — | 0.006336 | 0.454545 | — | — | — | — | 1.000000 | — |
| validation | historical_probability | — | — | — | 0.545455 | 0.250029 | 0.500000 | — | — | — | 1.000000 |
| validation | majority_direction | — | — | — | 0.545455 | — | — | — | — | — | — |
| test | zero_return | 0.011434 | — | 0.006515 | 0.570803 | — | — | — | — | 1.000982 | — |
| test | historical_mean | 0.011423 | — | 0.006541 | 0.429197 | — | — | — | — | 1.000000 | — |
| test | historical_probability | — | — | — | 0.570803 | 0.245404 | 0.500000 | — | — | — | 1.000000 |
| test | majority_direction | — | — | — | 0.570803 | — | — | — | — | — | — |
| validation | ARIMA | 0.010687 | — | 0.006341 | 0.454545 | — | — | 0.954219 | 9.950120 | 0.999793 | — |
| test | ARIMA | 0.011421 | — | 0.006548 | 0.429197 | — | — | 0.954219 | 11.043722 | 0.999828 | — |
| validation | LSTM_return | 0.010724 | 0.000060 | 0.006465 | 0.467062 | — | — | 2.010307 | 0.009285 | 1.003326 | — |
| test | LSTM_return | 0.011443 | 0.000010 | 0.006648 | 0.475426 | — | — | 2.010307 | 0.009459 | 1.001762 | — |
| validation | GRU_return | 0.010802 | 0.000046 | 0.006579 | 0.453887 | — | — | 1.940463 | 0.009276 | 1.010567 | — |
| test | GRU_return | 0.011480 | 0.000018 | 0.006736 | 0.490024 | — | — | 1.940463 | 0.008949 | 1.005009 | — |
| validation | LSTM_direction | — | — | — | 0.536891 | 0.249819 | 0.472994 | 2.247555 | 0.012624 | — | 0.999162 |
| test | LSTM_direction | — | — | — | 0.568856 | 0.245511 | 0.515203 | 2.247555 | 0.012518 | — | 1.000435 |
| validation | GRU_direction | — | — | — | 0.519763 | 0.250387 | 0.464052 | 2.205495 | 0.011934 | — | 1.001433 |
| test | GRU_direction | — | — | — | 0.560097 | 0.246627 | 0.497996 | 2.205495 | 0.010680 | — | 1.004983 |

### Epochs actually used

| model | seed | epochs_run | best_epoch | device | seconds |
|---|---|---|---|---|---|
| LSTM_return | 7 | 11 | 1 | cuda | 2.348595 |
| LSTM_return | 19 | 11 | 1 | cuda | 1.852469 |
| LSTM_return | 43 | 11 | 1 | cuda | 1.829858 |
| GRU_return | 7 | 13 | 3 | cuda | 2.101014 |
| GRU_return | 19 | 11 | 1 | cuda | 1.700836 |
| GRU_return | 43 | 11 | 1 | cuda | 2.019540 |
| LSTM_direction | 7 | 11 | 1 | cuda | 2.240519 |
| LSTM_direction | 19 | 11 | 1 | cuda | 2.355876 |
| LSTM_direction | 43 | 11 | 1 | cuda | 2.146270 |
| GRU_direction | 7 | 11 | 1 | cuda | 2.124217 |
| GRU_direction | 19 | 11 | 1 | cuda | 2.267847 |
| GRU_direction | 43 | 11 | 1 | cuda | 2.224422 |

### Strategy diagnostics

| model | net_return | max_drawdown | trades | status |
|---|---|---|---|---|
| zero_return | 0.000000 | 0.000000 | 0 | available |
| historical_mean | -0.547520 | -0.562108 | 684 | available |
| historical_probability | 0.000000 | 0.000000 | 0 | available |
| majority_direction | 0.000000 | 0.000000 | 0 | available |
| ARIMA | -0.547520 | -0.562108 | 684 | available |
| LSTM_return | -0.466909 | -0.499074 | 463 | available |
| GRU_return | -0.428891 | -0.458898 | 458 | available |
| LSTM_direction | -0.003780 | -0.003780 | 1 | available |
| GRU_direction | -0.240329 | -0.240329 | 36 | available |

### History and splits

![BIAT: History and splits](figures/BIAT/01_history.png)

### Training and validation curves

![BIAT: Training and validation curves](figures/BIAT/02_training.png)

### Forecasts and errors

![BIAT: Forecasts and errors](figures/BIAT/03_forecast.png)

### Confusion matrices

![BIAT: Confusion matrices](figures/BIAT/04_confusion.png)

### Probability diagnostics

![BIAT: Probability diagnostics](figures/BIAT/05_probability.png)

### Strategy and drawdown

![BIAT: Strategy and drawdown](figures/BIAT/06_strategy.png)


## ATB

3986 rows: 2010-01-04 to 2026-10-07. ARIMA order: (1, 0, 0). Input SHA-256: `0516228716c22e3d897e8657cd6a396566a9ce7b2cf873d3244921a3a1037a93`.

| split | n | first_origin | last_origin | last_label |
|---|---|---|---|---|
| train | 2799 | 2010-03-01 00:00:00 | 2021-12-30 00:00:00 | 2021-12-31 00:00:00 |
| validation | 485 | 2022-01-03 00:00:00 | 2023-12-28 00:00:00 | 2023-12-29 00:00:00 |
| test | 660 | 2024-01-02 00:00:00 | 2026-10-06 00:00:00 | 2026-10-07 00:00:00 |

| phase | model | rmse | rmse_std | mae | accuracy | brier | auc | training_seconds | prediction_ms_per_sample | relative_rmse | relative_brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | zero_return | 0.016119 | — | 0.010771 | 0.624742 | — | — | — | — | 0.999032 | — |
| validation | historical_mean | 0.016135 | — | 0.010881 | 0.624742 | — | — | — | — | 1.000000 | — |
| validation | historical_probability | — | — | — | 0.624742 | 0.236319 | 0.500000 | — | — | — | 1.000000 |
| validation | majority_direction | — | — | — | 0.624742 | — | — | — | — | — | — |
| test | zero_return | 0.016770 | — | 0.011350 | 0.619697 | — | — | — | — | 0.999027 | — |
| test | historical_mean | 0.016787 | — | 0.011453 | 0.619697 | — | — | — | — | 1.000000 | — |
| test | historical_probability | — | — | — | 0.619697 | 0.238015 | 0.500000 | — | — | — | 1.000000 |
| test | majority_direction | — | — | — | 0.619697 | — | — | — | — | — | — |
| validation | ARIMA | 0.016135 | — | 0.010875 | 0.624742 | — | — | 1.380705 | 11.697447 | 1.000020 | — |
| test | ARIMA | 0.016783 | — | 0.011443 | 0.619697 | — | — | 1.380705 | 12.356650 | 0.999803 | — |
| validation | LSTM_return | 0.016206 | 0.000096 | 0.011059 | 0.615808 | — | — | 2.346381 | 0.012779 | 1.004404 | — |
| test | LSTM_return | 0.016794 | 0.000001 | 0.011547 | 0.614141 | — | — | 2.346381 | 0.012109 | 1.000426 | — |
| validation | GRU_return | 0.016259 | 0.000079 | 0.011076 | 0.570447 | — | — | 1.777034 | 0.013357 | 1.007705 | — |
| test | GRU_return | 0.016801 | 0.000056 | 0.011545 | 0.604040 | — | — | 1.777034 | 0.013920 | 1.000831 | — |
| validation | LSTM_direction | — | — | — | 0.624055 | 0.232255 | 0.568630 | 2.741264 | 0.010584 | — | 0.982802 |
| test | LSTM_direction | — | — | — | 0.619192 | 0.235707 | 0.546609 | 2.741264 | 0.010853 | — | 0.990302 |
| validation | GRU_direction | — | — | — | 0.627491 | 0.233237 | 0.547812 | 3.251915 | 0.010916 | — | 0.986959 |
| test | GRU_direction | — | — | — | 0.623232 | 0.234970 | 0.544304 | 3.251915 | 0.011792 | — | 0.987205 |

### Epochs actually used

| model | seed | epochs_run | best_epoch | device | seconds |
|---|---|---|---|---|---|
| LSTM_return | 7 | 11 | 1 | cuda | 2.406239 |
| LSTM_return | 19 | 11 | 1 | cuda | 2.282711 |
| LSTM_return | 43 | 14 | 4 | cuda | 2.350193 |
| GRU_return | 7 | 11 | 1 | cuda | 1.719401 |
| GRU_return | 19 | 11 | 1 | cuda | 1.750260 |
| GRU_return | 43 | 11 | 1 | cuda | 1.861440 |
| LSTM_direction | 7 | 14 | 4 | cuda | 2.281760 |
| LSTM_direction | 19 | 20 | 10 | cuda | 3.302260 |
| LSTM_direction | 43 | 16 | 6 | cuda | 2.639770 |
| GRU_direction | 7 | 16 | 6 | cuda | 2.526941 |
| GRU_direction | 19 | 20 | 10 | cuda | 3.119314 |
| GRU_direction | 43 | 25 | 15 | cuda | 4.109489 |

### Strategy diagnostics

| model | net_return | max_drawdown | trades | status |
|---|---|---|---|---|
| zero_return | 0.000000 | 0.000000 | 0 | available |
| historical_mean | 0.000000 | 0.000000 | 0 | available |
| historical_probability | 0.000000 | 0.000000 | 0 | available |
| majority_direction | 0.000000 | 0.000000 | 0 | available |
| ARIMA | 0.000000 | 0.000000 | 0 | available |
| LSTM_return | -0.030841 | -0.113481 | 14 | available |
| GRU_return | -0.036181 | -0.262408 | 243 | available |
| LSTM_direction | -0.037874 | -0.037874 | 1 | available |
| GRU_direction | -0.040989 | -0.041312 | 2 | available |

### History and splits

![ATB: History and splits](figures/ATB/01_history.png)

### Training and validation curves

![ATB: Training and validation curves](figures/ATB/02_training.png)

### Forecasts and errors

![ATB: Forecasts and errors](figures/ATB/03_forecast.png)

### Confusion matrices

![ATB: Confusion matrices](figures/ATB/04_confusion.png)

### Probability diagnostics

![ATB: Probability diagnostics](figures/ATB/05_probability.png)

### Strategy and drawdown

![ATB: Strategy and drawdown](figures/ATB/06_strategy.png)


## ARTES

4075 rows: 2010-01-04 to 2026-10-07. ARIMA order: (0, 0, 1). Input SHA-256: `14fcd4052b9dddb1a6a6821fed7ac27af203c3e1dacd1bdefae446a71df792c2`.

| split | n | first_origin | last_origin | last_label |
|---|---|---|---|---|
| train | 2867 | 2010-03-01 00:00:00 | 2021-12-30 00:00:00 | 2021-12-31 00:00:00 |
| validation | 490 | 2022-01-03 00:00:00 | 2023-12-28 00:00:00 | 2023-12-29 00:00:00 |
| test | 676 | 2024-01-02 00:00:00 | 2026-10-06 00:00:00 | 2026-10-07 00:00:00 |

| phase | model | rmse | rmse_std | mae | accuracy | brier | auc | training_seconds | prediction_ms_per_sample | relative_rmse | relative_brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | zero_return | 0.018129 | — | 0.009450 | 0.657143 | — | — | — | — | 0.999951 | — |
| validation | historical_mean | 0.018130 | — | 0.009465 | 0.657143 | — | — | — | — | 1.000000 | — |
| validation | historical_probability | — | — | — | 0.657143 | 0.225542 | 0.500000 | — | — | — | 1.000000 |
| validation | majority_direction | — | — | — | 0.657143 | — | — | — | — | — | — |
| test | zero_return | 0.015865 | — | 0.010042 | 0.605030 | — | — | — | — | 0.999699 | — |
| test | historical_mean | 0.015870 | — | 0.010055 | 0.605030 | — | — | — | — | 1.000000 | — |
| test | historical_probability | — | — | — | 0.605030 | 0.240320 | 0.500000 | — | — | — | 1.000000 |
| test | majority_direction | — | — | — | 0.605030 | — | — | — | — | — | — |
| validation | ARIMA | 0.018088 | — | 0.009504 | 0.608163 | — | — | 0.852123 | 10.644019 | 0.997678 | — |
| test | ARIMA | 0.015845 | — | 0.010071 | 0.547337 | — | — | 0.852123 | 11.836854 | 0.998440 | — |
| validation | LSTM_return | 0.018186 | 0.000067 | 0.009563 | 0.534014 | — | — | 1.887904 | 0.012337 | 1.003122 | — |
| test | LSTM_return | 0.015950 | 0.000068 | 0.010210 | 0.522189 | — | — | 1.887904 | 0.011447 | 1.005061 | — |
| validation | GRU_return | 0.018231 | 0.000053 | 0.009630 | 0.557143 | — | — | 2.484488 | 0.009654 | 1.005560 | — |
| test | GRU_return | 0.015927 | 0.000009 | 0.010246 | 0.519724 | — | — | 2.484488 | 0.010025 | 1.003636 | — |
| validation | LSTM_direction | — | — | — | 0.655102 | 0.225937 | 0.540668 | 2.167334 | 0.009251 | — | 1.001753 |
| test | LSTM_direction | — | — | — | 0.611933 | 0.239161 | 0.521332 | 2.167334 | 0.011349 | — | 0.995176 |
| validation | GRU_direction | — | — | — | 0.648299 | 0.226978 | 0.539393 | 2.801365 | 0.010602 | — | 1.006368 |
| test | GRU_direction | — | — | — | 0.606016 | 0.238653 | 0.540785 | 2.801365 | 0.011490 | — | 0.993064 |

### Epochs actually used

| model | seed | epochs_run | best_epoch | device | seconds |
|---|---|---|---|---|---|
| LSTM_return | 7 | 11 | 1 | cuda | 1.796976 |
| LSTM_return | 19 | 11 | 1 | cuda | 1.875750 |
| LSTM_return | 43 | 12 | 2 | cuda | 1.990986 |
| GRU_return | 7 | 13 | 3 | cuda | 2.095129 |
| GRU_return | 19 | 15 | 5 | cuda | 2.586662 |
| GRU_return | 43 | 16 | 6 | cuda | 2.771675 |
| LSTM_direction | 7 | 13 | 3 | cuda | 2.191611 |
| LSTM_direction | 19 | 13 | 3 | cuda | 2.131921 |
| LSTM_direction | 43 | 12 | 2 | cuda | 2.178469 |
| GRU_direction | 7 | 16 | 6 | cuda | 2.548819 |
| GRU_direction | 19 | 16 | 6 | cuda | 2.524193 |
| GRU_direction | 43 | 19 | 9 | cuda | 3.331083 |

### Strategy diagnostics

| model | net_return | max_drawdown | trades | status |
|---|---|---|---|---|
| zero_return | 0.000000 | 0.000000 | 0 | available |
| historical_mean | 0.000000 | 0.000000 | 0 | available |
| historical_probability | 0.000000 | 0.000000 | 0 | available |
| majority_direction | 0.000000 | 0.000000 | 0 | available |
| ARIMA | 0.048324 | -0.195915 | 203 | available |
| LSTM_return | -0.368443 | -0.425317 | 345 | available |
| GRU_return | -0.360687 | -0.464633 | 343 | available |
| LSTM_direction | 0.110837 | -0.042160 | 12 | available |
| GRU_direction | 0.023616 | -0.175552 | 62 | available |

### History and splits

![ARTES: History and splits](figures/ARTES/01_history.png)

### Training and validation curves

![ARTES: Training and validation curves](figures/ARTES/02_training.png)

### Forecasts and errors

![ARTES: Forecasts and errors](figures/ARTES/03_forecast.png)

### Confusion matrices

![ARTES: Confusion matrices](figures/ARTES/04_confusion.png)

### Probability diagnostics

![ARTES: Probability diagnostics](figures/ARTES/05_probability.png)

### Strategy and drawdown

![ARTES: Strategy and drawdown](figures/ARTES/06_strategy.png)


## NAKL

3761 rows: 2010-07-16 to 2026-10-07. ARIMA order: (1, 0, 1). Input SHA-256: `6c220e71bf112d53e92f50df7d9c20fbe44ba58db0273828e1d31d5a89efdf57`.

| split | n | first_origin | last_origin | last_label |
|---|---|---|---|---|
| train | 2727 | 2010-09-17 00:00:00 | 2021-12-30 00:00:00 | 2021-12-31 00:00:00 |
| validation | 362 | 2022-01-03 00:00:00 | 2023-12-27 00:00:00 | 2023-12-28 00:00:00 |
| test | 630 | 2024-01-02 00:00:00 | 2026-10-06 00:00:00 | 2026-10-07 00:00:00 |

| phase | model | rmse | rmse_std | mae | accuracy | brier | auc | training_seconds | prediction_ms_per_sample | relative_rmse | relative_brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | zero_return | 0.015633 | — | 0.008304 | 0.734807 | — | — | — | — | 0.999989 | — |
| validation | historical_mean | 0.015633 | — | 0.008322 | 0.734807 | — | — | — | — | 1.000000 | — |
| validation | historical_probability | — | — | — | 0.734807 | 0.196701 | 0.500000 | — | — | — | 1.000000 |
| validation | majority_direction | — | — | — | 0.734807 | — | — | — | — | — | — |
| test | zero_return | 0.016036 | — | 0.010276 | 0.639683 | — | — | — | — | 0.999897 | — |
| test | historical_mean | 0.016037 | — | 0.010281 | 0.639683 | — | — | — | — | 1.000000 | — |
| test | historical_probability | — | — | — | 0.639683 | 0.233223 | 0.500000 | — | — | — | 1.000000 |
| test | majority_direction | — | — | — | 0.639683 | — | — | — | — | — | — |
| validation | ARIMA | 0.015995 | — | 0.008790 | 0.558011 | — | — | 1.212555 | 9.937107 | 1.023147 | — |
| test | ARIMA | 0.016088 | — | 0.010521 | 0.509524 | — | — | 1.212555 | 10.910001 | 1.003181 | — |
| validation | LSTM_return | 0.015386 | 0.000124 | 0.008677 | 0.455801 | — | — | 2.939182 | 0.011192 | 0.984235 | — |
| test | LSTM_return | 0.016374 | 0.000185 | 0.010821 | 0.487302 | — | — | 2.939182 | 0.010022 | 1.020989 | — |
| validation | GRU_return | 0.015510 | 0.000049 | 0.008621 | 0.467772 | — | — | 2.517602 | 0.011917 | 0.992169 | — |
| test | GRU_return | 0.016149 | 0.000098 | 0.010526 | 0.524339 | — | — | 2.517602 | 0.010116 | 1.006929 | — |
| validation | LSTM_direction | — | — | — | 0.734807 | 0.196466 | 0.562213 | 2.095695 | 0.011993 | — | 0.998806 |
| test | LSTM_direction | — | — | — | 0.638624 | 0.237070 | 0.486183 | 2.095695 | 0.010691 | — | 1.016495 |
| validation | GRU_direction | — | — | — | 0.733886 | 0.195722 | 0.571885 | 1.805829 | 0.012152 | — | 0.995023 |
| test | GRU_direction | — | — | — | 0.638624 | 0.234827 | 0.486699 | 1.805829 | 0.011954 | — | 1.006880 |

### Epochs actually used

| model | seed | epochs_run | best_epoch | device | seconds |
|---|---|---|---|---|---|
| LSTM_return | 7 | 17 | 7 | cuda | 2.844906 |
| LSTM_return | 19 | 19 | 9 | cuda | 3.000421 |
| LSTM_return | 43 | 18 | 8 | cuda | 2.972218 |
| GRU_return | 7 | 21 | 11 | cuda | 3.386405 |
| GRU_return | 19 | 15 | 5 | cuda | 2.272334 |
| GRU_return | 43 | 12 | 2 | cuda | 1.894066 |
| LSTM_direction | 7 | 14 | 4 | cuda | 2.759919 |
| LSTM_direction | 19 | 12 | 2 | cuda | 1.764640 |
| LSTM_direction | 43 | 12 | 2 | cuda | 1.762524 |
| GRU_direction | 7 | 13 | 3 | cuda | 1.816449 |
| GRU_direction | 19 | 14 | 4 | cuda | 1.963395 |
| GRU_direction | 43 | 12 | 2 | cuda | 1.637641 |

### Strategy diagnostics

| model | net_return | max_drawdown | trades | status |
|---|---|---|---|---|
| zero_return | 0.000000 | 0.000000 | 0 | available |
| historical_mean | 0.000000 | 0.000000 | 0 | available |
| historical_probability | 0.000000 | 0.000000 | 0 | available |
| majority_direction | 0.000000 | 0.000000 | 0 | available |
| ARIMA | -0.291377 | -0.523450 | 273 | available |
| LSTM_return | -0.321783 | -0.383271 | 278 | available |
| GRU_return | -0.358019 | -0.403683 | 216 | available |
| LSTM_direction | -0.020535 | -0.020535 | 2 | available |
| GRU_direction | 0.000000 | 0.000000 | 0 | available |

### History and splits

![NAKL: History and splits](figures/NAKL/01_history.png)

### Training and validation curves

![NAKL: Training and validation curves](figures/NAKL/02_training.png)

### Forecasts and errors

![NAKL: Forecasts and errors](figures/NAKL/03_forecast.png)

### Confusion matrices

![NAKL: Confusion matrices](figures/NAKL/04_confusion.png)

### Probability diagnostics

![NAKL: Probability diagnostics](figures/NAKL/05_probability.png)

### Strategy and drawdown

![NAKL: Strategy and drawdown](figures/NAKL/06_strategy.png)


## CIL

3009 rows: 2010-01-04 to 2026-10-07. ARIMA order: (1, 0, 0). Input SHA-256: `a0bd275d0319af6f2fbea3722f8a697900c099d0f708af5c9fb78b4a59a4533a`.

| split | n | first_origin | last_origin | last_label |
|---|---|---|---|---|
| train | 2158 | 2010-03-02 00:00:00 | 2021-12-24 00:00:00 | 2021-12-30 00:00:00 |
| validation | 189 | 2022-01-07 00:00:00 | 2023-12-27 00:00:00 | 2023-12-28 00:00:00 |
| test | 620 | 2024-01-02 00:00:00 | 2026-10-06 00:00:00 | 2026-10-07 00:00:00 |

| phase | model | rmse | rmse_std | mae | accuracy | brier | auc | training_seconds | prediction_ms_per_sample | relative_rmse | relative_brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | zero_return | 0.036110 | — | 0.016890 | 0.650794 | — | — | — | — | 0.999960 | — |
| validation | historical_mean | 0.036111 | — | 0.016930 | 0.650794 | — | — | — | — | 1.000000 | — |
| validation | historical_probability | — | — | — | 0.650794 | 0.227361 | 0.500000 | — | — | — | 1.000000 |
| validation | majority_direction | — | — | — | 0.650794 | — | — | — | — | — | — |
| test | zero_return | 0.019535 | — | 0.011777 | 0.620968 | — | — | — | — | 0.999692 | — |
| test | historical_mean | 0.019541 | — | 0.011813 | 0.620968 | — | — | — | — | 1.000000 | — |
| test | historical_probability | — | — | — | 0.620968 | 0.236953 | 0.500000 | — | — | — | 1.000000 |
| test | majority_direction | — | — | — | 0.620968 | — | — | — | — | — | — |
| validation | ARIMA | 0.035822 | — | 0.017024 | 0.671958 | — | — | 0.759456 | 7.573105 | 0.992003 | — |
| test | ARIMA | 0.019139 | — | 0.011837 | 0.641935 | — | — | 0.759456 | 8.583388 | 0.979410 | — |
| validation | LSTM_return | 0.035768 | 0.000022 | 0.017521 | 0.516755 | — | — | 2.034081 | 0.011869 | 0.990492 | — |
| test | LSTM_return | 0.019040 | 0.000018 | 0.012169 | 0.566129 | — | — | 2.034081 | 0.010918 | 0.974370 | — |
| validation | GRU_return | 0.035685 | 0.000061 | 0.017411 | 0.564374 | — | — | 1.915969 | 0.010370 | 0.988201 | — |
| test | GRU_return | 0.019012 | 0.000021 | 0.012164 | 0.572043 | — | — | 1.915969 | 0.008750 | 0.972928 | — |
| validation | LSTM_direction | — | — | — | 0.626102 | 0.223176 | 0.597191 | 2.013849 | 0.009252 | — | 0.981590 |
| test | LSTM_direction | — | — | — | 0.636022 | 0.227711 | 0.609093 | 2.013849 | 0.008838 | — | 0.960997 |
| validation | GRU_direction | — | — | — | 0.654321 | 0.223322 | 0.577030 | 1.781251 | 0.009020 | — | 0.982235 |
| test | GRU_direction | — | — | — | 0.640323 | 0.226021 | 0.618243 | 1.781251 | 0.009832 | — | 0.953864 |

### Epochs actually used

| model | seed | epochs_run | best_epoch | device | seconds |
|---|---|---|---|---|---|
| LSTM_return | 7 | 17 | 7 | cuda | 2.064999 |
| LSTM_return | 19 | 17 | 7 | cuda | 2.018911 |
| LSTM_return | 43 | 16 | 6 | cuda | 2.018334 |
| GRU_return | 7 | 17 | 7 | cuda | 1.919224 |
| GRU_return | 19 | 18 | 8 | cuda | 1.999450 |
| GRU_return | 43 | 17 | 7 | cuda | 1.829233 |
| LSTM_direction | 7 | 17 | 7 | cuda | 1.869370 |
| LSTM_direction | 19 | 17 | 7 | cuda | 2.036310 |
| LSTM_direction | 43 | 19 | 9 | cuda | 2.135867 |
| GRU_direction | 7 | 19 | 9 | cuda | 2.038717 |
| GRU_direction | 19 | 15 | 5 | cuda | 1.573866 |
| GRU_direction | 43 | 16 | 6 | cuda | 1.731171 |

### Strategy diagnostics

| model | net_return | max_drawdown | trades | status |
|---|---|---|---|---|
| zero_return | 0.000000 | 0.000000 | 0 | available |
| historical_mean | 0.000000 | 0.000000 | 0 | available |
| historical_probability | 0.000000 | 0.000000 | 0 | available |
| majority_direction | 0.000000 | 0.000000 | 0 | available |
| ARIMA | 0.359253 | -0.173920 | 198 | available |
| LSTM_return | 0.787228 | -0.124825 | 287 | available |
| GRU_return | 1.229381 | -0.081070 | 269 | available |
| LSTM_direction | 0.202525 | -0.073954 | 28 | available |
| GRU_direction | 0.655609 | -0.114457 | 57 | available |

### History and splits

![CIL: History and splits](figures/CIL/01_history.png)

### Training and validation curves

![CIL: Training and validation curves](figures/CIL/02_training.png)

### Forecasts and errors

![CIL: Forecasts and errors](figures/CIL/03_forecast.png)

### Confusion matrices

![CIL: Confusion matrices](figures/CIL/04_confusion.png)

### Probability diagnostics

![CIL: Probability diagnostics](figures/CIL/05_probability.png)

### Strategy and drawdown

![CIL: Strategy and drawdown](figures/CIL/06_strategy.png)

## Files and next decision

This report and the figures are the only exported results. Displayed tables are saved in notebook outputs; full predictions exist only in the live kernel and require rerunning after restart. No deployment model or data snapshot is written. Review validation consistency across stocks before accepting a model family. This report does not establish that LSTM is best, nor guarantee that any learned model improves on simple baselines. Future monitoring and release gates remain separate work.

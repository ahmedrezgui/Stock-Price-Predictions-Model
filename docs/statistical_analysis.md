# Statistical diagnostics

Run `python -m stock_forecasting.cli analyze --data data/raw/BIAT.csv --output reports/analysis`.
The benchmark also writes `statistical_analysis.json` and training-only plots.
Only the training portion feeds the diagnostic tests used for model reasoning.

ADF tests close, log price and log return, using a constant and AIC-selected lags.
Reports contain statistic, p-value, effective observations, lag, critical values
and regression specification. The null is a unit root; rejection is evidence
against that null, not proof of strict stationarity or forecastability. KPSS tests
level-stationarity as a complementary null. Its tabulated p-values may be bounds;
statsmodels warnings are retained in the JSON.

ARCH-LM (five lags), Ljung–Box (5/10 lags) and squared-return Ljung–Box diagnose
conditional variance and serial dependence. Histograms, normal Q-Q, ACF/PACF,
squared-return ACF, trailing moments and six-MAD outlier counts expose heavy tails,
volatility clustering and instability. Outliers are reported, not silently
deleted. Rolling regime subsets are descriptive; formal structural-break tests
and point-in-time corporate-action reconciliation remain future work.

ARIMA AIC order selection uses training data only; stationarity tests are not
mandatory gates for recurrent models. Actual experiment conclusions are recorded
in [model comparison](model_comparison.md) and generated reports, not inferred
from the mere choice of architecture.

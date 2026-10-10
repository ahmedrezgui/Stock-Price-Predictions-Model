import warnings

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss


def statistical_analysis(frame):
    result = {
        "scope": "training observations only",
        "tests": {},
        "interpretation": "ADF null: unit root; rejection is not proof of full stationarity. KPSS null: level stationarity.",
    }
    for name, series in {
        "close": frame.close,
        "log_price": np.log(frame.close),
        "log_return": np.log(frame.close).diff().dropna(),
    }.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if len(series) < 30 or np.std(series) < 1e-12:
                result["tests"][name] = {"unavailable": "Short/constant series"}
                continue
            adf = adfuller(series, regression="c", autolag="AIC")
            kp = kpss(series, regression="c", nlags="auto")
            result["tests"][name] = {
                "adf": {
                    "statistic": float(adf[0]),
                    "p_value": float(adf[1]),
                    "lags": int(adf[2]),
                    "n": int(adf[3]),
                    "critical_values": adf[4],
                    "regression": "c",
                    "autolag": "AIC",
                },
                "kpss": {
                    "statistic": float(kp[0]),
                    "p_value": float(kp[1]),
                    "lags": int(kp[2]),
                    "critical_values": kp[3],
                },
                "warnings": [str(w.message) for w in caught],
            }
    returns = np.log(frame.close).diff().dropna()
    if len(returns) > 30 and np.std(returns) > 1e-12:
        arch = het_arch(returns, nlags=5)
        result["arch_lm"] = {"statistic": float(arch[0]), "p_value": float(arch[1]), "lags": 5}
        result["ljung_box"] = acorr_ljungbox(returns, lags=[5, 10], return_df=True).to_dict(orient="index")
        result["squared_return_ljung_box"] = acorr_ljungbox(returns**2, lags=[5, 10], return_df=True).to_dict(
            orient="index"
        )
        median = returns.median()
        mad = np.median(abs(returns - median))
        result["outliers"] = {
            "mad_threshold": 6,
            "count": int((abs(returns - median) > 6 * max(mad, 1e-12)).sum()),
            "max_absolute_log_return": float(abs(returns).max()),
            "removed": False,
        }
    return result

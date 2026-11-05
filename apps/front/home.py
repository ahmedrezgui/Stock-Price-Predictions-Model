import os
import sys
import math
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from requests_cache import CachedSession
from requests.exceptions import RequestException

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trading_charts import build_chart
from ui import apply_theme

load_dotenv()
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8002")
cache_session = CachedSession("dashboard-cache-v2", expire_after=int(os.getenv("FRONTEND_CACHE_EXPIRATION_TIME", "60")))
st.set_page_config(page_title="StockWise | Market terminal", page_icon="S", layout="wide", initial_sidebar_state="collapsed")
apply_theme()

brand, home, terminal, assistant = st.columns([5, 1.3, 1.5, 1.5])
with brand:
    st.markdown('<div class="brand">Stock<span>Wise</span> <span style="font-size:10px;color:#7f91aa;letter-spacing:2px;margin-left:12px">MARKET INTELLIGENCE</span></div>', unsafe_allow_html=True)
page = st.query_params.get("page", "Home")
for column, name, label in [(home, "Home", "Overview"), (terminal, "Data & Predict", "Market terminal")]:
    with column:
        if st.button(label, key=f"nav-{name}", use_container_width=True, type="primary" if page == name else "secondary"):
            st.query_params["page"] = name
            st.rerun()
with assistant:
    st.page_link("pages/chatbot.py", label="AI assistant", use_container_width=True)


def fetch(path):
    response = cache_session.get(f"{BACKEND_URL}{path}", timeout=75)
    response.raise_for_status()
    return response.json()


if page != "Data & Predict":
    st.markdown('''<div class="hero"><div class="eyebrow">Your market. A clearer perspective.</div>
    <h1>Read the market.<br>Understand the forecast.</h1>
    <p>Explore stock history, compare technical indicators, and put deployed LSTM predictions in context. One workspace for your research.</p>
    <span class="badge">DAILY DATA &nbsp; / &nbsp; AZURE-HOSTED FORECASTS</span></div>''', unsafe_allow_html=True)
    if st.button("Open market terminal", type="primary"):
        st.query_params["page"] = "Data & Predict"
        st.rerun()
    a, b, c = st.columns(3)
    for column, title, description in [(a, "01 / Explore", "Candles, volume and price history in an interactive workspace."), (b, "02 / Compare", "Moving averages, Bollinger Bands and momentum indicators."), (c, "03 / Understand", "Model predictions with their dates, horizon and limitations.")]:
        with column, st.container(border=True):
            st.subheader(title)
            st.write(description)
    st.caption("Research tools built around historical observations. Prices are not a live market feed.")
else:
    st.markdown('<div class="eyebrow">Research workspace</div>', unsafe_allow_html=True)
    title, refresh = st.columns([8, 1])
    title.title("Market terminal")
    if refresh.button("Refresh", use_container_width=True):
        cache_session.cache.clear()
        st.rerun()
    try:
        companies = fetch("/companies")
        if not isinstance(companies, list):
            raise ValueError("Invalid catalog")
    except (RequestException, ValueError):
        st.error("The deployed model catalog is temporarily unavailable. Use Refresh to retry.")
        st.stop()
    if not companies:
        st.info("No stocks with deployed LSTM forecasts and historical data are currently available.")
        st.stop()
    selector, context = st.columns([2, 5])
    company = selector.selectbox("Select a company", companies)
    context.caption(f"{len(companies)} stocks with deployed models  /  Daily observations  /  Historical snapshot")
    try:
        payload = fetch(f"/stock/{company}")
        df = pd.DataFrame(payload["data"])
        df["date"] = pd.to_datetime(df.date)
        df = df.sort_values("date")
        if df.empty:
            raise ValueError("Empty history")
    except (RequestException, ValueError, KeyError):
        st.error(f"Data for {company} is currently unavailable. Please refresh or select another stock.")
        st.stop()
    latest = df.iloc[-1]
    change = (latest.cloture / df.iloc[-2].cloture - 1) * 100 if len(df) > 1 else None
    recent = df[df.date >= df.date.max() - pd.DateOffset(years=1)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{company} / Last close", f"{latest.cloture:,.3f}", f"{change:+.2f}% vs previous bar" if change is not None else None)
    vol = latest.get("volume")
    m2.metric("Last volume", f"{vol:,.0f}" if pd.notna(vol) else "Unavailable")
    m3.metric("1Y closing range", f"{recent.cloture.min():,.2f} - {recent.cloture.max():,.2f}")
    m4.metric("Data as of", latest.date.strftime("%d %b %Y"))
    st.caption("Source price units. Historical snapshot, not live quotes. Date ranges are anchored to the latest available bar.")
    with st.container(border=True):
        st.subheader(f"{company} ? Price action")
        c1, c2, c3, c4 = st.columns([2, 1.2, 1.2, 1.2])
        period = c1.select_slider("Chart range", options=["1M", "3M", "6M", "1Y", "5Y", "All"], value="3M")
        has_candles = {"ouverture", "haut", "bas"}.issubset(df) and df[["ouverture", "haut", "bas"]].notna().all(axis=1).any()
        style = c2.selectbox("Price style", ["Candles", "Line"] if has_candles else ["Line"])
        interval = c4.selectbox("Bar interval", ["Day", "Month", "Year"], help="Month and Year combine OHLC and volume into one bar per calendar period. Edge periods can be partial.")
        momentum = c3.selectbox("Momentum", ["RSI", "MACD", "None"])
        with st.expander("Indicators & overlays", expanded=False):
            i1, i2 = st.columns(2)
            sma = i1.multiselect("Simple moving averages", [5, 10, 20, 50, 100, 200], default=[20, 50])
            ema = i2.multiselect("Exponential moving averages", [5, 10, 20, 50, 100, 200], default=[])
            bands = i1.checkbox("Bollinger Bands (20, 2)")
            volume = i2.checkbox("Volume bars", value=True)
            st.caption("Windows count bars at the selected interval (20 monthly bars = 20 months). Edge periods may be incomplete. RSI: Wilder 14. MACD: 12/26/9. Indicators use full history before the selected view is cropped.")
        st.plotly_chart(build_chart(df, company, style, sma, ema, bands, momentum, period, volume, interval), use_container_width=True, theme=None, config={"displaylogo": False, "scrollZoom": True}, key=f"chart-{company}")
        if interval != "Day":
            st.caption(f"Each candle summarizes one {interval.lower()}: first open, highest high, lowest low and last close; volume is summed. Edge periods may be partial. Forecasts still use the deployed model's original horizon.")
        quality = payload.get("quality", {})
        if quality.get("invalid_ohlc_rows") or quality.get("missing_volume_rows"):
            st.caption(f"Data quality: {quality.get('invalid_ohlc_rows', 0)} bars have incomplete/inconsistent OHLC and are omitted from candles; {quality.get('missing_volume_rows', 0)} volume values are unavailable. Closing prices remain visible in line mode.")
    with st.container(border=True):
        st.markdown('<div class="section-label">Model outlook</div><span class="badge">DEPLOYED LSTM</span>', unsafe_allow_html=True)
        summary, details = st.columns([1, 2], gap="large")
        forecast = payload.get("forecast")
        if forecast:
            with summary:
                implied = forecast.get("implied_price")
                log_return = forecast.get("predicted_log_return")
                st.markdown("#### Implied next price")
                st.markdown(f'<div class="forecast-price">{implied:,.3f}</div>' if implied is not None else '<div class="forecast-price">Unavailable</div>', unsafe_allow_html=True)
                if log_return is not None:
                    pct = math.expm1(log_return) * 100
                    color = "forecast-up" if pct >= 0 else "forecast-down"
                    st.markdown(f'<p class="{color}">{pct:+.3f}% predicted return</p>', unsafe_allow_html=True)
            with details:
                st.write("**Forecast as of**", forecast.get("as_of"))
                st.write("**Horizon**", f"{forecast.get('forecast_horizon_observations', 1)} observed bar(s)")
                if forecast.get("forecast_for"):
                    st.write("**Target date**", forecast["forecast_for"])
                st.caption("One model estimate, not a future price path. News is not an input to this LSTM forecast.")
                st.caption("Price basis: " + str(forecast.get("price_basis", "unverified")) + ".")
                if forecast.get("horizon_basis") != "exchange_sessions":
                    st.caption("The forecast horizon is based on observed bars, not verified exchange sessions.")
                with st.expander("Model details"):
                    st.write({key: forecast.get(key) for key in ["model_version", "deployment_status", "horizon_basis", "predicted_log_return", "prediction_interval", "probability_up"]})
        else:
            st.warning("The deployed model's forecast is temporarily unavailable.")
            st.caption(payload.get("forecast_error", "Try Refresh shortly."))
        st.divider()
        st.page_link("pages/chatbot.py", label="Discuss with AI assistant", use_container_width=True)
        st.caption("Historical indicators and model estimates are research inputs, not guarantees of future performance.")
    with st.expander("Historical data / export"):
        st.dataframe(df.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{company}-history.csv", mime="text/csv")

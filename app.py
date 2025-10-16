# breakout_analyzer_final.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from math import floor, sqrt
from scipy.stats import norm, t

st.set_page_config(page_title="Backtest Data Analyzer", layout="wide")
st.title("📊 Backtest Data Analyzer — v6")


# ---------------------------
# Helper functions
# ---------------------------
def truncate_two_decimals(value: float) -> float:
    """Truncate positive float to 2 decimals (no rounding)."""
    if np.isnan(value) or value is None:
        return 0.0
    return floor(float(value) * 100) / 100.0


def parse_hour_col(df, time_col):
    """Try common formats to get hour (0-23). Accepts numeric hour or 'HH:MM' strings."""
    if time_col not in df.columns:
        return pd.Series([np.nan]*len(df))
    s = df[time_col]
    # If numeric already 0-23
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return s.astype('Int64')

    # try parsing "HH:MM" or "H:MM" or "HHMM"
    def parse_one(x):
        try:
            if pd.isna(x):
                return np.nan
            xs = str(x).strip()
            if xs.isdigit() and (len(xs) in (1, 2, 3, 4)):
                # e.g., '1300' or '9' or '09'
                if len(xs) <= 2:
                    return int(xs)
                if len(xs) == 3:  # e.g., '900' -> 9
                    return int(xs[:-2])
                return int(xs[:-2])
            if ":" in xs:
                parts = xs.split(":")
                return int(parts[0])  # hour portion
            # fallback: try pd.to_datetime
            return pd.to_datetime(xs, errors='coerce').hour
        except Exception:
            return np.nan
    return s.astype(str).apply(parse_one).astype('Int64')


def wilson_ci(k, n, conf=0.95):
    """Wilson score interval for a proportion. Returns (low, high) as decimals."""
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - (1 - conf) / 2)
    phat = k / n
    denom = 1 + z*z / n
    centre = phat + (z*z) / (2*n)
    margin = z * sqrt((phat*(1-phat)/n) + (z*z)/(4*n*n))
    low = max(0.0, (centre - margin) / denom)
    high = min(1.0, (centre + margin) / denom)
    return low, high


def required_sample_size(p_est=None, margin_error=0.05, conf=0.95):
    """n = Z^2 * p(1-p) / E^2 ; if p_est is None or 0/1 use 0.5 conservative"""
    z = norm.ppf(1 - (1 - conf) / 2)
    p = 0.5 if (p_est is None or p_est <= 0 or p_est >= 1) else p_est
    n = (z**2) * p * (1 - p) / (margin_error**2)
    return int(np.ceil(n))


def expectancy_ci(series: pd.Series, conf=0.95):
    """Mean CI using t-distribution (returns (low, high))."""
    series = series.dropna()
    n = len(series)
    if n == 0:
        return 0.0, 0.0
    mean = series.mean()
    if n == 1:
        return mean, mean
    sem = series.std(ddof=1) / sqrt(n)
    tval = t.ppf(1 - (1 - conf)/2, df=n-1)
    return mean - tval * sem, mean + tval * sem


# ---------------------------
# Sidebar: Upload CSV
# ---------------------------
st.sidebar.header("⬆️ Upload CSV")
uploaded = st.sidebar.file_uploader("Upload backtest CSV (required columns: Date, Pips/result_pips or pips)", type=["csv", "txt"])

# defaults for TP/SL (also used in Simulation inputs but also shown here for quick override)
st.sidebar.header("⚙️ Settings and Filters")
default_tp = st.sidebar.number_input("TP (pips)", value=400, step=50)
default_sl = st.sidebar.number_input("SL (pips)", value=2000, step=100)

# ---------------------------
# Load data
# ---------------------------
if uploaded is None:
    st.info("Upload a backtest CSV to begin. Expected minimum columns: Date, Pips (signed).")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Accept either 'pips' or 'result_pips' or 'ResultPips' etc.
pips_col_candidates = [c for c in df.columns if c.lower() in ("pips", "result_pips", "resultpips", "result")]
if not pips_col_candidates:
    st.error("CSV must contain a 'pips' (signed) column or 'result_pips'. Rename your column and re-upload.")
    st.stop()
pips_col = pips_col_candidates[0]

# Ensure date/time columns exist
date_col_candidates = [c for c in df.columns if c.lower() in ("date", "datetime", "trade_date", "entry date", "entry_date")]
if not date_col_candidates:
    st.error("CSV must contain a 'date' column (e.g., YYYY-MM-DD).")
    st.stop()
date_col = date_col_candidates[0]

time_col_candidates = [c for c in df.columns if c.lower() in ("time", "hour", "trade_time")]
time_col = time_col_candidates[0] if time_col_candidates else None

# --- Duration normalization ---
duration_col_candidates = [c for c in df.columns if c.lower() in ("duration", "trade_duration", "hours_duration")]
if duration_col_candidates:
    duration_col = duration_col_candidates[0]
    df["duration"] = pd.to_numeric(df[duration_col], errors="coerce")
else:
    df["duration"] = np.nan  # fill with NaN if not found

# Preprocess columns
df = df.copy()
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
if df[date_col].isna().any():
    st.warning("Some date values could not be parsed. Rows with invalid dates will be ignored for time grouping.")

df['year'] = df[date_col].dt.year
df['month'] = df[date_col].dt.month_name()
df['day_of_week'] = df[date_col].dt.day_name()
if time_col:
    df['hour'] = parse_hour_col(df, time_col)
else:
    # if no time column, try using hour from date_col if datetime includes hour
    df['hour'] = df[date_col].dt.hour.astype('Int64')

# pips numeric
df['pips'] = pd.to_numeric(df[pips_col], errors='coerce')
df = df.dropna(subset=['pips', 'year'])  # drop invalids

# derive win indicator
df['win'] = (df['pips'] > 0).astype(int)

# ---------------------------
# Sidebar: Settings & Filters
# ---------------------------
# Set defaults
unique_years = sorted(df['year'].dropna().unique())
unique_months = sorted(df['month'].dropna().unique(), key=lambda m: pd.to_datetime(m, format="%B").month)
unique_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

selected_years = st.sidebar.multiselect("Select Years", options=unique_years, default=unique_years, key="sel_years")
selected_months = st.sidebar.multiselect("Select Months", options=unique_months, default=unique_months, key="sel_months")
selected_days = st.sidebar.multiselect("Select Days", options=unique_days, default=unique_days, key="sel_days")
hour_range = st.sidebar.slider("Hour range (inclusive)", 0, 23, (0, 23), key="sel_hours")

# Apply filters
filtered = df[
    df['year'].isin(selected_years) &
    df['month'].isin(selected_months) &
    df['day_of_week'].isin(selected_days) &
    (df['hour'] >= hour_range[0]) & (df['hour'] <= hour_range[1])
].copy()

# Save the latest filtered snapshot for simulation import
st.session_state['main_filtered_df'] = filtered.copy()

# ---------------------------
# Tabs: Main / Confidence / Simulation
# ---------------------------
tab_main, tab_conf, tab_sim = st.tabs(["📊 Main (Performance)", "📈 Confidence & Stats", "💰 Simulation"])

# ---------------------------
# MAIN TAB
# ---------------------------
with tab_main:
    # Top-level metrics
    total_trades = len(filtered)
    total_wins = int(filtered['win'].sum())
    total_losses = total_trades - total_wins
    win_rate = (total_wins / total_trades) if total_trades else 0.0
    avg_win = filtered.loc[filtered['win'] == 1, 'pips'].mean() if total_wins else 0.0
    avg_loss = abs(filtered.loc[filtered['win'] == 0, 'pips'].mean()) if total_losses else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) if total_trades else 0.0

    # Profit factor = total profit / abs(total loss)
    total_profit = filtered.loc[filtered['pips'] > 0, 'pips'].sum()
    total_loss = filtered.loc[filtered['pips'] < 0, 'pips'].sum()  # negative
    profit_factor = (total_profit / abs(total_loss)) if total_loss != 0 else np.nan

    # average trades per month
    if total_trades:
        months_span = filtered[date_col].dt.to_period('M').nunique()
        avg_trades_per_month = total_trades / months_span if months_span > 0 else total_trades
    else:
        avg_trades_per_month = 0.0

    # Display metrics
    st.subheader("Summary Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades", total_trades)
    c2.metric("Win Rate (%)", f"{win_rate*100:.2f}")
    c3.metric("Expectancy (pips/trade)", f"{expectancy:.2f}")
    c4.metric("Profit Factor", f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A")
    c5.metric("Avg Trades / Month", f"{avg_trades_per_month:.1f}")

    st.markdown("---")
    # Equity curve
    st.subheader("Equity Curve")
    if total_trades:
        eq = filtered.sort_values(by=date_col).copy()
        eq['equity'] = eq['pips'].cumsum()
        fig_eq = px.line(eq, x=date_col, y='equity', title="Cumulative P/L (pips)")
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("No trades in filtered selection.")

    st.markdown("---")
    # Monthly win rate chart (as %)
    if total_trades:
        monthly = filtered.groupby(filtered[date_col].dt.to_period("M"))['win'].mean().reset_index()
        monthly['period'] = monthly[date_col].astype(str)
        monthly['win_pct'] = monthly['win']*100
        fig_month = px.bar(monthly, x='period', y='win_pct', title="Monthly Win Rate (%)")
        fig_month.update_layout(xaxis_title="Month", yaxis_title="Win Rate (%)")
        st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("---")
    st.subheader("Tables")

    # Hour table (trades & win rate)
    st.write("#### By Hour (all years combined)")
    hour_group = filtered.groupby('hour').agg(trades=('win', 'count'), wins=('win', 'sum')).reset_index().sort_values(
        'hour')
    if hour_group.empty:
        st.write("No hourly data.")
    else:
        hour_group['win_rate_%'] = hour_group['wins'] / hour_group['trades'] * 100
        st.dataframe(hour_group[['hour', 'trades', 'win_rate_%']].style.format({'win_rate_%': '{:.2f}'}),
                     use_container_width=True)

    # By Day table
    st.write("#### By Day (all years combined)")
    day_group = filtered.groupby('day_of_week').agg(trades=('win', 'count'), wins=('win', 'sum')).reset_index()
    if not day_group.empty:
        # sort by weekday order
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_group['order'] = day_group['day_of_week'].apply(lambda x: weekday_order.index(x) if x in weekday_order else 99)
        day_group = day_group.sort_values('order').drop(columns='order')
        day_group['win_rate_%'] = day_group['wins'] / day_group['trades'] * 100
        st.dataframe(day_group[['day_of_week', 'trades', 'win_rate_%']].rename(columns={'day_of_week': 'Day'}).style.format({'win_rate_%': '{:.2f}'}), use_container_width=True)
    else:
        st.write("No day data.")

    # By Week table (week-of-month 1-5)
    st.write("#### By Week (Week 1–5 across months)")

    # compute week-of-month for each row
    def week_of_month(d):
        return int(((d.day - 1) // 7) + 1)
    filtered['week_of_month'] = filtered[date_col].dt.day.apply(lambda x: ((x - 1) // 7) + 1)
    week_group = filtered.groupby('week_of_month').agg(trades=('win', 'count'), wins=('win', 'sum')).reset_index().sort_values('week_of_month')
    if not week_group.empty:
        week_group['win_rate_%'] = week_group['wins'] / week_group['trades'] * 100
        week_group = week_group.rename(columns={'week_of_month': 'WeekOfMonth'})
        st.dataframe(week_group[['WeekOfMonth', 'trades', 'win_rate_%']].style.format({'win_rate_%': '{:.2f}'}), use_container_width=True)
    else:
        st.write("No week data.")

    st.markdown("---")
    st.write("#### Year → Day")
    years = sorted(filtered['year'].dropna().unique())
    if not years:
        st.write("No yearly data.")
    else:
        for y in years:
            with st.expander(f"{y}", expanded=False):
                yf = filtered[filtered['year'] == y]
                if yf.empty:
                    st.write("No trades this year.")
                    continue
                day_group = yf.groupby('day_of_week').agg(trades=('win', 'count'), wins=('win', 'sum')).reset_index()
                # sort by weekday order
                weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                day_group['order'] = day_group['day_of_week'].apply(lambda x: weekday_order.index(x) if x in weekday_order else 99)
                day_group = day_group.sort_values('order').drop(columns='order')
                day_group['win_rate_%'] = day_group['wins'] / day_group['trades'] * 100
                st.dataframe(day_group[['day_of_week', 'trades', 'win_rate_%']].rename(columns={'day_of_week': 'Day'}).style.format({'win_rate_%': '{:.2f}'}), use_container_width=True)

    st.markdown("---")
    st.write("#### Year → Hour")
    if not years:
        st.write("No yearly data.")
    else:
        for y in years:
            with st.expander(f"{y}", expanded=False):
                yf = filtered[filtered['year'] == y]
                if yf.empty:
                    st.write("No trades this year.")
                else:
                    yg = yf.groupby('hour').agg(trades=('win', 'count'), wins=('win', 'sum')).reset_index().sort_values('hour')
                    yg['win_rate_%'] = yg['wins'] / yg['trades'] * 100
                    st.dataframe(yg[['hour', 'trades', 'win_rate_%']].rename(columns={'hour': 'Hour'}).style.format({'win_rate_%': '{:.2f}'}), use_container_width=True)

    st.markdown("---")
    st.write("#### Year → Week")
    if not years:
        st.write("No yearly data.")
    else:
        for y in years:
            with st.expander(f"{y}", expanded=False):
                yf = filtered[filtered['year'] == y]
                if yf.empty:
                    st.write("No trades this year.")
                else:
                    yf['week_of_month'] = yf[date_col].dt.day.apply(lambda x: ((x - 1) // 7) + 1)
                    yg = yf.groupby('week_of_month').agg(trades=('win', 'count'), wins=('win', 'sum')).reset_index().sort_values('week_of_month')
                    yg['win_rate_%'] = yg['wins'] / yg['trades'] * 100
                    yg = yg.rename(columns={'week_of_month': 'WeekOfMonth'})
                    st.dataframe(yg[['WeekOfMonth', 'trades', 'win_rate_%']].style.format({'win_rate_%': '{:.2f}'}), use_container_width=True)

# ---------------------------
# CONFIDENCE TAB (unchanged)
# ---------------------------
with tab_conf:
    st.subheader("Confidence Intervals & Sample Size Check")

    # Confidence fixed at 95%
    CONFIDENCE = 0.95
    st.markdown(f"Confidence level is fixed at **{int(CONFIDENCE * 100)}%**")

    # margin of error for sample size (used in Confidence tab)
    margin_error = st.slider("Margin of error for sample-size (±%)", min_value=1, max_value=10, value=5) / 100.0

    # Win rate CI (Wilson)
    wins = total_wins
    n = total_trades
    win_low, win_high = wilson_ci(wins, n, CONFIDENCE)

    # Expectancy CI
    exp_low, exp_high = expectancy_ci(filtered['pips'], conf=CONFIDENCE)

    req_n = required_sample_size(win_rate if n > 0 else None, margin_error, CONFIDENCE)
    sufficient = n >= req_n

    colA, colB = st.columns([3, 1])
    with colA:
        st.write(f"- Observed win rate: **{win_rate*100:.2f}%** ({wins}/{n})")
        st.write(f"- 95% Wilson CI for win rate: **[{win_low*100:.2f}%, {win_high*100:.2f}%]**")
        st.write(f"- Expectancy (mean pips/trade): **{expectancy:.2f}**")
        st.write(f"- 95% CI for expectancy: **[{exp_low:.2f}, {exp_high:.2f}] pips**")
        st.write(f"- Required sample size for ±{margin_error*100:.0f}% margin: **{req_n} trades**")
        st.write(f"- Actual filtered sample size: **{n} trades**")
    with colB:
        if sufficient:
            st.success(f"Sample requirement satisfied ✅ (actual {n} ≥ required {req_n})")
        else:
            st.warning(f"Sample requirement NOT met ⚠️ (actual {n} < required {req_n})")

    st.markdown("---")
    st.subheader("Per-group Win Rate CI & Sample Check (Hours, Days, Months)")

    def build_group_table(groupby_col, order_map=None):
        g = filtered.groupby(groupby_col).agg(trades=('win', 'count'), wins=('win', 'sum')).reset_index()
        if g.empty:
            return g
        g['win_rate'] = g['wins'] / g['trades']
        g[['ci_low', 'ci_high']] = g.apply(lambda r: pd.Series(wilson_ci(int(r['wins']), int(r['trades']), CONFIDENCE)), axis=1)
        g['required_n'] = g['win_rate'].apply(lambda p: required_sample_size(p, margin_error, CONFIDENCE))
        g['sufficient'] = g['trades'] >= g['required_n']
        g['win_rate_%'] = g['win_rate'] * 100
        g['ci_low_%'] = g['ci_low'] * 100
        g['ci_high_%'] = g['ci_high'] * 100
        if order_map:
            g['order'] = g[groupby_col].map(order_map).fillna(999)
            g = g.sort_values('order').drop(columns='order')
        else:
            g = g.sort_values('trades', ascending=False)
        return g[[groupby_col, 'trades', 'win_rate_%', 'ci_low_%', 'ci_high_%', 'required_n', 'sufficient']]

    st.write("By Hour")
    ht = build_group_table('hour')
    if ht.empty:
        st.write("No hourly groups.")
    else:
        st.dataframe(ht.style.format({'win_rate_%': '{:.2f}', 'ci_low_%': '{:.2f}', 'ci_high_%': '{:.2f}'}), use_container_width=True)

    st.write("By Day")
    day_order_map = {d: i for i, d in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])}
    dt = build_group_table('day_of_week', order_map=day_order_map)
    if dt.empty:
        st.write("No day groups.")
    else:
        st.dataframe(dt.style.format({'win_rate_%': '{:.2f}', 'ci_low_%': '{:.2f}', 'ci_high_%': '{:.2f}'}), use_container_width=True)

    st.write("By Month")
    month_map = {m: i for i, m in enumerate(pd.date_range("2000-01-01", periods=12, freq="MS").strftime("%B"), start=1)}
    mt = build_group_table('month', order_map=month_map)
    if mt.empty:
        st.write("No month groups.")
    else:
        st.dataframe(mt.style.format({'win_rate_%': '{:.2f}', 'ci_low_%': '{:.2f}', 'ci_high_%': '{:.2f}'}), use_container_width=True)

# ---------------------------
# SIMULATION TAB
# ---------------------------
with tab_sim:
    st.subheader("Simulation (import snapshot from Main Tab)")

    sim_col1, sim_col2 = st.columns(2)
    starting_balance_input = sim_col1.number_input("Starting Balance ($)", value=200.00, format="%.2f")
    lot_divider = sim_col2.number_input("Lot Divider (balance / divider = lot)", value=5000.0, step=1000.0)

    # --- helper callback to enforce exclusivity ---
    def toggle_mode(selected_mode):
        # if the user unchecked it, turn all off
        if not st.session_state[selected_mode]:
            for mode in ["fixed_lot", "recalc_monthly", "recalc_on_1k"]:
                st.session_state[mode] = False
        else:
            # otherwise, set only the selected one True
            for mode in ["fixed_lot", "recalc_monthly", "recalc_on_1k"]:
                st.session_state[mode] = (mode == selected_mode)

    # --- UI (Simulation options) ---
    st.markdown("Lot Size Options")

    lot_col1, lot_col2, lot_col3 = st.columns(3)
    fixed_lot_checkbox = lot_col1.checkbox(
        "Use Fixed Lot Size (disable divider)",
        key="fixed_lot",
        on_change=toggle_mode,
        args=("fixed_lot",)
    )

    recalc_monthly_checkbox = lot_col2.checkbox(
        "Recalculate Lot Size Monthly",
        key="recalc_monthly",
        on_change=toggle_mode,
        args=("recalc_monthly",)
    )

    recalc_on_1k_checkbox = lot_col3.checkbox(
        "Recalculate Lot Size every $1K",
        key="recalc_on_1k",
        on_change=toggle_mode,
        args=("recalc_on_1k",)
    )

    # If fixed, allow user to input fixed lot size (2 decimals, truncated)
    fixed_lot_value = None
    if fixed_lot_checkbox:
        fixed_lot_value = st.number_input("Fixed Lot Size (e.g., 0.04)", value=truncate_two_decimals(starting_balance_input / lot_divider), format="%.2f")

    tp_col1, tp_col2 = st.columns(2)
    # TP / SL inputs for simulation (defaults from sidebar)
    tp_sim = tp_col1.number_input("TP for simulation (pips)", value=int(default_tp), step=50)
    sl_sim = tp_col2.number_input("SL for simulation (pips)", value=int(default_sl), step=100)

    st.markdown("⤵️ Click to import the currently filtered trades into the simulator snapshot.")
    import_btn = st.button("📥 Import filtered trades from Main Tab (snapshot)")

    if import_btn:
        # Copy filtered snapshot into session state for simulation
        st.session_state['sim_source'] = st.session_state.get('main_filtered_df', pd.DataFrame()).copy()
        st.success("Imported filtered trades snapshot for simulation. (Simulation will not update until you re-import.)")

    if 'sim_source' not in st.session_state or st.session_state['sim_source'].empty:
        st.info("No simulation snapshot loaded. Use 'Import filtered trades' to load the current filtered dataset.")
    else:
        sim_df = st.session_state['sim_source'].copy()
        sim_df[date_col] = pd.to_datetime(sim_df[date_col], errors='coerce')
        sim_df = sim_df.sort_values(by=date_col, ascending=True).reset_index(drop=True)

        # ensure columns
        if 'pips' not in sim_df.columns:
            st.error("Simulation snapshot missing 'pips' column.")
        else:
            # prepare columns
            sim_df['entry_date'] = sim_df[date_col]
            sim_df['start_balance'] = 0.0
            sim_df['lot_size'] = 0.0
            sim_df['profit_$'] = 0.0
            sim_df['end_balance'] = 0.0

            bal = starting_balance_input
            # determine initial fixed lot (if selected)
            if st.session_state.get("fixed_lot"):
                # truncate fixed lot to 2 decimals (no rounding)
                fixed_lot = truncate_two_decimals(float(fixed_lot_value))
            else:
                fixed_lot = None

            # tracking for new options
            current_month = None
            next_milestone = (int(bal // 1000) + 1) * 1000  # next $1K target
            lot = truncate_two_decimals(bal / lot_divider)  # initial dynamic lot

            # iterate trades (compounding: end_balance becomes next start_balance)
            for i, row in sim_df.iterrows():
                entry_date = pd.to_datetime(row['date'], errors='coerce')
                sim_df.at[i, 'start_balance'] = bal

                if fixed_lot is not None:
                    # Option 1: Fixed lot, never changes
                    lot = fixed_lot

                elif st.session_state.get("recalc_monthly"):
                    # Option 2: Recalculate lot at start of each month
                    month = entry_date.month
                    if month != current_month:
                        lot = truncate_two_decimals(bal / lot_divider)
                        current_month = month

                elif st.session_state.get("recalc_on_1k"):
                    # Option 3: Recalculate lot when hitting next $1K balance milestone
                    if bal >= next_milestone:
                        lot = truncate_two_decimals(bal / lot_divider)
                        next_milestone = (int(bal // 1000) + 1) * 1000

                else:
                    # Default: recalc every trade
                    raw_lot = bal / lot_divider
                    lot = truncate_two_decimals(raw_lot)

                sim_df.at[i, 'lot_size'] = lot

                # --- 🔥 check for burnout conditions ---
                if bal <= 0 or lot == 0:
                    sim_df.at[i, 'profit_$'] = 0
                    sim_df.at[i, 'end_balance'] = 0
                    sim_df.at[i, 'burned'] = True
                    # fill remaining rows (optional) to mark simulation stop
                    sim_df.loc[i + 1:, ['profit_$', 'end_balance', 'lot_size', 'burned']] = [0, 0, 0, True]
                    break  # stop simulation loop
                else:
                    sim_df.at[i, 'burned'] = False

                # === PROFIT CALCULATION ===
                # profit in $: pips * lot (pips is signed)
                pip_value = row['pips']
                # but simulation spec: if win -> TP pips * lot, if loss -> -SL pips * lot
                # we will determine win from pip sign, but use TP/SL constants
                if pip_value > 0:
                    profit_usd = tp_sim * lot
                else:
                    profit_usd = -sl_sim * lot

                sim_df.at[i, 'profit_$'] = profit_usd
                sim_df.at[i, 'end_balance'] = bal + profit_usd

                # next starting balance is ending balance (always compounding)
                bal = bal + profit_usd

            # Trim the dataframe up to the last executed trade (before burn)
            if 'burned' in sim_df.columns and sim_df['burned'].any():
                burn_index = sim_df.index[sim_df['burned']].min()
                sim_result_df = sim_df.loc[:burn_index].copy()
            else:
                sim_result_df = sim_df.copy()

            # Calculate final balance
            final_balance = sim_result_df['end_balance'].iloc[-1] if not sim_result_df.empty else 0

            # Calculate total gain percentage
            if starting_balance_input > 0:
                total_gain_pct = ((final_balance - starting_balance_input) / starting_balance_input) * 100
            else:
                total_gain_pct = 0

            # Calculate final total pips
            total_pips = sim_result_df['pips'].sum() if 'pips' in sim_result_df.columns else 0

            # Win / Loss tracking
            wins = sim_result_df[sim_result_df['pips'] > 0]
            losses = sim_result_df[sim_result_df['pips'] < 0]
            total_trades = len(sim_result_df) - 1
            win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

            # --- 📊 Average trades per month ---
            if total_trades > 0:
                total_trades = total_trades
                if 'date' in sim_result_df.columns:
                    date_col = 'date'
                else:
                    date_col = None

                if date_col:
                    months_span = sim_result_df[date_col].dt.to_period('M').nunique()
                    avg_trades_per_month = total_trades / months_span if months_span > 0 else total_trades
                else:
                    avg_trades_per_month = 0.0
            else:
                avg_trades_per_month = 0.0

            # Max drawdown (peak to trough)
            if not sim_result_df.empty:
                balance = sim_result_df['end_balance']
                rolling_max = balance.cummax()
                drawdown = (balance - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100  # in %
            else:
                max_drawdown = 0

            # --- 🕒 Average Trade Duration (in hours) ---
            avg_trade_duration_hours = 0.0  # Set a default value
            if "duration" in sim_result_df.columns:
                # .mean() ignores NaN values by default, but returns NaN if all values are NaN
                mean_val = sim_result_df["duration"].mean()
                # Check if the calculated mean is a valid number before assigning it
                if pd.notna(mean_val):
                    avg_trade_duration_hours = mean_val

            # Win/loss streaks
            def get_streaks(series):
                streaks = []
                current = 0
                last_sign = None
                for value in series:
                    sign = np.sign(value)
                    if sign == 0:
                        continue
                    if sign == last_sign:
                        current += 1
                    else:
                        current = 1
                        last_sign = sign
                    streaks.append((sign, current))
                win_streaks = [s for sign, s in streaks if sign > 0]
                loss_streaks = [s for sign, s in streaks if sign < 0]
                return max(win_streaks, default=0), max(loss_streaks, default=0)

            max_win_streak, max_loss_streak = get_streaks(sim_result_df['pips'])

            # Profit factor = sum of gains / sum of losses (absolute)
            total_profit = wins['pips'].sum()
            total_loss = abs(losses['pips'].sum())
            profit_factor = (total_profit / total_loss) if total_loss > 0 else np.nan

            # Expectancy = (AvgWin * WinRate) - (AvgLoss * LossRate)
            avg_win = wins['pips'].mean() if not wins.empty else 0
            avg_loss = abs(losses['pips'].mean()) if not losses.empty else 0
            expectancy_pips = (avg_win * (win_rate / 100)) - (avg_loss * (1 - win_rate / 100))
            expectancy_dollars = expectancy_pips * sim_result_df['PipValue'].iloc[
                0] if 'PipValue' in sim_result_df.columns else 0

            # show table
            display_df = sim_df[['entry_date', 'start_balance', 'lot_size', 'profit_$', 'end_balance']].copy()
            display_df = display_df.rename(columns={
                'entry_date': 'Entry Date',
                'start_balance': 'Starting Balance ($)',
                'lot_size': 'Lot Size',
                'profit_$': 'Profit ($)',
                'end_balance': 'Ending Balance ($)'
            })
            st.dataframe(display_df.style.format({
                'Starting Balance ($)': '${:,.2f}',
                'Lot Size': '{:.2f}',
                'Profit ($)': '${:,.2f}',
                'Ending Balance ($)': '${:,.2f}'
            }), use_container_width=True)

            final_balance = sim_result_df['end_balance'].iloc[-1] if not sim_result_df.empty else starting_balance_input

            st.subheader("📊 Performance Metrics")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏁 Final Balance ($)", f"{final_balance:,.2f}")
                st.metric("📊 Final Pips", total_pips)
                st.metric("🎯 Win Rate (%)", f"{win_rate:.1f}")
                st.metric("🔢 Total Trades", total_trades)

            with col2:
                st.metric("📈 Total Gain (%)", f"{total_gain_pct:,.2f}")
                st.metric("🔥 Max Win Streak", max_win_streak)
                st.metric("💀 Max Loss Streak", max_loss_streak)
                st.metric("🕒 Avg Trade Duration (hrs)", f"{avg_trade_duration_hours:.1f}")

            with col3:
                st.metric("📉 Max Drawdown (%)", f"{max_drawdown:.2f}")
                st.metric("🎯 Avg Trades/Month", f"{avg_trades_per_month:.1f}")
                st.metric("📈 Profit Factor", f"{profit_factor:.2f}")
                st.metric("💡 Expectancy (pips)", f"{expectancy_pips:.2f}")

            st.markdown("---")
            # Account growth curve
            st.subheader("Account Growth Curve")
            growth = sim_df[['entry_date', 'end_balance']].copy()
            growth = growth.rename(columns={'entry_date': 'Date', 'end_balance': 'Balance'})
            fig_growth = px.line(growth, x='Date', y='Balance', title="Account Balance Over Time")
            st.plotly_chart(fig_growth, use_container_width=True)

            # Monthly gains % by year tabs
            st.subheader("Monthly Gains % by Year")
            sim_df['year'] = sim_df['entry_date'].dt.year
            sim_df['month'] = sim_df['entry_date'].dt.to_period('M')
            # compute monthly gains as % change of balance at month end vs month start
            monthly_gains = []
            for (y, m), group in sim_df.groupby(['year', sim_df['entry_date'].dt.to_period('M')]):
                # percent gain relative to starting balance for that month (use first start_balance)
                start_bal = group['start_balance'].iloc[0]
                end_bal = group['end_balance'].iloc[-1]
                pct = ((end_bal - start_bal) / start_bal) * 100 if start_bal != 0 else 0.0
                monthly_gains.append({'year': int(y), 'month': str(m), 'pct_gain': pct})
            monthly_df = pd.DataFrame(monthly_gains)
            if monthly_df.empty:
                st.write("No monthly data to display.")
            else:
                years_present = sorted(monthly_df['year'].unique())
                tabs = st.tabs([str(y) for y in years_present])
                for tab_obj, y in zip(tabs, years_present):
                    with tab_obj:
                        ydf = monthly_df[monthly_df['year'] == y].copy()
                        if ydf.empty:
                            st.write("No data.")
                        else:
                            fig_m = px.bar(ydf, x='month', y='pct_gain', title=f"Monthly % Gains — {y}")
                            fig_m.update_layout(xaxis_title="Month", yaxis_title="% Gain")
                            st.plotly_chart(fig_m, use_container_width=True)

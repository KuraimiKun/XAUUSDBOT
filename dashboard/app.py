"""
XAUUSD Multi-Timeframe Scalping Dashboard

Professional dashboard displaying:
- Multi-timeframe trend analysis (H1, M15, M5)
- Key support/resistance levels
- High-probability signals targeting 30+ pips
- Live chart with MTF indicators
- Backtesting with historical analysis
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# Page config - must be first Streamlit call
st.set_page_config(
    page_title="XAUUSD MTF Scalping",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .trend-bullish {
        background-color: #1a472a;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #00ff00;
        margin: 5px 0;
    }
    .trend-bearish {
        background-color: #4a1a1a;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #ff4444;
        margin: 5px 0;
    }
    .trend-neutral {
        background-color: #3a3a3a;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #888888;
        margin: 5px 0;
    }
    .signal-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        border: 1px solid #30363d;
    }
    .key-level {
        background-color: #2a2a3a;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 3px 0;
        display: inline-block;
    }
    .level-support { border-left: 3px solid #00ff00; }
    .level-resistance { border-left: 3px solid #ff4444; }
    .aligned { color: #00ff00; font-weight: bold; }
    .not-aligned { color: #888888; }
    .stat-card {
        background-color: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = "http://localhost:8000"


def fetch_state() -> dict | None:
    """Fetch current state from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/state", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fetch_prices() -> list:
    """Fetch price history from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/prices", timeout=5)
        resp.raise_for_status()
        return resp.json().get("prices", [])
    except Exception:
        return []


def fetch_analysis() -> dict | None:
    """Fetch MTF analysis from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/analysis", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fetch_ml_signals() -> list:
    """Fetch ML signals from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/ml/signals", timeout=5)
        resp.raise_for_status()
        return resp.json().get("signals", [])
    except Exception:
        return []


def fetch_ml_analysis() -> dict | None:
    """Fetch ML analysis from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/ml/analysis", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fetch_settings() -> dict | None:
    """Fetch settings from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/settings", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def get_trend_class(trend: str) -> str:
    """Get CSS class for trend."""
    if trend == "BULLISH":
        return "trend-bullish"
    elif trend == "BEARISH":
        return "trend-bearish"
    return "trend-neutral"


def get_trend_emoji(trend: str) -> str:
    """Get emoji for trend."""
    if trend == "BULLISH":
        return "🟢"
    elif trend == "BEARISH":
        return "🔴"
    return "⚪"


def create_mtf_chart(prices: list, key_levels: list, active_trade: dict | None) -> go.Figure:
    """Create price chart with key levels and trade info."""
    if not prices:
        fig = go.Figure()
        fig.add_annotation(text="Waiting for data...", x=0.5, y=0.5, showarrow=False)
        return fig

    df = pd.DataFrame(prices)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2],
        subplot_titles=("XAUUSD Price", "Spread"),
    )

    # Main price line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["mid"],
            mode="lines",
            name="Price",
            line=dict(color="#ffd700", width=2),
        ),
        row=1, col=1,
    )

    # Bid/Ask spread area
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["ask"],
            mode="lines",
            name="Ask",
            line=dict(color="rgba(255,100,100,0.3)", width=1),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["bid"],
            mode="lines",
            name="Bid",
            line=dict(color="rgba(100,255,100,0.3)", width=1),
            fill="tonexty",
            fillcolor="rgba(100,100,100,0.1)",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Add key levels
    for level in key_levels[:5]:
        color = "#00ff00" if level.get("type") == "support" else "#ff4444"
        fig.add_hline(
            y=level["price"],
            line=dict(color=color, width=1, dash="dot"),
            annotation_text=f"{level['type'][:1].upper()} ({level['timeframe']}): {level['price']:.2f}",
            annotation_position="left",
            row=1, col=1,
        )

    # Add active trade visualization
    if active_trade:
        entry = active_trade["entry_price"]
        tp = active_trade["take_profit"]
        sl = active_trade["stop_loss"]
        trade_color = "#00ff00" if active_trade["type"] == "BUY" else "#ff4444"

        fig.add_hline(
            y=entry,
            line=dict(color=trade_color, width=2),
            annotation_text=f"Entry: {entry:.2f}",
            row=1, col=1,
        )
        fig.add_hline(
            y=tp,
            line=dict(color="#00ff00", width=1, dash="dash"),
            annotation_text=f"TP: {tp:.2f}",
            row=1, col=1,
        )
        fig.add_hline(
            y=sl,
            line=dict(color="#ff0000", width=1, dash="dash"),
            annotation_text=f"SL: {sl:.2f}",
            row=1, col=1,
        )

    # Spread chart
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["spread"],
            mode="lines",
            name="Spread",
            line=dict(color="#888888", width=1),
            fill="tozeroy",
            fillcolor="rgba(136,136,136,0.2)",
        ),
        row=2, col=1,
    )

    # Layout
    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=50, r=50, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    return fig


# ============================================================================
# FRAGMENT COMPONENTS
# ============================================================================

@st.fragment(run_every=timedelta(seconds=2))
def live_price_fragment():
    """Live price and spread display."""
    prices = fetch_prices()
    if not prices:
        st.warning("Waiting for price data...")
        return

    last = prices[-1]
    prev = prices[-2] if len(prices) > 1 else last
    delta = last["mid"] - prev["mid"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💲 XAUUSD", f"{last['mid']:.2f}", f"{delta:+.2f}")
    with col2:
        st.metric("📊 Spread", f"{last['spread']:.3f}")
    with col3:
        st.metric("🕐 Updated", last["timestamp"][11:19])


@st.fragment(run_every=timedelta(seconds=5))
def mtf_analysis_fragment():
    """Multi-timeframe trend analysis."""
    analysis = fetch_analysis()
    
    if not analysis or "analysis" not in analysis:
        st.info("Waiting for MTF analysis...")
        return

    # Alignment status
    aligned = analysis.get("aligned", False)
    if aligned:
        st.markdown('<p class="aligned">✅ ALL TIMEFRAMES ALIGNED</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="not-aligned">⏳ Waiting for alignment...</p>', unsafe_allow_html=True)

    # Timeframe cards
    for tf in ["H1", "M15", "M5"]:
        tf_data = analysis["analysis"].get(tf)
        if tf_data:
            trend = tf_data["trend"]
            css_class = get_trend_class(trend)
            emoji = get_trend_emoji(trend)

            st.markdown(f"""
            <div class="{css_class}">
                <strong>{tf}</strong> {emoji} {trend}
                <br><small>RSI: {tf_data['rsi']:.0f} | ATR: {tf_data['atr']:.2f} | Strength: {tf_data['trend_strength']:.0%}</small>
            </div>
            """, unsafe_allow_html=True)


@st.fragment(run_every=timedelta(seconds=5))
def key_levels_fragment():
    """Display key support/resistance levels."""
    analysis = fetch_analysis()
    
    if not analysis:
        return

    levels = analysis.get("key_levels", [])[:6]
    if not levels:
        st.info("No key levels detected")
        return

    for level in levels:
        level_type = level["type"]
        css_class = "level-support" if level_type == "support" else "level-resistance"
        emoji = "🟢" if level_type == "support" else "🔴"
        
        st.markdown(f"""
        <div class="key-level {css_class}">
            {emoji} {level['timeframe']} {level_type.title()}: <strong>{level['price']:.2f}</strong>
        </div>
        """, unsafe_allow_html=True)


@st.fragment(run_every=timedelta(seconds=2))
def live_chart_fragment():
    """Live price chart."""
    state = fetch_state()
    prices = fetch_prices()

    if not prices:
        st.info("Waiting for price data...")
        return

    key_levels = state.get("key_levels", []) if state else []
    active_trade = state.get("active_trade") if state else None

    fig = create_mtf_chart(prices, key_levels, active_trade)
    st.plotly_chart(fig, key="live_chart")


@st.fragment(run_every=timedelta(seconds=2))
def active_trade_fragment():
    """Active trade display."""
    state = fetch_state()

    if not state:
        st.warning("Connecting...")
        return

    active_trade = state.get("active_trade")
    if active_trade:
        trade_type = active_trade["type"]
        entry = active_trade["entry_price"]
        current = active_trade["current_price"]
        tp = active_trade["take_profit"]
        sl = active_trade["stop_loss"]
        pnl_pips = active_trade["pnl_pips"]
        duration = active_trade["duration_minutes"]
        trailing = active_trade.get("trailing_active", False)

        emoji = "🟢" if trade_type == "BUY" else "🔴"
        pnl_class = "profit" if pnl_pips >= 0 else "loss"
        pnl_color = "#00ff00" if pnl_pips >= 0 else "#ff4444"

        st.markdown(f"""
        <div class="signal-card">
            <h3>{emoji} {trade_type} ACTIVE</h3>
            <p><strong>Entry:</strong> {entry:.2f}</p>
            <p><strong>Current:</strong> {current:.2f}</p>
            <p><strong>TP:</strong> {tp:.2f}</p>
            <p><strong>SL:</strong> {sl:.2f} {"🔒" if trailing else ""}</p>
            <p><strong>PnL:</strong> <span style="color: {pnl_color}; font-size: 1.3em;">{pnl_pips:+.1f} pips</span></p>
            <p><small>Duration: {duration} min {"| Trailing SL active" if trailing else ""}</small></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("⏳ No active trade - Waiting for setup...")


@st.fragment(run_every=timedelta(seconds=3))
def recent_signals_fragment():
    """Recent signals display."""
    state = fetch_state()

    if not state:
        return

    signals = state.get("recent_signals", [])
    if signals:
        for sig in reversed(signals[-3:]):
            sig_type = sig["type"]
            emoji = "🟢" if sig_type == "BUY" else "🔴" if sig_type == "SELL" else "⚪"
            css_class = get_trend_class("BULLISH" if sig_type == "BUY" else "BEARISH" if sig_type == "SELL" else "NEUTRAL")

            ts = sig.get("timestamp", "")[:19].replace("T", " ")
            
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{emoji} {sig_type}</strong> @ {sig.get('entry_price', 0):.2f}
                <br><small>Target: {sig.get('pip_target', 0):.0f} pips | RR: 1:{sig.get('risk_reward', 0):.1f}</small>
                <br><small style="color: #888;">{ts}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No signals yet")


@st.fragment(run_every=timedelta(seconds=5))
def ml_signals_fragment():
    """ML Signals display."""
    signals = fetch_ml_signals()
    
    if not signals:
        st.info("🤖 No ML signals yet")
        return
    
    for sig in reversed(signals[-5:]):
        sig_type = sig.get("type", "")
        emoji = "🤖🟢" if sig_type == "LONG" else "🤖🔴"
        css_class = get_trend_class("BULLISH" if sig_type == "LONG" else "BEARISH")
        
        ts = sig.get("timestamp", "")[:19].replace("T", " ")
        conf = sig.get("confidence", 0) * 100
        
        st.markdown(f"""
        <div class="{css_class}">
            <strong>{emoji} ML {sig_type}</strong> @ {sig.get('entry_price', 0):.2f}
            <br><small>TP: {sig.get('take_profit', 0):.2f} | SL: {sig.get('stop_loss', 0):.2f}</small>
            <br><small>Confidence: {conf:.0f}% | R:R: 1:{sig.get('risk_reward', 0):.1f}</small>
            <br><small style="color: #888;">{ts}</small>
        </div>
        """, unsafe_allow_html=True)


@st.fragment(run_every=timedelta(seconds=10))
def ml_analysis_fragment():
    """ML Analysis display."""
    analysis = fetch_ml_analysis()
    
    if not analysis or "error" in analysis:
        st.info("Waiting for ML analysis...")
        return
    
    # Trend display
    h1_trend = analysis.get("h1_trend", "UNKNOWN")
    h4_trend = analysis.get("h4_trend", "UNKNOWN")
    aligned = analysis.get("mtf_aligned", False)
    
    h1_emoji = "📈" if h1_trend == "BULLISH" else "📉"
    h4_emoji = "📈" if h4_trend == "BULLISH" else "📉"
    
    st.markdown(f"""
    **ML Trend Analysis**
    - H1: {h1_emoji} {h1_trend}
    - H4: {h4_emoji} {h4_trend}
    - Aligned: {"✅" if aligned else "⏳"}
    
    **Indicators:**
    - RSI(14): {analysis.get('rsi14', 0):.1f}
    - ATR: {analysis.get('atr', 0):.2f}
    - EMA Score: {analysis.get('ema_alignment', 0)}
    """)
    
    # Session info
    session = analysis.get("session", {})
    sessions = []
    if session.get("london"):
        sessions.append("🇬🇧 London")
    if session.get("ny"):
        sessions.append("🇺🇸 NY")
    if session.get("overlap"):
        sessions.append("🔥 Overlap")
    
    if sessions:
        st.markdown(f"**Active:** {' | '.join(sessions)}")


@st.fragment(run_every=timedelta(seconds=5))
def stats_fragment():
    """Trading statistics."""
    state = fetch_state()

    if not state or not state.get("stats"):
        return

    stats = state["stats"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Trades", stats.get("total_trades", 0))
    with c2:
        st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    with c3:
        pnl = stats.get("total_pnl_pips", 0)
        st.metric("Total PnL", f"{pnl:+.1f} pips")
    with c4:
        st.metric("Avg Trade", f"{stats.get('avg_pnl_pips', 0):+.1f} pips")


@st.fragment(run_every=timedelta(seconds=10))
def trade_history_fragment():
    """Trade history table."""
    state = fetch_state()

    if not state:
        return

    trades = state.get("recent_trades", [])

    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["result"] = trade_df["won"].apply(lambda x: "✅" if x else "❌")
        trade_df["pnl_pips"] = trade_df["pnl_pips"].apply(lambda x: f"{x:+.1f}")
        trade_df["duration"] = trade_df["duration_minutes"].apply(lambda x: f"{x}m")

        display_cols = ["type", "entry_price", "exit_price", "pnl_pips", "duration", "reason", "result"]
        st.dataframe(trade_df[display_cols].tail(10), hide_index=True)
    else:
        st.info("No trades completed yet")


@st.fragment(run_every=timedelta(seconds=5))
def connection_status_fragment():
    """Connection status."""
    state = fetch_state()
    if state:
        mt5_ok = state.get("mt5_connected", False)
        ml_ok = state.get("ml_service_active", False)
        
        if mt5_ok and ml_ok:
            st.success("🟢 MT5 Connected | 🤖 ML Active")
        elif mt5_ok:
            st.warning("🟢 MT5 Connected | 🔴 ML Inactive")
        else:
            st.error("🔴 MT5 Disconnected")
    else:
        st.error("🔴 Backend Disconnected")


# ============================================================================
# BACKTESTING FUNCTIONS
# ============================================================================

def run_backtest(start_date: datetime, end_date: datetime, initial_balance: float, risk_per_trade: float) -> dict | None:
    """Run backtest via API."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/backtest",
            json={
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_balance": initial_balance,
                "risk_per_trade": risk_per_trade,
            },
            timeout=120,  # Backtests can take time
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return None


def create_equity_curve_chart(equity_curve: list, trades: list) -> go.Figure:
    """Create equity curve with trade markers."""
    if not equity_curve:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure()

    # Equity line
    fig.add_trace(
        go.Scatter(
            x=list(range(len(equity_curve))),
            y=equity_curve,
            mode="lines",
            name="Equity",
            line=dict(color="#ffd700", width=2),
            fill="tozeroy",
            fillcolor="rgba(255,215,0,0.1)",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title="Trade #",
        yaxis_title="Balance ($)",
        showlegend=False,
    )

    return fig


def create_monthly_chart(monthly_breakdown: dict) -> go.Figure:
    """Create monthly performance bar chart."""
    if not monthly_breakdown:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    months = list(monthly_breakdown.keys())
    values = [v.get("pnl", 0) for v in monthly_breakdown.values()]
    colors = ["#00ff00" if v >= 0 else "#ff4444" for v in values]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=months,
            y=values,
            marker_color=colors,
            name="Monthly PnL",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=30, b=50),
        xaxis_title="Month",
        yaxis_title="PnL ($)",
        showlegend=False,
    )

    return fig


def create_trade_distribution_chart(trades: list) -> go.Figure:
    """Create trade PnL distribution histogram."""
    if not trades:
        fig = go.Figure()
        fig.add_annotation(text="No trades", x=0.5, y=0.5, showarrow=False)
        return fig

    pnls = [t.get("pnl", 0) for t in trades]
    colors = ["#00ff00" if p >= 0 else "#ff4444" for p in pnls]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=pnls,
            nbinsx=30,
            marker=dict(
                color="#ffd700",
                line=dict(color="#ffffff", width=1),
            ),
            name="Trade PnL",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title="PnL ($)",
        yaxis_title="Frequency",
        showlegend=False,
    )

    return fig


def render_backtest_tab():
    """Render the backtesting tab."""
    st.header("📈 Strategy Backtesting")
    st.caption("Test the MTF scalping strategy on historical data")

    # Settings
    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Backtest Settings")

        # Date range
        today = datetime.now()
        default_start = today - timedelta(days=30)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                max_value=today - timedelta(days=1),
                key="bt_start",
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=today,
                max_value=today,
                key="bt_end",
            )

        # Parameters
        initial_balance = st.number_input(
            "Initial Balance ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            key="bt_balance",
        )

        risk_pct = st.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            key="bt_risk",
        )

        run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

    # Check if we have results in session state
    if "backtest_result" not in st.session_state:
        st.session_state.backtest_result = None

    # Run backtest
    if run_btn:
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return

        with st.spinner("Running backtest... This may take a minute..."):
            result = run_backtest(
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.min.time()),
                initial_balance,
                risk_pct / 100,
            )

            if result and result.get("success"):
                st.session_state.backtest_result = result
                st.success("✅ Backtest completed!")
            elif result:
                st.error(f"Backtest failed: {result.get('error', 'Unknown error')}")
                return
            else:
                return

    # Display results
    result = st.session_state.backtest_result

    if not result:
        st.info("👆 Configure parameters and click 'Run Backtest' to start")
        
        # Show some tips
        with st.expander("📚 Backtesting Tips"):
            st.markdown("""
            **How to use:**
            1. Select a date range (longer periods give more reliable results)
            2. Set your initial balance and risk per trade
            3. Click 'Run Backtest' and wait for results
            
            **What we test:**
            - Multi-timeframe alignment (H1 + M15 + M5)
            - Key level entries (support/resistance)
            - 30+ pip targets with trailing stops
            - Maximum 20 pip stop loss
            
            **Metrics explained:**
            - **Win Rate**: Percentage of profitable trades
            - **Profit Factor**: Gross profit / Gross loss (>1.5 is good)
            - **Sharpe Ratio**: Risk-adjusted return (>1 is good)
            - **Max Drawdown**: Largest peak-to-trough decline
            """)
        return

    # Summary metrics
    st.subheader("📊 Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        return_pct = result.get("total_return_pct", 0)
        color = "normal" if return_pct >= 0 else "inverse"
        st.metric(
            "Total Return",
            f"${result.get('total_return', 0):,.2f}",
            f"{return_pct:+.2f}%",
            delta_color=color,
        )

    with col2:
        st.metric(
            "Win Rate",
            f"{result.get('win_rate', 0):.1f}%",
            f"{result.get('winning_trades', 0)}/{result.get('total_trades', 0)} trades",
        )

    with col3:
        st.metric(
            "Profit Factor",
            f"{result.get('profit_factor', 0):.2f}",
            "Good" if result.get('profit_factor', 0) > 1.5 else "Needs work",
        )

    with col4:
        st.metric(
            "Max Drawdown",
            f"${result.get('max_drawdown', 0):,.2f}",
            f"{result.get('max_drawdown_pct', 0):.2f}%",
            delta_color="inverse",
        )

    # Second row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.2f}")

    with col2:
        st.metric("Avg Win", f"${result.get('avg_win', 0):,.2f}")

    with col3:
        st.metric("Avg Loss", f"${result.get('avg_loss', 0):,.2f}")

    with col4:
        st.metric(
            "Best / Worst",
            f"${result.get('best_trade', 0):,.2f}",
            f"${result.get('worst_trade', 0):,.2f}",
        )

    # Comparison with buy & hold
    st.divider()
    
    cmp_col1, cmp_col2, cmp_col3 = st.columns(3)
    
    with cmp_col1:
        st.metric(
            "Strategy Return",
            f"{result.get('total_return_pct', 0):+.2f}%",
        )
    
    with cmp_col2:
        st.metric(
            "Buy & Hold Return",
            f"{result.get('buy_hold_pct', 0):+.2f}%",
        )
    
    with cmp_col3:
        alpha = result.get('total_return_pct', 0) - result.get('buy_hold_pct', 0)
        st.metric(
            "Strategy Alpha",
            f"{alpha:+.2f}%",
            "Outperforming" if alpha > 0 else "Underperforming",
        )

    # Charts
    st.divider()
    st.subheader("📈 Equity Curve")

    equity_curve = result.get("equity_curve", [])
    trades = result.get("trades", [])

    fig_equity = create_equity_curve_chart(equity_curve, trades)
    st.plotly_chart(fig_equity, use_container_width=True)

    # Monthly breakdown and trade distribution
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📅 Monthly Performance")
        monthly = result.get("monthly_breakdown", {})
        if monthly:
            fig_monthly = create_monthly_chart(monthly)
            st.plotly_chart(fig_monthly, use_container_width=True)

            # Monthly table
            monthly_df = pd.DataFrame([
                {
                    "Month": k,
                    "Trades": v.get("trades", 0),
                    "Wins": v.get("wins", 0),
                    "Win Rate": f"{v.get('win_rate', 0):.1f}%",
                    "PnL": f"${v.get('pnl', 0):,.2f}",
                }
                for k, v in monthly.items()
            ])
            st.dataframe(monthly_df, hide_index=True, use_container_width=True)
        else:
            st.info("No monthly data available")

    with chart_col2:
        st.subheader("📊 Trade Distribution")
        fig_dist = create_trade_distribution_chart(trades)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Win/Loss streaks
        st.markdown(f"""
        **Streaks:**
        - 🔥 Best winning streak: **{result.get('consecutive_wins', 0)}** trades
        - ❄️ Worst losing streak: **{result.get('consecutive_losses', 0)}** trades
        """)

    # Trade list
    st.divider()
    st.subheader("📋 Trade History")

    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["result"] = trade_df["pnl"].apply(lambda x: "✅" if x >= 0 else "❌")
        trade_df["pnl_fmt"] = trade_df["pnl"].apply(lambda x: f"${x:+,.2f}")
        trade_df["entry_time_fmt"] = trade_df["entry_time"].apply(lambda x: x[:16] if x else "")
        trade_df["exit_time_fmt"] = trade_df["exit_time"].apply(lambda x: x[:16] if x else "")

        display_df = trade_df[[
            "id", "signal_type", "entry_price", "exit_price",
            "pnl_fmt", "exit_reason", "entry_time_fmt", "exit_time_fmt", "result"
        ]].rename(columns={
            "id": "ID",
            "signal_type": "Type",
            "entry_price": "Entry",
            "exit_price": "Exit",
            "pnl_fmt": "PnL",
            "exit_reason": "Reason",
            "entry_time_fmt": "Entry Time",
            "exit_time_fmt": "Exit Time",
            "result": "Result",
        })

        st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)

        # Download button
        csv = trade_df.to_csv(index=False)
        st.download_button(
            "📥 Download Trades CSV",
            csv,
            "backtest_trades.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        st.info("No trades in this period")


def render_live_tab():
    """Render the live trading tab."""
    # Live content
    live_price_fragment()

    st.divider()

    # Chart and Active Trade
    chart_col, trade_col = st.columns([2.5, 1])

    with chart_col:
        st.subheader("📈 Live Price Chart")
        live_chart_fragment()

    with trade_col:
        st.subheader("📡 Active Trade")
        active_trade_fragment()

        st.subheader("📋 Recent Signals")
        recent_signals_fragment()
    
    # ML Signals Section
    st.divider()
    ml_col1, ml_col2 = st.columns([1, 1])
    
    with ml_col1:
        st.subheader("🤖 ML Signals (54% WR)")
        ml_signals_fragment()
    
    with ml_col2:
        st.subheader("🧠 ML Analysis")
        ml_analysis_fragment()

    # Statistics and History
    st.divider()
    st.subheader("📊 Performance")
    stats_fragment()

    st.divider()
    st.subheader("📜 Trade History")
    trade_history_fragment()

    # Footer
    st.divider()
    _, status_col = st.columns([3, 1])
    with status_col:
        connection_status_fragment()


# ============================================================================
# MAIN LAYOUT
# ============================================================================

# Title
st.title("📊 XAUUSD Multi-Timeframe Scalping")
st.caption("Targeting 30+ pip moves with MTF confluence")

# Tabs
tab1, tab2 = st.tabs(["📡 Live Trading", "📈 Backtesting"])

# Sidebar (common settings)
with st.sidebar:
    st.header("⚙️ Settings")
    
    current_settings = fetch_settings()
    if current_settings:
        st.subheader("Target Configuration")
        st.write(f"📎 Min TP: **{current_settings.get('min_tp_pips', 30):.0f} pips**")
        st.write(f"🛡️ Max SL: **{current_settings.get('max_sl_pips', 20):.0f} pips**")
        st.write(f"⏱️ Cooldown: **{current_settings.get('cooldown_minutes', 15)} min**")
        st.write(f"🎯 Min Confidence: **{current_settings.get('min_signal_confidence', 0.7):.0%}**")
        
        st.divider()
        st.subheader("Indicators")
        st.write(f"EMA Fast: {current_settings.get('ema_fast', 9)}")
        st.write(f"EMA Slow: {current_settings.get('ema_slow', 21)}")
        st.write(f"EMA Trend: {current_settings.get('ema_trend', 50)}")
        st.write(f"RSI Period: {current_settings.get('rsi_period', 14)}")
    
    st.divider()
    st.subheader("📡 MTF Analysis")
    mtf_analysis_fragment()
    
    st.divider()
    st.subheader("📍 Key Levels")
    key_levels_fragment()

# Tab content
with tab1:
    render_live_tab()

with tab2:
    render_backtest_tab()

"""
XAUUSD Scalping Signal Dashboard

Real-time scalping signals with live chart visualization.
Uses st.fragment for smooth updates without page refresh.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# Page config - must be first Streamlit call
st.set_page_config(
    page_title="XAUUSD Scalping Signals",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
    }
    .signal-buy {
        background-color: #1a472a;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00ff00;
        margin-bottom: 10px;
    }
    .signal-sell {
        background-color: #4a1a1a;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
        margin-bottom: 10px;
    }
    .signal-exit {
        background-color: #3a3a3a;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #888888;
        margin-bottom: 10px;
    }
    .active-trade {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .profit { color: #00ff00 !important; }
    .loss { color: #ff4444 !important; }
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


def fetch_settings() -> dict | None:
    """Fetch settings from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/settings", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def create_price_chart(prices: list, active_trade: dict | None, indicators: dict | None) -> go.Figure:
    """Create interactive price chart with indicators."""
    if not prices:
        fig = go.Figure()
        fig.add_annotation(text="Waiting for data...", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", height=450)
        return fig

    df = pd.DataFrame(prices)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
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

    # Bid/Ask spread visualization
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["ask"],
            mode="lines", name="Ask",
            line=dict(color="rgba(255,100,100,0.4)", width=1),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["bid"],
            mode="lines", name="Bid",
            line=dict(color="rgba(100,255,100,0.4)", width=1),
            fill="tonexty",
            fillcolor="rgba(100,100,100,0.1)",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Add EMA lines
    if indicators and indicators.get("ema_fast") and indicators.get("ema_slow"):
        fig.add_hline(
            y=indicators["ema_fast"],
            line=dict(color="#00bfff", width=1, dash="dash"),
            annotation_text=f"EMA Fast: {indicators['ema_fast']:.2f}",
            row=1, col=1,
        )
        fig.add_hline(
            y=indicators["ema_slow"],
            line=dict(color="#ff6b6b", width=1, dash="dash"),
            annotation_text=f"EMA Slow: {indicators['ema_slow']:.2f}",
            row=1, col=1,
        )

    # Add active trade levels
    if active_trade:
        entry = active_trade["entry_price"]
        tp = active_trade["take_profit"]
        sl = active_trade["stop_loss"]
        color = "#00ff00" if active_trade["type"] == "BUY" else "#ff4444"

        fig.add_hline(y=entry, line=dict(color=color, width=2),
                      annotation_text=f"Entry: {entry:.2f}", row=1, col=1)
        fig.add_hline(y=tp, line=dict(color="#00ff00", width=1, dash="dot"),
                      annotation_text=f"TP: {tp:.2f}", row=1, col=1)
        fig.add_hline(y=sl, line=dict(color="#ff0000", width=1, dash="dot"),
                      annotation_text=f"SL: {sl:.2f}", row=1, col=1)

    # Spread chart
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["spread"],
            mode="lines", name="Spread",
            line=dict(color="#888888", width=1),
            fill="tozeroy",
            fillcolor="rgba(136,136,136,0.2)",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    return fig


# ============ FRAGMENT: Auto-updating data section ============
@st.fragment(run_every=2)
def live_data_section():
    """This fragment updates every 2 seconds without refreshing the whole page."""
    
    state = fetch_state()
    prices = fetch_prices()
    
    if not state:
        st.error("⚠️ Cannot connect to backend")
        return
    
    indicators = state.get("indicators") or {}
    stats = state.get("stats") or {}
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if prices:
            last_price = prices[-1]["mid"]
            prev_price = prices[-2]["mid"] if len(prices) > 1 else last_price
            delta = last_price - prev_price
            st.metric("💲 Price", f"{last_price:.2f}", f"{delta:+.2f}")
        else:
            st.metric("💲 Price", "---")

    with col2:
        rsi = indicators.get("rsi", 50)
        rsi_icon = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "⚪"
        st.metric(f"{rsi_icon} RSI", f"{rsi:.1f}")

    with col3:
        spread = prices[-1]["spread"] if prices else 0
        st.metric("📊 Spread", f"{spread:.3f}")

    with col4:
        win_rate = stats.get("win_rate", 0)
        st.metric("🎯 Win Rate", f"{win_rate:.1f}%")

    with col5:
        total_pnl = stats.get("total_pnl_points", 0)
        st.metric("💵 Total PnL", f"{total_pnl:+.2f} pts")

    st.divider()

    # Chart and signals
    chart_col, signal_col = st.columns([2, 1])

    with chart_col:
        st.subheader("📈 Live Chart")
        fig = create_price_chart(prices, state.get("active_trade"), indicators)
        st.plotly_chart(fig, use_container_width=True, key="live_chart")

    with signal_col:
        st.subheader("📡 Active Trade")
        
        active_trade = state.get("active_trade")
        if active_trade:
            trade_type = active_trade["type"]
            entry = active_trade["entry_price"]
            tp = active_trade["take_profit"]
            sl = active_trade["stop_loss"]
            
            current = prices[-1]["mid"] if prices else entry
            pnl = (current - entry) if trade_type == "BUY" else (entry - current)
            pnl_class = "profit" if pnl >= 0 else "loss"
            emoji = "🟢" if trade_type == "BUY" else "🔴"

            st.markdown(f"""
            <div class="active-trade">
                <h3>{emoji} {trade_type} ACTIVE</h3>
                <p><strong>Entry:</strong> {entry:.2f}</p>
                <p><strong>TP:</strong> {tp:.2f}</p>
                <p><strong>SL:</strong> {sl:.2f}</p>
                <p><strong>Current PnL:</strong> <span class="{pnl_class}">{pnl:+.2f} pts</span></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⏳ Waiting for signal...")

        st.subheader("📋 Recent Signals")
        signals = state.get("recent_signals", [])
        if signals:
            for sig in reversed(signals[-5:]):
                sig_type = sig["type"]
                if sig_type == "BUY":
                    css_class, emoji = "signal-buy", "🟢"
                elif sig_type == "SELL":
                    css_class, emoji = "signal-sell", "🔴"
                else:
                    css_class, emoji = "signal-exit", "⚪"

                ts = sig.get("timestamp", "")[:19].replace("T", " ")
                entry_price = sig.get('entry_price', 0)
                reason = sig.get('reason', '')
                
                st.markdown(f"""
                <div class="{css_class}">
                    <strong>{emoji} {sig_type}</strong> @ {entry_price:.2f}<br>
                    <small>{reason}</small><br>
                    <small style="color: #888;">{ts}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No signals yet")

    # Trade history
    st.divider()
    st.subheader("📜 Trade History")

    trades = state.get("recent_trades", [])
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["won"] = trade_df["won"].apply(lambda x: "✅" if x else "❌")
        trade_df["pnl_points"] = trade_df["pnl_points"].apply(lambda x: f"{x:+.2f}")
        trade_df["pnl_usd"] = trade_df["pnl_usd"].apply(lambda x: f"${x:+.2f}")

        cols = ["type", "entry_price", "exit_price", "pnl_points", "pnl_usd", "reason", "won"]
        st.dataframe(trade_df[cols], use_container_width=True, hide_index=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", stats.get("total_trades", 0))
        c2.metric("Winning", stats.get("winning_trades", 0))
        c3.metric("Losing", stats.get("losing_trades", 0))
        c4.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    else:
        st.info("No trades completed yet")

    # Connection status
    st.divider()
    _, status_col = st.columns([3, 1])
    with status_col:
        if state.get("mt5_connected"):
            st.success("🟢 MT5 Connected")
        else:
            st.error("🔴 MT5 Disconnected")


# ============ MAIN PAGE LAYOUT ============
st.title("💰 XAUUSD Scalping Signals")

# Sidebar (static, doesn't need refresh)
with st.sidebar:
    st.header("⚙️ Settings")
    st.divider()
    
    settings = fetch_settings()
    if settings:
        st.subheader("Engine Parameters")
        st.json(settings)
    else:
        st.warning("Cannot load settings")

# Run the auto-updating fragment
live_data_section()

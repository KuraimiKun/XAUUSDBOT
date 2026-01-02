# 💰 XAUUSD Scalping Signal System

Real-time scalping signals for XAUUSD (Gold) with live chart visualization.

## Features

- **Real-time Price Monitoring** - Connects to MetaTrader 5 for live tick data
- **Smart Entry Signals** - BUY/SELL signals based on:
  - EMA crossovers (fast/slow)
  - RSI overbought/oversold levels
  - Price momentum analysis
  - Spread filtering
- **Automatic TP/SL** - ATR-based take profit and stop loss levels
- **Live Dashboard** - Real-time chart with indicators and signals
- **Telegram Alerts** - Instant notifications for entries and exits
- **Trade Tracking** - Win rate, PnL, and performance stats

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
# MT5 Connection
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe

# Telegram Alerts (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Engine Settings
EMA_FAST=5
EMA_SLOW=13
RSI_PERIOD=7
RSI_OVERBOUGHT=70
RSI_OVERSOLD=30
ATR_TP_MULTIPLIER=1.5
ATR_SL_MULTIPLIER=1.0
MAX_SPREAD=0.50
MIN_SIGNAL_CONFIDENCE=0.6
FETCH_INTERVAL_SECONDS=0.5
```

### 3. Start the Backend

```bash
python -m uvicorn backend.app.main:app --reload
```

### 4. Start the Dashboard

```bash
streamlit run dashboard/app.py
```

## Signal Logic

### Entry Conditions

**BUY Signal** (confidence score ≥ 0.6):
- EMA fast crosses above EMA slow (+40%)
- RSI < 30 (oversold) (+25%)
- Strong upward momentum (+15%)
- Price above EMAs (+10%)

**SELL Signal** (confidence score ≥ 0.6):
- EMA fast crosses below EMA slow (+40%)
- RSI > 70 (overbought) (+25%)
- Strong downward momentum (+15%)
- Price below EMAs (+10%)

### Exit Conditions

- Take Profit hit (ATR × 1.5)
- Stop Loss hit (ATR × 1.0)
- EMA reversal with profit (early exit)

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | System status |
| `GET /health` | Health check |
| `GET /api/state` | Current system state |
| `GET /api/prices` | Price history (500 ticks) |
| `GET /api/signals` | Signal history |
| `GET /api/trades` | Trade history |
| `GET /api/settings` | Engine settings |
| `WS /ws` | WebSocket for real-time updates |

## Architecture

```
backend/
├── app/
│   ├── main.py           # FastAPI app with WebSocket
│   ├── config.py         # Settings management
│   └── services/
│       ├── scalping_engine.py  # Core signal logic
│       ├── mt5_provider.py     # MT5 data feed
│       └── alerts.py           # Telegram notifications

dashboard/
└── app.py                # Streamlit dashboard
```

## Disclaimer

⚠️ **This is for educational purposes only.** Trading forex involves substantial risk of loss. Past performance is not indicative of future results. Never trade with money you cannot afford to lose.

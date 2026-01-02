"""
XAUUSD Multi-Timeframe Scalping Signal System - Main FastAPI Application

Uses MTF analysis (H1, M15, M5) to generate high-probability signals
targeting 30+ pips with proper risk management.
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .config import settings
from .services.mtf_scalping_engine import (
    MTFScalpingEngine,
    PriceTick,
    TradeSetup,
    SignalType,
)
from .services.mt5_provider import MT5Provider
from .services.alerts import get_telegram_alerts
from .services.backtesting import BacktestEngine
from .services.ml_signal_service import get_ml_signal_service, MLSignal

# Global state
engine: MTFScalpingEngine | None = None
mt5_provider: MT5Provider | None = None
ml_service = None  # ML Signal Service
price_history: deque[dict] = deque(maxlen=500)  # For chart data
signal_history: deque[dict] = deque(maxlen=100)
ml_signal_history: deque[dict] = deque(maxlen=100)  # ML signals
active_connections: list[WebSocket] = []
_running = False
_last_candle_update = datetime.now(timezone.utc)
_last_ml_check = datetime.now(timezone.utc)


async def broadcast(data: dict) -> None:
    """Broadcast data to all connected WebSocket clients."""
    disconnected = []
    for ws in active_connections:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_connections:
            active_connections.remove(ws)


async def on_signal(setup: TradeSetup) -> None:
    """Handle new signal from engine."""
    signal_data = {
        "type": setup.signal_type.value,
        "entry_price": setup.entry_price,
        "take_profit": setup.take_profit,
        "stop_loss": setup.stop_loss,
        "risk_reward": setup.risk_reward,
        "confidence": setup.confidence,
        "reason": setup.entry_reason,
        "factors": setup.confluence_factors,
        "pip_target": setup.pip_target,
        "pip_risk": setup.pip_risk,
        "h1_trend": setup.h1_trend.value,
        "m15_trend": setup.m15_trend.value,
        "m5_trend": setup.m5_trend.value,
        "timestamp": setup.timestamp.isoformat(),
    }
    signal_history.append(signal_data)

    # Broadcast to WebSocket clients
    await broadcast({"event": "signal", "data": signal_data})

    # Send Telegram alert
    alerts = get_telegram_alerts()
    if alerts.enabled:
        message = (
            f"🚀 *{setup.signal_type.value} SIGNAL*\n\n"
            f"💰 Entry: `{setup.entry_price:.2f}`\n"
            f"🎯 TP: `{setup.take_profit:.2f}` ({setup.pip_target:.0f} pips)\n"
            f"🛡️ SL: `{setup.stop_loss:.2f}` ({setup.pip_risk:.0f} pips)\n"
            f"📊 R:R = 1:{setup.risk_reward:.1f}\n"
            f"✅ Confidence: {setup.confidence:.0%}\n\n"
            f"*MTF Analysis:*\n"
            f"H1: {setup.h1_trend.value} | M15: {setup.m15_trend.value} | M5: {setup.m5_trend.value}\n\n"
            f"*Factors:*\n{chr(10).join('• ' + f for f in setup.confluence_factors)}"
        )
        asyncio.create_task(alerts.send_message(message))


async def on_trade_close(trade: dict) -> None:
    """Handle trade closure."""
    await broadcast({"event": "trade_closed", "data": trade})

    # Send Telegram alert
    alerts = get_telegram_alerts()
    if alerts.enabled:
        emoji = "✅" if trade.get("won") else "❌"
        pnl = trade.get("pnl_pips", 0)
        message = (
            f"{emoji} *TRADE CLOSED*\n\n"
            f"📈 {trade.get('type')}: {trade.get('entry_price'):.2f} → {trade.get('exit_price'):.2f}\n"
            f"💵 PnL: `{pnl:+.1f} pips`\n"
            f"⏱️ Duration: {trade.get('duration_minutes', 0)} min\n"
            f"📝 {trade.get('reason')}"
        )
        asyncio.create_task(alerts.send_message(message))


async def on_ml_signal(signal: MLSignal) -> None:
    """Handle new ML signal."""
    signal_data = {
        "type": signal.direction,
        "entry_price": signal.entry_price,
        "take_profit": signal.take_profit,
        "stop_loss": signal.stop_loss,
        "confidence": signal.confidence,
        "risk_pips": signal.risk_pips,
        "reward_pips": signal.reward_pips,
        "risk_reward": signal.risk_reward,
        "h1_trend": "BULLISH" if signal.h1_trend > 0 else "BEARISH",
        "h4_trend": "BULLISH" if signal.h4_trend > 0 else "BEARISH",
        "atr": signal.atr,
        "source": "ML",
        "timestamp": signal.timestamp.isoformat(),
    }
    ml_signal_history.append(signal_data)

    # Broadcast to WebSocket clients
    await broadcast({"event": "ml_signal", "data": signal_data})

    # Send Telegram alert
    alerts = get_telegram_alerts()
    if alerts.enabled:
        emoji = "🤖🟢" if signal.direction == "LONG" else "🤖🔴"
        trend_emoji = "📈" if signal.h1_trend > 0 else "📉"
        message = (
            f"{emoji} *ML {signal.direction} SIGNAL*\n\n"
            f"💰 Entry: `{signal.entry_price:.2f}`\n"
            f"🎯 TP: `{signal.take_profit:.2f}` (+{signal.reward_pips:.0f} pips)\n"
            f"🛡️ SL: `{signal.stop_loss:.2f}` (-{signal.risk_pips:.0f} pips)\n"
            f"📊 R:R = 1:{signal.risk_reward:.1f}\n"
            f"🎲 Confidence: {signal.confidence:.0%}\n\n"
            f"{trend_emoji} *Trend:* H1={signal_data['h1_trend']} | H4={signal_data['h4_trend']}\n"
            f"📏 ATR: {signal.atr:.2f}\n\n"
            f"_Model: XGBoost Daily (54% WR)_"
        )
        asyncio.create_task(alerts.send_message(message))


async def price_loop() -> None:
    """Main loop to fetch prices and generate signals."""
    global _running, engine, mt5_provider, ml_service, _last_candle_update, _last_ml_check

    engine = MTFScalpingEngine(
        symbol="XAUUSD",
        ema_fast=settings.ema_fast,
        ema_slow=settings.ema_slow,
        ema_trend=settings.ema_trend,
        rsi_period=settings.rsi_period,
        rsi_ob=settings.rsi_overbought,
        rsi_os=settings.rsi_oversold,
        atr_period=settings.atr_period,
        min_tp_points=settings.min_tp_points,
        max_sl_points=settings.max_sl_points,
        max_spread=settings.max_spread,
        min_confidence=settings.min_signal_confidence,
        cooldown_minutes=settings.cooldown_minutes,
        candle_lookback=settings.candle_lookback,
    )

    # Set callbacks
    engine.set_signal_callback(lambda s: asyncio.create_task(on_signal(s)))
    engine.set_trade_close_callback(lambda t: asyncio.create_task(on_trade_close(t)))

    mt5_provider = MT5Provider()
    
    # Initialize ML Signal Service
    ml_service = get_ml_signal_service()
    logger.info(f"ML Signal Service loaded: {ml_service.is_loaded}")

    try:
        await mt5_provider.connect()
        logger.info("MTF Scalping engine started - analyzing H1, M15, M5 timeframes")
        logger.info("ML Signal Service active - checking every 5 minutes")
        _running = True

        # Initial candle fetch
        await engine.update_all_candles()
        engine.analyze_all_timeframes()
        logger.info("Initial candle analysis complete")

        tick_count = 0
        candle_update_interval = settings.candle_update_seconds
        ml_check_interval = 300  # Check ML signals every 5 minutes

        while _running:
            try:
                # Fetch current tick
                tick = await mt5_provider.fetch_tick()
                if not tick:
                    await asyncio.sleep(0.5)
                    continue

                tick_count += 1

                # Store for chart
                price_data = {
                    "timestamp": tick.timestamp.isoformat(),
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "mid": tick.mid,
                    "spread": round(tick.spread, 3),
                }
                price_history.append(price_data)

                # Update candles periodically (every N seconds)
                now = datetime.now(timezone.utc)
                if (now - _last_candle_update).total_seconds() >= candle_update_interval:
                    await engine.update_all_candles()
                    engine.analyze_all_timeframes()
                    _last_candle_update = now
                    logger.debug("Candles updated and analyzed")
                
                # Check ML signals periodically
                if (now - _last_ml_check).total_seconds() >= ml_check_interval:
                    try:
                        ml_signal = ml_service.check_signal()
                        if ml_signal:
                            await on_ml_signal(ml_signal)
                    except Exception as e:
                        logger.error(f"ML signal check error: {e}")
                    _last_ml_check = now

                # Process tick for signals
                setup = await engine.process_tick(tick)
                
                # Build state update
                indicators = engine.get_current_indicators()
                active_trade = engine.get_active_trade_info()
                key_levels = engine.get_key_levels()

                # Broadcast tick update
                await broadcast({
                    "event": "tick",
                    "data": {
                        **price_data,
                        "indicators": indicators,
                        "active_trade": active_trade,
                        "stats": engine.get_stats(),
                        "key_levels": key_levels[:5],  # Top 5 levels
                    }
                })

                await asyncio.sleep(settings.fetch_interval_seconds)

            except Exception as e:
                logger.error(f"Price loop error: {e}")
                await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Failed to start: {e}")
    finally:
        if mt5_provider:
            await mt5_provider.disconnect()
        _running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    task = asyncio.create_task(price_loop())
    yield
    global _running
    _running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": settings.app_name, "status": "running" if _running else "stopped"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mt5_connected": mt5_provider.is_connected if mt5_provider else False,
        "engine_active": engine is not None,
    }


@app.get("/api/state")
async def get_state():
    """Get current system state with MTF analysis."""
    if not engine:
        return {"running": False, "error": "Engine not started"}
    
    return {
        "running": _running,
        "mt5_connected": mt5_provider.is_connected if mt5_provider else False,
        "ml_service_active": ml_service.is_loaded if ml_service else False,
        "stats": engine.get_stats(),
        "indicators": engine.get_current_indicators(),
        "active_trade": engine.get_active_trade_info(),
        "key_levels": engine.get_key_levels()[:5],
        "recent_signals": list(signal_history)[-10:],
        "recent_ml_signals": list(ml_signal_history)[-10:],
        "recent_trades": engine.trade_history[-10:] if engine else [],
    }


@app.get("/api/prices")
async def get_prices():
    """Get price history for chart."""
    return {"prices": list(price_history)}


@app.get("/api/signals")
async def get_signals():
    """Get signal history."""
    return {"signals": list(signal_history)}


@app.get("/api/ml/signals")
async def get_ml_signals():
    """Get ML signal history."""
    return {"signals": list(ml_signal_history)}


@app.get("/api/ml/analysis")
async def get_ml_analysis():
    """Get current ML analysis."""
    if not ml_service:
        return {"error": "ML service not initialized"}
    return ml_service.get_current_analysis()


@app.post("/api/ml/check")
async def trigger_ml_check():
    """Manually trigger ML signal check."""
    if not ml_service:
        return {"error": "ML service not initialized"}
    
    try:
        signal = ml_service.check_signal()
        if signal:
            await on_ml_signal(signal)
            return {
                "signal": True,
                "direction": signal.direction,
                "entry": signal.entry_price,
                "tp": signal.take_profit,
                "sl": signal.stop_loss,
                "confidence": signal.confidence,
            }
        return {"signal": False, "message": "No signal at this time"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/ml/backtest")
async def run_ml_backtest(request: dict):
    """Run backtest using ML model predictions."""
    if not ml_service:
        return {"error": "ML service not initialized"}
    
    try:
        from datetime import datetime
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np
        
        # Parse dates
        start_date_str = request.get("start_date")
        end_date_str = request.get("end_date")
        
        if not start_date_str or not end_date_str:
            return {"error": "start_date and end_date are required"}
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        # Get parameters
        prob_threshold = request.get("prob_threshold", 0.35)
        sl_mult = request.get("sl_multiplier", 1.2)
        tp_mult = request.get("tp_multiplier", 1.2)
        
        logger.info(f"Starting ML backtest from {start_date} to {end_date}")
        
        # Run backtest in executor
        import concurrent.futures
        loop = asyncio.get_event_loop()
        
        def run_ml_backtest_sync():
            if not mt5.initialize():
                return {"error": "MT5 initialization failed"}
            
            # Fetch data
            df = pd.DataFrame(mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, start_date, end_date))
            if df.empty:
                return {"error": "No data fetched"}
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            h1_df = pd.DataFrame(mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H1, start_date, end_date))
            h1_df['time'] = pd.to_datetime(h1_df['time'], unit='s')
            h1_df.set_index('time', inplace=True)
            
            h4_df = pd.DataFrame(mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H4, start_date, end_date))
            h4_df['time'] = pd.to_datetime(h4_df['time'], unit='s')
            h4_df.set_index('time', inplace=True)
            
            # Build features (simplified version from ml_signal_service)
            def ema(s, p): return s.ewm(span=p, adjust=False).mean()
            def rsi(s, p=14):
                d = s.diff()
                g = d.where(d > 0, 0).rolling(p).mean()
                l = (-d.where(d < 0, 0)).rolling(p).mean()
                return 100 - 100 / (1 + g/l)
            def atr(df, p=14):
                tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
                return tr.rolling(p).mean()
            
            # Basic features
            df['ret'] = df['close'].pct_change()
            df['ret5'] = df['close'].pct_change(5)
            df['vol10'] = df['ret'].rolling(10).std()
            df['vol20'] = df['ret'].rolling(20).std()
            df['body'] = (df['close'] - df['open']) / df['open'] * 100
            df['range'] = df['high'] - df['low']
            df['bull'] = (df['close'] > df['open']).astype(int)
            
            for p in [8, 14, 21, 34, 55, 100]:
                df[f'ema{p}'] = ema(df['close'], p)
                df[f'cvs{p}'] = (df['close'] - df[f'ema{p}']) / df[f'ema{p}'] * 100
            
            df['ema_align'] = ((df['ema8'] > df['ema14']).astype(int) + 
                               (df['ema14'] > df['ema21']).astype(int) +
                               (df['ema21'] > df['ema55']).astype(int) +
                               (df['ema55'] > df['ema100']).astype(int)) - 2
            df['ema21_slope'] = df['ema21'].pct_change(3) * 100
            df['rsi14'] = rsi(df['close'], 14)
            df['rsi7'] = rsi(df['close'], 7)
            df['atr'] = atr(df, 14)
            df['atr_pct'] = df['atr'] / df['close'] * 100
            df['macd'] = ema(df['close'], 12) - ema(df['close'], 26)
            df['macd_sig'] = ema(df['macd'], 9)
            df['macd_hist'] = df['macd'] - df['macd_sig']
            
            low_min = df['low'].rolling(14).min()
            high_max = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            df['high20'] = df['high'].rolling(20).max()
            df['low20'] = df['low'].rolling(20).min()
            df['dist_hi'] = (df['high20'] - df['close']) / df['atr']
            df['dist_lo'] = (df['close'] - df['low20']) / df['atr']
            
            if 'tick_volume' in df.columns:
                df['vol_ma'] = df['tick_volume'].rolling(10).mean()
                df['vol_ratio'] = df['tick_volume'] / df['vol_ma']
            
            df['hour'] = df.index.hour
            df['dow'] = df.index.dayofweek
            df['london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            df['overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
            df['active'] = ((df['london'] == 1) | (df['ny'] == 1)).astype(int)
            
            # H1/H4 features
            h1_df['h1_ema21'] = ema(h1_df['close'], 21)
            h1_df['h1_ema55'] = ema(h1_df['close'], 55)
            h1_df['h1_rsi'] = rsi(h1_df['close'], 14)
            h1_df['h1_trend'] = np.where(h1_df['h1_ema21'] > h1_df['h1_ema55'], 1, -1)
            
            h4_df['h4_ema21'] = ema(h4_df['close'], 21)
            h4_df['h4_ema55'] = ema(h4_df['close'], 55)
            h4_df['h4_trend'] = np.where(h4_df['h4_ema21'] > h4_df['h4_ema55'], 1, -1)
            
            for c in ['h1_trend', 'h1_rsi']:
                df[c] = df.index.map(lambda x: h1_df.loc[:x, c].iloc[-1] if len(h1_df.loc[:x]) > 0 else 0)
            df['h4_trend'] = df.index.map(lambda x: h4_df.loc[:x, 'h4_trend'].iloc[-1] if len(h4_df.loc[:x]) > 0 else 0)
            df['mtf'] = df['h1_trend'] + df['h4_trend']
            
            df.dropna(inplace=True)
            
            # Get predictions
            X = df[ml_service.features]
            X_scaled = ml_service.scaler.transform(X)
            proba = ml_service.model.predict_proba(X_scaled)
            
            df['p0'], df['p1'], df['p2'] = proba[:,0], proba[:,1], proba[:,2]
            
            # Backtest
            trades = []
            last_idx = -2
            
            for i, (idx, row) in enumerate(df.iterrows()):
                if i - last_idx < 2:
                    continue
                
                if row['p1'] >= prob_threshold and row['h1_trend'] > 0:
                    d = 'LONG'
                elif row['p2'] >= prob_threshold and row['h1_trend'] < 0:
                    d = 'SHORT'
                else:
                    continue
                
                if row['rsi14'] > 75 or row['rsi14'] < 25:
                    continue
                
                e = row['close']
                a = row['atr']
                fut = df.iloc[i+1:i+20]
                
                for _, f in fut.iterrows():
                    if d == 'LONG':
                        if f['low'] <= e - a*sl_mult:
                            trades.append({'d': d, 'r': 'SL', 'pips': -a*sl_mult*10, 'entry': e, 'time': idx})
                            last_idx = i
                            break
                        if f['high'] >= e + a*tp_mult:
                            trades.append({'d': d, 'r': 'TP', 'pips': a*tp_mult*10, 'entry': e, 'time': idx})
                            last_idx = i
                            break
                    else:
                        if f['high'] >= e + a*sl_mult:
                            trades.append({'d': d, 'r': 'SL', 'pips': -a*sl_mult*10, 'entry': e, 'time': idx})
                            last_idx = i
                            break
                        if f['low'] <= e - a*tp_mult:
                            trades.append({'d': d, 'r': 'TP', 'pips': a*tp_mult*10, 'entry': e, 'time': idx})
                            last_idx = i
                            break
            
            mt5.shutdown()
            
            if not trades:
                return {"error": "No trades generated", "trades": 0}
            
            # Calculate stats
            tdf = pd.DataFrame(trades)
            n = len(tdf)
            w = len(tdf[tdf['pips'] > 0])
            wr = w/n*100
            total_pips = tdf['pips'].sum()
            gp = tdf[tdf['pips'] > 0]['pips'].sum()
            gl = abs(tdf[tdf['pips'] < 0]['pips'].sum())
            pf = gp/gl if gl > 0 else 999
            
            return {
                "success": True,
                "period": f"{start_date.date()} to {end_date.date()}",
                "total_trades": n,
                "winning_trades": w,
                "losing_trades": n - w,
                "win_rate": round(wr, 1),
                "total_pips": round(total_pips, 0),
                "profit_factor": round(pf, 2),
                "long_trades": len(tdf[tdf['d']=='LONG']),
                "short_trades": len(tdf[tdf['d']=='SHORT']),
                "settings": {
                    "prob_threshold": prob_threshold,
                    "sl_multiplier": sl_mult,
                    "tp_multiplier": tp_mult,
                },
                "trades": [
                    {"direction": t['d'], "result": t['r'], "pips": round(t['pips'], 1), "entry": round(t['entry'], 2)}
                    for t in trades[-50:]  # Last 50 trades
                ]
            }
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, run_ml_backtest_sync)
        
        return result
        
    except Exception as e:
        logger.error(f"ML backtest error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False}


@app.get("/api/trades")
async def get_trades():
    """Get trade history."""
    return {"trades": engine.trade_history if engine else []}


@app.get("/api/analysis")
async def get_analysis():
    """Get current MTF analysis."""
    if not engine:
        return {"error": "Engine not started"}
    
    analysis = {}
    for tf in ["H1", "M15", "M5"]:
        tf_analysis = engine._analysis.get(tf)
        if tf_analysis:
            analysis[tf] = {
                "trend": tf_analysis.trend.value,
                "trend_strength": round(tf_analysis.trend_strength, 2),
                "ema_fast": round(tf_analysis.ema_fast, 2),
                "ema_slow": round(tf_analysis.ema_slow, 2),
                "rsi": round(tf_analysis.rsi, 1),
                "atr": round(tf_analysis.atr, 2),
                "momentum": round(tf_analysis.momentum, 2),
                "volume_trend": tf_analysis.volume_trend,
            }
    
    return {
        "analysis": analysis,
        "aligned": engine._check_trend_alignment() is not None if engine else False,
        "key_levels": engine.get_key_levels() if engine else [],
    }


@app.get("/api/settings")
async def get_settings_route():
    """Get current engine settings."""
    return {
        "ema_fast": settings.ema_fast,
        "ema_slow": settings.ema_slow,
        "ema_trend": settings.ema_trend,
        "rsi_period": settings.rsi_period,
        "rsi_overbought": settings.rsi_overbought,
        "rsi_oversold": settings.rsi_oversold,
        "atr_period": settings.atr_period,
        "min_tp_points": settings.min_tp_points,
        "max_sl_points": settings.max_sl_points,
        "min_tp_pips": settings.min_tp_points * 10,
        "max_sl_pips": settings.max_sl_points * 10,
        "max_spread": settings.max_spread,
        "min_signal_confidence": settings.min_signal_confidence,
        "cooldown_minutes": settings.cooldown_minutes,
        "candle_update_seconds": settings.candle_update_seconds,
    }


@app.post("/api/refresh-candles")
async def refresh_candles():
    """Manually trigger candle refresh."""
    if not engine:
        return {"error": "Engine not started"}
    
    await engine.update_all_candles()
    analysis = engine.analyze_all_timeframes()
    
    return {
        "success": True,
        "timeframes_updated": list(analysis.keys()),
    }


# Backtesting state
backtest_engine: BacktestEngine | None = None
backtest_result: dict | None = None
backtest_running: bool = False


@app.post("/api/backtest")
async def run_backtest(request: dict):
    """Run backtesting for specified date range."""
    global backtest_engine, backtest_result, backtest_running
    
    if backtest_running:
        return {"error": "Backtest already running", "running": True}
    
    try:
        from datetime import datetime
        
        # Parse dates
        start_date_str = request.get("start_date")
        end_date_str = request.get("end_date")
        
        if not start_date_str or not end_date_str:
            return {"error": "start_date and end_date are required"}
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        if start_date >= end_date:
            return {"error": "start_date must be before end_date"}
        
        # Get optional parameters
        initial_balance = request.get("initial_balance", 10000.0)
        risk_per_trade = request.get("risk_per_trade", 0.02)
        
        backtest_running = True
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Create backtest engine
        backtest_engine = BacktestEngine(
            symbol="XAUUSD",
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            settings=settings,
        )
        
        # Run backtest in executor to not block
        import concurrent.futures
        loop = asyncio.get_event_loop()
        
        def run_sync():
            return backtest_engine.run_backtest(start_date, end_date)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, run_sync)
        
        # Convert result to dict
        backtest_result = {
            "success": True,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_balance": result.initial_balance,
            "final_balance": result.final_balance,
            "total_return": result.total_return,
            "total_return_pct": result.total_return_pct,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "profit_factor": result.profit_factor,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "avg_trade_duration": str(result.avg_trade_duration) if result.avg_trade_duration else None,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "consecutive_wins": result.consecutive_wins,
            "consecutive_losses": result.consecutive_losses,
            "monthly_breakdown": result.monthly_breakdown,
            "buy_hold_return": result.buy_hold_return,
            "buy_hold_pct": result.buy_hold_pct,
            "trades": [
                {
                    "id": t.id,
                    "signal_type": t.signal_type.value,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                    "max_favorable": t.max_favorable,
                    "max_adverse": t.max_adverse,
                }
                for t in result.trades[:200]  # Limit to 200 trades for response size
            ],
            "equity_curve": result.equity_curve[-500:] if len(result.equity_curve) > 500 else result.equity_curve,
        }
        
        backtest_running = False
        logger.info(f"Backtest completed: {result.total_trades} trades, {result.total_return_pct:.2f}% return")
        
        return backtest_result
        
    except Exception as e:
        backtest_running = False
        logger.error(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False}


@app.get("/api/backtest/status")
async def get_backtest_status():
    """Get backtest status."""
    return {
        "running": backtest_running,
        "has_result": backtest_result is not None,
    }


@app.get("/api/backtest/result")
async def get_backtest_result():
    """Get last backtest result."""
    if backtest_result is None:
        return {"error": "No backtest result available"}
    return backtest_result


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(active_connections)}")

    try:
        # Send initial state
        await websocket.send_json({
            "event": "connected",
            "data": {
                "prices": list(price_history)[-100:],
                "signals": list(signal_history)[-20:],
                "stats": engine.get_stats() if engine else None,
                "indicators": engine.get_current_indicators() if engine else None,
                "key_levels": engine.get_key_levels()[:5] if engine else [],
            }
        })

        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(active_connections)}")

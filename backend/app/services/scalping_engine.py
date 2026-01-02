"""
XAUUSD Scalping Signal Engine

Generates BUY/SELL signals based on multiple technical indicators:
- EMA crossovers (fast/slow)
- RSI overbought/oversold
- Price momentum
- Spread filter
- ATR-based take profit and stop loss
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

import numpy as np
from loguru import logger


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT_BUY = "EXIT_BUY"
    EXIT_SELL = "EXIT_SELL"


@dataclass
class PriceTick:
    timestamp: datetime
    bid: float
    ask: float
    volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class ScalpingSignal:
    signal_type: SignalType
    entry_price: float
    take_profit: float | None = None
    stop_loss: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 0.0
    reason: str = ""


@dataclass
class ActiveTrade:
    signal_type: SignalType  # BUY or SELL
    entry_price: float
    entry_time: datetime
    take_profit: float
    stop_loss: float
    lot_size: float = 0.01

    def current_pnl_points(self, current_price: float) -> float:
        if self.signal_type == SignalType.BUY:
            return current_price - self.entry_price
        else:
            return self.entry_price - current_price

    def current_pnl_usd(self, current_price: float, point_value: float = 0.1) -> float:
        """For XAUUSD, 1 point = $0.1 per 0.01 lot"""
        return self.current_pnl_points(current_price) * point_value * (self.lot_size / 0.01)


class ScalpingEngine:
    def __init__(
        self,
        ema_fast: int = 8,
        ema_slow: int = 21,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        atr_period: int = 14,
        atr_tp_multiplier: float = 2.0,
        atr_sl_multiplier: float = 1.5,
        max_spread: float = 0.40,  # Max allowed spread in points
        min_confidence: float = 0.7,  # Higher threshold for better signals
        lookback_size: int = 200,
        cooldown_ticks: int = 50,  # Wait ticks between trades
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.atr_tp_multiplier = atr_tp_multiplier
        self.atr_sl_multiplier = atr_sl_multiplier
        self.max_spread = max_spread
        self.min_confidence = min_confidence
        self.cooldown_ticks = cooldown_ticks

        # Price history
        self._prices: deque[PriceTick] = deque(maxlen=lookback_size)
        self._highs: deque[float] = deque(maxlen=lookback_size)
        self._lows: deque[float] = deque(maxlen=lookback_size)
        self._closes: deque[float] = deque(maxlen=lookback_size)

        # Cooldown tracking
        self._ticks_since_trade = 0

        # Active trade tracking
        self.active_trade: ActiveTrade | None = None

        # Signal history
        self.signal_history: deque[ScalpingSignal] = deque(maxlen=50)
        self.trade_history: list[dict] = []

        # Callbacks
        self._on_signal: Callable[[ScalpingSignal], None] | None = None
        self._on_trade_close: Callable[[dict], None] | None = None

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    def set_signal_callback(self, callback: Callable[[ScalpingSignal], None]) -> None:
        self._on_signal = callback

    def set_trade_close_callback(self, callback: Callable[[dict], None]) -> None:
        self._on_trade_close = callback

    def add_tick(self, tick: PriceTick) -> ScalpingSignal | None:
        """Process a new tick and generate signal if conditions met."""
        self._prices.append(tick)
        self._closes.append(tick.mid)
        self._highs.append(tick.ask)
        self._lows.append(tick.bid)

        # Increment cooldown counter
        self._ticks_since_trade += 1

        # Check for exit signals first if we have an active trade
        if self.active_trade:
            exit_signal = self._check_exit(tick)
            if exit_signal:
                self._close_trade(tick, exit_signal.reason)
                self.signal_history.append(exit_signal)
                if self._on_signal:
                    self._on_signal(exit_signal)
                return exit_signal

        # Need enough data for indicators
        if len(self._closes) < self.ema_slow + 10:
            return None

        # Check cooldown
        if self._ticks_since_trade < self.cooldown_ticks:
            return None

        # Check for entry signals
        signal = self._check_entry(tick)
        if signal and signal.signal_type in (SignalType.BUY, SignalType.SELL):
            self._open_trade(signal)
            self.signal_history.append(signal)
            if self._on_signal:
                self._on_signal(signal)
            return signal

        return None

    def _check_exit(self, tick: PriceTick) -> ScalpingSignal | None:
        """Check if active trade should be closed."""
        if not self.active_trade:
            return None

        trade = self.active_trade
        current_price = tick.bid if trade.signal_type == SignalType.BUY else tick.ask

        # Check take profit
        if trade.signal_type == SignalType.BUY:
            if current_price >= trade.take_profit:
                return ScalpingSignal(
                    signal_type=SignalType.EXIT_BUY,
                    entry_price=current_price,
                    confidence=1.0,
                    reason="Take Profit hit",
                    timestamp=tick.timestamp,
                )
            if current_price <= trade.stop_loss:
                return ScalpingSignal(
                    signal_type=SignalType.EXIT_BUY,
                    entry_price=current_price,
                    confidence=1.0,
                    reason="Stop Loss hit",
                    timestamp=tick.timestamp,
                )
        else:  # SELL trade
            if current_price <= trade.take_profit:
                return ScalpingSignal(
                    signal_type=SignalType.EXIT_SELL,
                    entry_price=current_price,
                    confidence=1.0,
                    reason="Take Profit hit",
                    timestamp=tick.timestamp,
                )
            if current_price >= trade.stop_loss:
                return ScalpingSignal(
                    signal_type=SignalType.EXIT_SELL,
                    entry_price=current_price,
                    confidence=1.0,
                    reason="Stop Loss hit",
                    timestamp=tick.timestamp,
                )

        # Check for reversal signal (optional early exit)
        closes = np.array(self._closes)
        ema_fast = self._ema(closes, self.ema_fast)
        ema_slow = self._ema(closes, self.ema_slow)

        if trade.signal_type == SignalType.BUY and ema_fast < ema_slow:
            # EMA crossed down while in BUY
            pnl = trade.current_pnl_points(current_price)
            if pnl > 0:  # Only exit early if in profit
                return ScalpingSignal(
                    signal_type=SignalType.EXIT_BUY,
                    entry_price=current_price,
                    confidence=0.8,
                    reason="EMA reversal (profit lock)",
                    timestamp=tick.timestamp,
                )

        if trade.signal_type == SignalType.SELL and ema_fast > ema_slow:
            pnl = trade.current_pnl_points(current_price)
            if pnl > 0:
                return ScalpingSignal(
                    signal_type=SignalType.EXIT_SELL,
                    entry_price=current_price,
                    confidence=0.8,
                    reason="EMA reversal (profit lock)",
                    timestamp=tick.timestamp,
                )

        return None

    def _check_entry(self, tick: PriceTick) -> ScalpingSignal | None:
        """Check for entry signal with strong trend confirmation."""
        # Don't enter if already in a trade
        if self.active_trade:
            return None

        # Spread filter
        if tick.spread > self.max_spread:
            return None

        closes = np.array(self._closes)
        highs = np.array(self._highs)
        lows = np.array(self._lows)

        # Calculate indicators
        ema_fast = self._ema(closes, self.ema_fast)
        ema_slow = self._ema(closes, self.ema_slow)
        ema_fast_prev = self._ema(closes[:-1], self.ema_fast)
        ema_slow_prev = self._ema(closes[:-1], self.ema_slow)

        # Calculate trend strength (EMA separation)
        ema_diff = abs(ema_fast - ema_slow)
        atr = self._atr(highs, lows, closes, self.atr_period)
        trend_strength = ema_diff / atr if atr > 0 else 0

        rsi = self._rsi(closes, self.rsi_period)

        # Momentum (price change over last 10 ticks for more stability)
        momentum = closes[-1] - closes[-10] if len(closes) >= 10 else 0

        # Higher highs / lower lows check (trend quality)
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        uptrend_quality = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1]) / 9
        downtrend_quality = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1]) / 9

        # Signal confidence scoring
        buy_score = 0.0
        sell_score = 0.0
        buy_reasons = []
        sell_reasons = []

        # EMA crossover (most important) - fresh crossover within last 3 ticks
        ema_fast_2 = self._ema(closes[:-2], self.ema_fast)
        ema_slow_2 = self._ema(closes[:-2], self.ema_slow)
        
        fresh_bullish_cross = (
            ema_fast > ema_slow and 
            (ema_fast_prev <= ema_slow_prev or ema_fast_2 <= ema_slow_2)
        )
        fresh_bearish_cross = (
            ema_fast < ema_slow and 
            (ema_fast_prev >= ema_slow_prev or ema_fast_2 >= ema_slow_2)
        )

        if fresh_bullish_cross:
            buy_score += 0.35
            buy_reasons.append("EMA bullish cross")
        if fresh_bearish_cross:
            sell_score += 0.35
            sell_reasons.append("EMA bearish cross")

        # Trend alignment (EMA positioning)
        if ema_fast > ema_slow and trend_strength > 0.3:
            buy_score += 0.15
            buy_reasons.append("Strong uptrend")
        elif ema_fast < ema_slow and trend_strength > 0.3:
            sell_score += 0.15
            sell_reasons.append("Strong downtrend")

        # RSI conditions (extreme levels)
        if rsi < 25:
            buy_score += 0.25
            buy_reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 35 and momentum > 0:
            buy_score += 0.15
            buy_reasons.append(f"RSI low ({rsi:.0f})")
        
        if rsi > 75:
            sell_score += 0.25
            sell_reasons.append(f"RSI overbought ({rsi:.0f})")
        elif rsi > 65 and momentum < 0:
            sell_score += 0.15
            sell_reasons.append(f"RSI high ({rsi:.0f})")

        # Momentum confirmation (strong directional move)
        if momentum > atr * 0.5:
            buy_score += 0.15
            buy_reasons.append("Strong momentum")
        elif momentum < -atr * 0.5:
            sell_score += 0.15
            sell_reasons.append("Strong momentum")

        # Trend quality bonus
        if uptrend_quality > 0.6:
            buy_score += 0.1
            buy_reasons.append("Higher highs pattern")
        if downtrend_quality > 0.6:
            sell_score += 0.1
            sell_reasons.append("Lower lows pattern")

        # Price position relative to EMAs
        if closes[-1] > ema_fast > ema_slow:
            buy_score += 0.05
        elif closes[-1] < ema_fast < ema_slow:
            sell_score += 0.05

        # Generate signal if confidence threshold met
        if buy_score >= self.min_confidence:
            tp = tick.ask + atr * self.atr_tp_multiplier
            sl = tick.ask - atr * self.atr_sl_multiplier
            return ScalpingSignal(
                signal_type=SignalType.BUY,
                entry_price=tick.ask,
                take_profit=tp,
                stop_loss=sl,
                confidence=min(buy_score, 1.0),
                reason=" | ".join(buy_reasons),
                timestamp=tick.timestamp,
            )

        if sell_score >= self.min_confidence:
            tp = tick.bid - atr * self.atr_tp_multiplier
            sl = tick.bid + atr * self.atr_sl_multiplier
            return ScalpingSignal(
                signal_type=SignalType.SELL,
                entry_price=tick.bid,
                take_profit=tp,
                stop_loss=sl,
                confidence=min(sell_score, 1.0),
                reason=" | ".join(sell_reasons),
                timestamp=tick.timestamp,
            )

        return None

    def _open_trade(self, signal: ScalpingSignal) -> None:
        """Open a new trade."""
        self.active_trade = ActiveTrade(
            signal_type=signal.signal_type,
            entry_price=signal.entry_price,
            entry_time=signal.timestamp,
            take_profit=signal.take_profit or signal.entry_price,
            stop_loss=signal.stop_loss or signal.entry_price,
        )
        self._ticks_since_trade = 0  # Reset cooldown
        logger.info(
            f"TRADE OPENED: {signal.signal_type.value} @ {signal.entry_price:.2f} "
            f"TP: {signal.take_profit:.2f} SL: {signal.stop_loss:.2f} "
            f"[{signal.reason}]"
        )

    def _close_trade(self, tick: PriceTick, reason: str) -> None:
        """Close the active trade and record results."""
        if not self.active_trade:
            return

        trade = self.active_trade
        exit_price = tick.bid if trade.signal_type == SignalType.BUY else tick.ask
        pnl_points = trade.current_pnl_points(exit_price)
        pnl_usd = trade.current_pnl_usd(exit_price)

        self.total_trades += 1
        self.total_pnl += pnl_points
        if pnl_points > 0:
            self.winning_trades += 1

        trade_record = {
            "type": trade.signal_type.value,
            "entry_price": trade.entry_price,
            "exit_price": exit_price,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": tick.timestamp.isoformat(),
            "pnl_points": round(pnl_points, 2),
            "pnl_usd": round(pnl_usd, 2),
            "reason": reason,
            "won": pnl_points > 0,
        }
        self.trade_history.append(trade_record)

        logger.info(
            f"TRADE CLOSED: {trade.signal_type.value} @ {exit_price:.2f} "
            f"PnL: {pnl_points:+.2f} pts (${pnl_usd:+.2f}) - {reason}"
        )

        if self._on_trade_close:
            self._on_trade_close(trade_record)

        self.active_trade = None

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0.0
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return float(np.convolve(data, weights, mode='valid')[-1])

    @staticmethod
    def _rsi(data: np.ndarray, period: int) -> float:
        """Calculate RSI."""
        if len(data) < period + 1:
            return 50.0
        deltas = np.diff(data[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        """Calculate ATR."""
        if len(closes) < period + 1:
            return float(highs[-1] - lows[-1]) if len(highs) > 0 else 0.5
        tr_list = []
        for i in range(-period, 0):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            tr_list.append(max(high_low, high_close, low_close))
        return float(np.mean(tr_list))

    def get_stats(self) -> dict:
        """Get trading statistics."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.total_trades - self.winning_trades,
            "win_rate": round(win_rate, 1),
            "total_pnl_points": round(self.total_pnl, 2),
            "active_trade": self.active_trade is not None,
        }

    def get_current_indicators(self) -> dict | None:
        """Get current indicator values for display."""
        if len(self._closes) < self.ema_slow + 5:
            return None

        closes = np.array(self._closes)
        highs = np.array(self._highs)
        lows = np.array(self._lows)

        return {
            "ema_fast": round(self._ema(closes, self.ema_fast), 2),
            "ema_slow": round(self._ema(closes, self.ema_slow), 2),
            "rsi": round(self._rsi(closes, self.rsi_period), 1),
            "atr": round(self._atr(highs, lows, closes, self.atr_period), 2),
            "last_price": round(closes[-1], 2),
        }

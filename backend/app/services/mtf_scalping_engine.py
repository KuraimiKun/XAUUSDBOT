"""
Advanced Sunrise EMA Crossover Strategy for XAUUSD

A sophisticated 4-phase volatility expansion entry system optimized for gold trading.

Key Features:
- 4-Phase State Machine: SCANNING → ARMED → WINDOW_OPEN → BREAKOUT
- Multi-EMA crossover system (fast/medium/slow/filter EMAs)
- ATR-based volatility filtering with increment/decrement thresholds
- Pullback confirmation before entry (3 red for LONG, 2 green for SHORT)
- Dynamic SL/TP based on ATR multipliers
- Price filter EMA requirement
- Time range filter (00:00-08:00 UTC)
- Angle filter for EMA slope validation
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Optional, List, Dict
import numpy as np
from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT_BUY = "EXIT_BUY"
    EXIT_SELL = "EXIT_SELL"


class TrendDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Timeframe(Enum):
    M1 = (mt5.TIMEFRAME_M1 if mt5 else 1, "M1", 1)
    M5 = (mt5.TIMEFRAME_M5 if mt5 else 5, "M5", 5)
    M15 = (mt5.TIMEFRAME_M15 if mt5 else 15, "M15", 15)
    M30 = (mt5.TIMEFRAME_M30 if mt5 else 30, "M30", 30)
    H1 = (mt5.TIMEFRAME_H1 if mt5 else 60, "H1", 60)
    H4 = (mt5.TIMEFRAME_H4 if mt5 else 240, "H4", 240)
    D1 = (mt5.TIMEFRAME_D1 if mt5 else 1440, "D1", 1440)

    @property
    def mt5_tf(self):
        return self.value[0]

    @property
    def name_str(self):
        return self.value[1]

    @property
    def minutes(self):
        return self.value[2]


@dataclass
class Candle:
    """OHLC Candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float = 0.0

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def range(self) -> float:
        return self.high - self.low


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
class KeyLevel:
    """Support/Resistance level"""
    price: float
    level_type: str  # "support" or "resistance"
    strength: int  # Number of touches
    timeframe: str


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe"""
    timeframe: str
    trend: TrendDirection
    trend_strength: float  # 0-1
    ema_fast: float
    ema_slow: float
    ema_trend: float  # 200 EMA for higher TF
    rsi: float
    atr: float
    last_close: float
    momentum: float
    volume_trend: str  # "increasing", "decreasing", "stable"
    key_levels: List[KeyLevel]
    order_block: Optional[Dict] = None


@dataclass
class TradeSetup:
    """Complete trade setup with all analysis"""
    signal_type: SignalType
    entry_price: float
    take_profit: float
    stop_loss: float
    risk_reward: float
    confidence: float
    timestamp: datetime
    
    # Analysis details
    h1_trend: TrendDirection
    m15_trend: TrendDirection
    m5_trend: TrendDirection
    
    entry_reason: str
    confluence_factors: List[str]
    
    # Risk management
    pip_target: float  # Expected pips
    pip_risk: float    # Risk in pips


@dataclass
class ActiveTrade:
    """Currently active trade"""
    signal_type: SignalType
    entry_price: float
    entry_time: datetime
    take_profit: float
    stop_loss: float
    original_sl: float  # For trailing
    lot_size: float = 0.01
    
    # Tracking
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    trailing_active: bool = False

    def current_pnl_points(self, current_price: float) -> float:
        if self.signal_type == SignalType.BUY:
            return current_price - self.entry_price
        else:
            return self.entry_price - current_price


class MTFScalpingEngine:
    """
    Advanced Sunrise EMA Crossover Strategy
    
    Strategy Overview:
    A 4-phase volatility expansion entry system:
    
    Phase 1 - SCANNING:
        Wait for EMA crossover signal with candle direction filter
        
    Phase 2 - ARMED:
        Count pullback candles (3 red for LONG, 2 green for SHORT)
        Store ATR at signal detection
        
    Phase 3 - WINDOW_OPEN:
        Calculate breakout window limits with price offset
        Set expiry bar count
        
    Phase 4 - BREAKOUT:
        Monitor for price break of window limits
        Execute trade on confirmed breakout
    
    Exit Criteria:
    - Take Profit hit (ATR-based: LONG=6.5x, SHORT=6.5x)
    - Stop Loss hit (ATR-based: LONG=4.5x, SHORT=2.5x)
    - Trailing stop after 60% of target reached
    """
    
    # XAUUSD specifics: 1 pip = 0.1 point, 1 point = $0.1 per 0.01 lot
    PIP_VALUE = 0.1
    MIN_RR_RATIO = 1.0
    
    # Entry state machine states
    STATE_SCANNING = "SCANNING"
    STATE_ARMED_LONG = "ARMED_LONG"
    STATE_ARMED_SHORT = "ARMED_SHORT"
    STATE_WINDOW_OPEN_LONG = "WINDOW_OPEN_LONG"
    STATE_WINDOW_OPEN_SHORT = "WINDOW_OPEN_SHORT"
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        # === SUNRISE EMA PERIODS ===
        ema_confirm: int = 1,      # Confirmation EMA (fastest)
        ema_fast: int = 14,        # Fast EMA
        ema_medium: int = 14,      # Medium EMA (same as fast for this config)
        ema_slow: int = 24,        # Slow EMA
        ema_filter: int = 100,     # Price filter EMA (trend filter)
        ema_exit: int = 25,        # Exit EMA
        ema_trend: int = 50,       # Legacy param for compatibility
        # === RSI (legacy, not used in Sunrise) ===
        rsi_period: int = 14,
        rsi_ob: float = 70.0,
        rsi_os: float = 30.0,
        # === ATR SETTINGS ===
        atr_period: int = 14,
        atr_min: float = 0.0,      # Minimum ATR threshold
        atr_max: float = 999.0,    # Maximum ATR threshold
        atr_increment_filter: float = 0.5,    # Max ATR increase per bar
        atr_decrement_filter: float = -0.5,   # Max ATR decrease per bar
        # === SL/TP MULTIPLIERS (ATR-based) ===
        sl_mult_long: float = 1.2,    # Wider SL for fewer stop-outs
        tp_mult_long: float = 2.4,    # 1:2 R:R ratio  
        sl_mult_short: float = 1.2,   # Same for shorts
        tp_mult_short: float = 2.4,   # 1:2 R:R ratio
        # === H1 TREND FILTER ===
        use_h1_trend_filter: bool = True,  # Only trade in H1 trend direction
        # === CONSECUTIVE LOSS HANDLING ===
        max_consecutive_losses: int = 3,   # Max consecutive losses before extended cooldown
        extended_cooldown_minutes: int = 60,  # Cooldown after max consecutive losses
        # === PULLBACK SETTINGS ===
        pullback_candles_long: int = 2,   # Reduced from 3 for more signals
        pullback_candles_short: int = 2,  # Green candles needed for SHORT setup
        # === ENTRY WINDOW SETTINGS ===
        window_period_long: int = 5,      # Increased from 1 - gives 5 bars to catch breakout
        window_period_short: int = 5,     # Reduced from 7 for consistency
        window_price_offset: float = 0.5, # Increased for gold volatility
        # === TIME FILTER (UTC hours) ===
        time_filter_start: int = 0,   # Start hour (00:00 UTC)
        time_filter_end: int = 20,    # Expanded to 20 hours (00:00-20:00 UTC)
        # === ANGLE FILTER ===
        angle_filter_enabled: bool = False,  # Disabled for more signals
        angle_threshold: float = 0.2,  # Minimum EMA slope angle
        # === TRADE MANAGEMENT ===
        max_spread: float = 0.50,
        min_confidence: float = 0.70,  # Legacy, not used in Sunrise
        cooldown_minutes: int = 5,
        candle_lookback: int = 200,
        # Legacy params
        min_tp_points: float = 3.0,
        max_sl_points: float = 2.0,
    ) -> None:
        self.symbol = symbol
        
        # EMA periods
        self.ema_confirm = ema_confirm
        self.ema_fast = ema_fast
        self.ema_medium = ema_medium
        self.ema_slow = ema_slow
        self.ema_filter = ema_filter
        self.ema_exit = ema_exit
        self.ema_trend = ema_trend
        
        # RSI (legacy)
        self.rsi_period = rsi_period
        self.rsi_ob = rsi_ob
        self.rsi_os = rsi_os
        
        # ATR settings
        self.atr_period = atr_period
        self.atr_min = atr_min
        self.atr_max = atr_max
        self.atr_increment_filter = atr_increment_filter
        self.atr_decrement_filter = atr_decrement_filter
        
        # SL/TP multipliers
        self.sl_mult_long = sl_mult_long
        self.tp_mult_long = tp_mult_long
        self.sl_mult_short = sl_mult_short
        self.tp_mult_short = tp_mult_short
        
        # H1 trend filter
        self.use_h1_trend_filter = use_h1_trend_filter
        
        # Consecutive loss handling
        self.max_consecutive_losses = max_consecutive_losses
        self.extended_cooldown_minutes = extended_cooldown_minutes
        self._consecutive_losses = 0
        self._last_trade_won = True
        
        # Pullback settings
        self.pullback_candles_long = pullback_candles_long
        self.pullback_candles_short = pullback_candles_short
        
        # Entry window settings
        self.window_period_long = window_period_long
        self.window_period_short = window_period_short
        self.window_price_offset = window_price_offset
        
        # Time filter
        self.time_filter_start = time_filter_start
        self.time_filter_end = time_filter_end
        
        # Angle filter
        self.angle_filter_enabled = angle_filter_enabled
        self.angle_threshold = angle_threshold
        
        # Trade management
        self.max_spread = max_spread
        self.min_confidence = min_confidence
        self.cooldown_minutes = cooldown_minutes
        self.candle_lookback = candle_lookback
        self.min_tp_points = min_tp_points
        self.max_sl_points = max_sl_points
        
        # === STATE MACHINE VARIABLES ===
        self._entry_state = self.STATE_SCANNING
        self._pullback_candle_count = 0
        self._signal_detection_atr = 0.0
        self._window_top_limit = 0.0
        self._window_bottom_limit = 0.0
        self._window_expiry_bar = 0
        self._bar_count = 0  # Track bar count for window expiry
        self._last_bar_time: Optional[datetime] = None
        
        # Candle storage per timeframe (Sunrise uses M15 primarily)
        self._candles: Dict[str, List[Candle]] = {
            "M5": [],
            "M15": [],
            "H1": [],
        }
        
        # EMA values storage (computed on each analysis)
        self._ema_values: Dict[str, float] = {}
        
        # Current tick
        self._current_tick: Optional[PriceTick] = None
        
        # Last analysis per timeframe
        self._analysis: Dict[str, TimeframeAnalysis] = {}
        
        # Key levels (support/resistance)
        self._key_levels: List[KeyLevel] = []
        
        # Trade tracking
        self.active_trade: Optional[ActiveTrade] = None
        self._last_trade_time: Optional[datetime] = None
        
        # History
        self.signal_history: deque[TradeSetup] = deque(maxlen=50)
        self.trade_history: list[dict] = []
        
        # Callbacks
        self._on_signal: Optional[Callable[[TradeSetup], None]] = None
        self._on_trade_close: Optional[Callable[[dict], None]] = None
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        logger.info(
            f"Sunrise Strategy Engine initialized - "
            f"EMAs: {ema_fast}/{ema_medium}/{ema_slow}/{ema_filter} | "
            f"SL/TP LONG: {sl_mult_long}x/{tp_mult_long}x ATR | "
            f"SL/TP SHORT: {sl_mult_short}x/{tp_mult_short}x ATR | "
            f"Time Filter: {time_filter_start:02d}:00-{time_filter_end:02d}:00 UTC"
        )

    def set_signal_callback(self, callback: Callable[[TradeSetup], None]) -> None:
        self._on_signal = callback

    def set_trade_close_callback(self, callback: Callable[[dict], None]) -> None:
        self._on_trade_close = callback

    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    async def fetch_candles(self, timeframe: Timeframe, count: int = 100) -> List[Candle]:
        """Fetch OHLC candles from MT5"""
        if mt5 is None:
            logger.warning("MT5 not available")
            return []
        
        def _fetch():
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe.mt5_tf, 0, count)
            if rates is None:
                return []
            
            candles = []
            for r in rates:
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(r['time'], tz=timezone.utc),
                    open=r['open'],
                    high=r['high'],
                    low=r['low'],
                    close=r['close'],
                    volume=r['real_volume'] if 'real_volume' in r.dtype.names else r['tick_volume'],
                    spread=r['spread'] if 'spread' in r.dtype.names else 0,
                ))
            return candles
        
        try:
            return await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.error(f"Failed to fetch {timeframe.name_str} candles: {e}")
            return []

    async def update_all_candles(self) -> None:
        """Fetch fresh candles for all timeframes"""
        tasks = [
            self.fetch_candles(Timeframe.M5, self.candle_lookback),
            self.fetch_candles(Timeframe.M15, self.candle_lookback),
            self.fetch_candles(Timeframe.H1, self.candle_lookback),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for tf_name, result in zip(["M5", "M15", "H1"], results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {tf_name}: {result}")
            elif result:
                self._candles[tf_name] = result
                logger.debug(f"Updated {tf_name}: {len(result)} candles")

    async def fetch_tick(self) -> Optional[PriceTick]:
        """Fetch current tick from MT5"""
        if mt5 is None:
            return None
        
        def _fetch():
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None
            return PriceTick(
                timestamp=datetime.fromtimestamp(tick.time, tz=timezone.utc),
                bid=tick.bid,
                ask=tick.ask,
                volume=tick.volume,
            )
        
        try:
            tick = await asyncio.to_thread(_fetch)
            if tick:
                self._current_tick = tick
            return tick
        except Exception as e:
            logger.error(f"Failed to fetch tick: {e}")
            return None

    # =========================================================================
    # TECHNICAL ANALYSIS
    # =========================================================================
    
    def _calculate_ema(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA series"""
        if len(closes) < period:
            return np.full_like(closes, np.nan)
        
        ema = np.zeros_like(closes)
        multiplier = 2 / (period + 1)
        
        # SMA for first value
        ema[period-1] = np.mean(closes[:period])
        
        for i in range(period, len(closes)):
            ema[i] = (closes[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema

    def _calculate_rsi(self, closes: np.ndarray, period: int) -> float:
        """Calculate RSI"""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use exponential moving average for smoothing
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, candles: List[Candle], period: int) -> float:
        """Calculate Average True Range"""
        if len(candles) < period + 1:
            return 0.5  # Default for XAUUSD
        
        tr_values = []
        for i in range(-period, 0):
            high_low = candles[i].high - candles[i].low
            high_close = abs(candles[i].high - candles[i-1].close)
            low_close = abs(candles[i].low - candles[i-1].close)
            tr_values.append(max(high_low, high_close, low_close))
        
        return np.mean(tr_values)

    def _find_key_levels(self, candles: List[Candle], timeframe: str) -> List[KeyLevel]:
        """Find support and resistance levels using swing highs/lows"""
        if len(candles) < 20:
            return []
        
        levels = []
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        # Find swing highs (local maxima)
        for i in range(2, len(candles) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                # Count touches of this level (within 0.5 points)
                touches = sum(1 for c in candles if abs(c.high - highs[i]) < 0.5)
                if touches >= 2:
                    levels.append(KeyLevel(
                        price=highs[i],
                        level_type="resistance",
                        strength=touches,
                        timeframe=timeframe,
                    ))
        
        # Find swing lows (local minima)
        for i in range(2, len(candles) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                touches = sum(1 for c in candles if abs(c.low - lows[i]) < 0.5)
                if touches >= 2:
                    levels.append(KeyLevel(
                        price=lows[i],
                        level_type="support",
                        strength=touches,
                        timeframe=timeframe,
                    ))
        
        # Sort by strength and remove duplicates (within 1 point)
        levels.sort(key=lambda x: x.strength, reverse=True)
        
        filtered = []
        for level in levels:
            if not any(abs(l.price - level.price) < 1.0 for l in filtered):
                filtered.append(level)
        
        return filtered[:10]  # Top 10 levels

    def _find_order_block(self, candles: List[Candle]) -> Optional[Dict]:
        """
        Find recent order block (last strong candle before reversal)
        Order blocks are zones where large orders were placed
        """
        if len(candles) < 10:
            return None
        
        # Look for bullish order block (last bearish candle before strong move up)
        for i in range(-10, -2):
            if candles[i].is_bearish and candles[i].body_size > 0.3:
                # Check if followed by strong bullish move
                if all(candles[j].is_bullish for j in range(i+1, min(i+4, -1))):
                    move_size = candles[-1].close - candles[i].close
                    if move_size > 1.0:  # 10+ pips move
                        return {
                            "type": "bullish",
                            "high": candles[i].high,
                            "low": candles[i].low,
                            "timestamp": candles[i].timestamp,
                        }
        
        # Look for bearish order block
        for i in range(-10, -2):
            if candles[i].is_bullish and candles[i].body_size > 0.3:
                if all(candles[j].is_bearish for j in range(i+1, min(i+4, -1))):
                    move_size = candles[i].close - candles[-1].close
                    if move_size > 1.0:
                        return {
                            "type": "bearish",
                            "high": candles[i].high,
                            "low": candles[i].low,
                            "timestamp": candles[i].timestamp,
                        }
        
        return None

    def _is_price_near_recent_swing(
        self,
        trend: TrendDirection,
        atr: float,
        lookback: int = 40,
    ) -> bool:
        """Validate that current price is reacting at a recent swing level"""
        candles = self._candles.get("M5", [])
        if len(candles) < lookback + 5 or atr <= 0:
            return False

        recent_window = candles[-lookback:]
        current = recent_window[-1]
        prior = recent_window[:-1]
        if not prior:
            return False

        recent_high = max(c.high for c in prior)
        recent_low = min(c.low for c in prior)
        price = current.close
        buffer = atr * 1.2
        safety_gap = atr * 0.8

        if trend == TrendDirection.BULLISH:
            return (price - recent_low) <= buffer and (recent_high - price) >= safety_gap
        if trend == TrendDirection.BEARISH:
            return (recent_high - price) <= buffer and (price - recent_low) >= safety_gap
        return False

    def _analyze_volume_trend(self, candles: List[Candle]) -> str:
        """Analyze volume trend"""
        if len(candles) < 20:
            return "stable"
        
        recent_vol = np.mean([c.volume for c in candles[-5:]])
        older_vol = np.mean([c.volume for c in candles[-20:-5]])
        
        if older_vol == 0:
            return "stable"
        
        ratio = recent_vol / older_vol
        if ratio > 1.3:
            return "increasing"
        elif ratio < 0.7:
            return "decreasing"
        return "stable"

    def analyze_timeframe(self, timeframe: str) -> Optional[TimeframeAnalysis]:
        """Complete analysis for a single timeframe"""
        candles = self._candles.get(timeframe, [])
        if len(candles) < self.ema_trend + 10:
            return None
        
        closes = np.array([c.close for c in candles])
        
        # EMAs
        ema_fast = self._calculate_ema(closes, self.ema_fast)
        ema_slow = self._calculate_ema(closes, self.ema_slow)
        ema_trend = self._calculate_ema(closes, self.ema_trend)
        
        last_ema_fast = ema_fast[-1]
        last_ema_slow = ema_slow[-1]
        last_ema_trend = ema_trend[-1]
        
        # Determine trend
        if last_ema_fast > last_ema_slow > last_ema_trend:
            trend = TrendDirection.BULLISH
        elif last_ema_fast < last_ema_slow < last_ema_trend:
            trend = TrendDirection.BEARISH
        else:
            trend = TrendDirection.NEUTRAL
        
        # Trend strength (EMA separation relative to ATR)
        atr = self._calculate_atr(candles, self.atr_period)
        ema_separation = abs(last_ema_fast - last_ema_slow)
        trend_strength = min(ema_separation / atr, 1.0) if atr > 0 else 0.5
        
        # RSI
        rsi = self._calculate_rsi(closes, self.rsi_period)
        
        # Momentum (close vs 10 bars ago)
        momentum = closes[-1] - closes[-10] if len(closes) >= 10 else 0
        
        # Key levels
        key_levels = self._find_key_levels(candles, timeframe)
        
        # Order block
        order_block = self._find_order_block(candles)
        
        # Volume trend
        volume_trend = self._analyze_volume_trend(candles)
        
        analysis = TimeframeAnalysis(
            timeframe=timeframe,
            trend=trend,
            trend_strength=trend_strength,
            ema_fast=last_ema_fast,
            ema_slow=last_ema_slow,
            ema_trend=last_ema_trend,
            rsi=rsi,
            atr=atr,
            last_close=closes[-1],
            momentum=momentum,
            volume_trend=volume_trend,
            key_levels=key_levels,
            order_block=order_block,
        )
        
        self._analysis[timeframe] = analysis
        return analysis

    def analyze_all_timeframes(self) -> Dict[str, TimeframeAnalysis]:
        """Analyze all timeframes"""
        results = {}
        for tf in ["H1", "M15", "M5"]:
            analysis = self.analyze_timeframe(tf)
            if analysis:
                results[tf] = analysis
        return results

    # =========================================================================
    # SUNRISE STRATEGY - SIGNAL GENERATION
    # =========================================================================
    
    def _check_time_filter(self, timestamp: datetime) -> bool:
        """Check if current time is within allowed trading window"""
        hour = timestamp.hour
        if self.time_filter_start <= self.time_filter_end:
            return self.time_filter_start <= hour < self.time_filter_end
        else:  # Wraps around midnight
            return hour >= self.time_filter_start or hour < self.time_filter_end
    
    def _calculate_ema_angle(self, ema_values: np.ndarray, lookback: int = 5) -> float:
        """Calculate the angle/slope of EMA over lookback period"""
        if len(ema_values) < lookback + 1:
            return 0.0
        recent = ema_values[-lookback:]
        # Simple slope calculation
        return (recent[-1] - recent[0]) / lookback
    
    def _check_atr_filter(self, atr: float, prev_atr: float) -> bool:
        """Check if ATR passes volatility filters"""
        # Min/Max ATR filter
        if atr < self.atr_min or atr > self.atr_max:
            return False
        
        # ATR increment/decrement filter
        if prev_atr > 0:
            atr_change = atr - prev_atr
            if atr_change > self.atr_increment_filter:
                return False  # ATR increased too much
            if atr_change < self.atr_decrement_filter:
                return False  # ATR decreased too much
        
        return True
    
    def _detect_ema_crossover(self, candles: List[Candle]) -> Optional[TrendDirection]:
        """
        Detect EMA crossover for Sunrise strategy.
        Returns BULLISH for buy signal, BEARISH for sell signal, None otherwise.
        """
        if len(candles) < max(self.ema_slow, self.ema_filter) + 5:
            return None
        
        closes = np.array([c.close for c in candles])
        
        # Calculate all EMAs
        ema_confirm = self._calculate_ema(closes, self.ema_confirm)
        ema_fast = self._calculate_ema(closes, self.ema_fast)
        ema_medium = self._calculate_ema(closes, self.ema_medium)
        ema_slow = self._calculate_ema(closes, self.ema_slow)
        ema_filter = self._calculate_ema(closes, self.ema_filter)
        
        # Store current EMA values
        self._ema_values = {
            'confirm': ema_confirm[-1],
            'fast': ema_fast[-1],
            'medium': ema_medium[-1],
            'slow': ema_slow[-1],
            'filter': ema_filter[-1],
        }
        
        # Current bar values
        cur_fast = ema_fast[-1]
        cur_medium = ema_medium[-1]
        cur_slow = ema_slow[-1]
        cur_filter = ema_filter[-1]
        
        # Previous bar values
        prev_fast = ema_fast[-2]
        prev_slow = ema_slow[-2]
        
        # Current candle direction
        current_candle = candles[-1]
        is_bullish_candle = current_candle.is_bullish
        is_bearish_candle = current_candle.is_bearish
        
        # === BULLISH CROSSOVER DETECTION ===
        # Fast EMA crosses above Slow EMA + bullish candle + price above filter EMA
        bullish_cross = (prev_fast <= prev_slow) and (cur_fast > cur_slow)
        if bullish_cross and is_bullish_candle and closes[-1] > cur_filter:
            # Angle filter check
            if self.angle_filter_enabled:
                angle = self._calculate_ema_angle(ema_fast)
                if angle < self.angle_threshold:
                    return None
            return TrendDirection.BULLISH
        
        # === BEARISH CROSSOVER DETECTION ===
        # Fast EMA crosses below Slow EMA + bearish candle + price below filter EMA
        bearish_cross = (prev_fast >= prev_slow) and (cur_fast < cur_slow)
        if bearish_cross and is_bearish_candle and closes[-1] < cur_filter:
            # Angle filter check
            if self.angle_filter_enabled:
                angle = self._calculate_ema_angle(ema_fast)
                if angle > -self.angle_threshold:
                    return None
            return TrendDirection.BEARISH
        
        return None
    
    def _check_trend_alignment(self) -> Optional[TrendDirection]:
        """Legacy method - returns current crossover direction for compatibility"""
        candles = self._candles.get("M15", [])
        if len(candles) < self.ema_filter + 5:
            return None
        return self._detect_ema_crossover(candles)

    def _check_entry_conditions(self, tick: PriceTick) -> Optional[TradeSetup]:
        """
        Simplified EMA Pullback Strategy with H1 Trend Filter
        
        LONG: EMAs aligned bullish (fast > slow > filter) + price pulls back to fast EMA + bullish candle + H1 bullish
        SHORT: EMAs aligned bearish (fast < slow < filter) + price pulls back to fast EMA + bearish candle + H1 bearish
        """
        # Basic filters
        if tick.spread > self.max_spread:
            return None
        
        if self._last_trade_time:
            elapsed = datetime.now(timezone.utc) - self._last_trade_time
            # Extended cooldown after consecutive losses
            cooldown = self.cooldown_minutes
            if self._consecutive_losses >= self.max_consecutive_losses:
                cooldown = self.extended_cooldown_minutes
            if elapsed < timedelta(minutes=cooldown):
                return None
        
        if self.active_trade:
            return None
        
        # Time filter
        if not self._check_time_filter(tick.timestamp):
            return None
        
        candles = self._candles.get("M15", [])
        if len(candles) < max(self.ema_slow, self.ema_filter) + 10:
            return None
        
        m15 = self._analysis.get("M15")
        if not m15:
            return None
        
        current_candle = candles[-1]
        atr = m15.atr
        
        # Get EMA values
        closes = [c.close for c in candles]
        ema_fast = self._calculate_ema(closes, self.ema_fast)
        ema_slow = self._calculate_ema(closes, self.ema_slow)
        ema_filter = self._calculate_ema(closes, self.ema_filter)
        
        cur_fast = ema_fast[-1]
        cur_slow = ema_slow[-1]
        cur_filter = ema_filter[-1]
        
        # H1 trend filter: get H1 EMA alignment
        h1_bullish = True
        h1_bearish = True
        if self.use_h1_trend_filter:
            h1_candles = self._candles.get("H1", [])
            if len(h1_candles) >= self.ema_slow + 5:
                h1_closes = [c.close for c in h1_candles]
                h1_ema_fast = self._calculate_ema(h1_closes, self.ema_fast)
                h1_ema_slow = self._calculate_ema(h1_closes, self.ema_slow)
                h1_bullish = h1_ema_fast[-1] > h1_ema_slow[-1]
                h1_bearish = h1_ema_fast[-1] < h1_ema_slow[-1]
        
        # Pullback zone: 0.3% of fast EMA
        pullback_zone = cur_fast * 0.003
        
        # === LONG ENTRY ===
        # EMAs aligned bullish + price pulled back to fast EMA + bullish candle + H1 bullish
        if cur_fast > cur_slow > cur_filter and h1_bullish:
            price_near_ema = abs(current_candle.low - cur_fast) <= pullback_zone
            if price_near_ema and current_candle.is_bullish:
                entry_price = tick.ask
                sl_distance = atr * self.sl_mult_long
                tp_distance = atr * self.tp_mult_long
                
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
                rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                
                factors = [
                    f"H1 trend filter: BULLISH" if self.use_h1_trend_filter else "H1 filter: OFF",
                    f"EMA alignment: fast({self.ema_fast})>{cur_fast:.2f} > slow({self.ema_slow})>{cur_slow:.2f} > filter({self.ema_filter})>{cur_filter:.2f}",
                    f"Pullback to EMA: low={current_candle.low:.2f}, zone={pullback_zone:.2f}",
                    f"Bullish confirmation candle",
                    f"ATR: {atr:.2f}",
                ]
                
                logger.info(f"🚀 LONG ENTRY: Price pullback to EMA + bullish candle + H1 bullish")
                
                return TradeSetup(
                    signal_type=SignalType.BUY,
                    entry_price=round(entry_price, 2),
                    take_profit=round(take_profit, 2),
                    stop_loss=round(stop_loss, 2),
                    risk_reward=round(rr_ratio, 2),
                    confidence=0.80,
                    timestamp=tick.timestamp,
                    h1_trend=TrendDirection.BULLISH,
                    m15_trend=TrendDirection.BULLISH,
                    m5_trend=TrendDirection.BULLISH,
                    entry_reason="EMA Pullback LONG: Bullish alignment + pullback + reversal candle + H1 filter",
                    confluence_factors=factors,
                    pip_target=round(tp_distance * 10, 1),
                    pip_risk=round(sl_distance * 10, 1),
                )
        
        # === SHORT ENTRY ===
        # EMAs aligned bearish + price pulled back to fast EMA + bearish candle + H1 bearish
        if cur_fast < cur_slow < cur_filter and h1_bearish:
            price_near_ema = abs(current_candle.high - cur_fast) <= pullback_zone
            if price_near_ema and current_candle.is_bearish:
                entry_price = tick.bid
                sl_distance = atr * self.sl_mult_short
                tp_distance = atr * self.tp_mult_short
                
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance
                rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                
                factors = [
                    f"H1 trend filter: BEARISH" if self.use_h1_trend_filter else "H1 filter: OFF",
                    f"EMA alignment: fast({self.ema_fast})>{cur_fast:.2f} < slow({self.ema_slow})>{cur_slow:.2f} < filter({self.ema_filter})>{cur_filter:.2f}",
                    f"Pullback to EMA: high={current_candle.high:.2f}, zone={pullback_zone:.2f}",
                    f"Bearish confirmation candle",
                    f"ATR: {atr:.2f}",
                ]
                
                logger.info(f"🚀 SHORT ENTRY: Price pullback to EMA + bearish candle + H1 bearish")
                
                return TradeSetup(
                    signal_type=SignalType.SELL,
                    entry_price=round(entry_price, 2),
                    take_profit=round(take_profit, 2),
                    stop_loss=round(stop_loss, 2),
                    risk_reward=round(rr_ratio, 2),
                    confidence=0.80,
                    timestamp=tick.timestamp,
                    h1_trend=TrendDirection.BEARISH,
                    m15_trend=TrendDirection.BEARISH,
                    m5_trend=TrendDirection.BEARISH,
                    entry_reason="EMA Pullback SHORT: Bearish alignment + pullback + reversal candle + H1 filter",
                    confluence_factors=factors,
                    pip_target=round(tp_distance * 10, 1),
                    pip_risk=round(sl_distance * 10, 1),
                )
        
        return None

    def _check_exit_conditions(self, tick: PriceTick) -> Optional[str]:
        """Check if active trade should be closed"""
        if not self.active_trade:
            return None
        
        trade = self.active_trade
        
        if trade.signal_type == SignalType.BUY:
            current_price = tick.bid
            pnl = current_price - trade.entry_price
            
            # TP hit
            if current_price >= trade.take_profit:
                return "Take Profit hit"
            
            # SL hit
            if current_price <= trade.stop_loss:
                return "Stop Loss hit"
            
            # Update max favorable
            if pnl > trade.max_favorable:
                trade.max_favorable = pnl
            if pnl < 0 and abs(pnl) > trade.max_adverse:
                trade.max_adverse = abs(pnl)
            
            # Trailing stop: if 60% of target reached, move SL to breakeven
            target_distance = trade.take_profit - trade.entry_price
            if pnl >= target_distance * 0.6 and not trade.trailing_active:
                trade.stop_loss = trade.entry_price + 0.1  # Breakeven + small buffer
                trade.trailing_active = True
                logger.info(f"Trailing stop activated - SL moved to breakeven")
            
            # Strong M15 trend reversal check
            m15 = self._analysis.get("M15")
            if m15 and m15.trend == TrendDirection.BEARISH and pnl > 0:
                if pnl >= target_distance * 0.3:  # At least 30% profit
                    return "M15 trend reversal - profit lock"
        
        else:  # SELL trade
            current_price = tick.ask
            pnl = trade.entry_price - current_price
            
            if current_price <= trade.take_profit:
                return "Take Profit hit"
            
            if current_price >= trade.stop_loss:
                return "Stop Loss hit"
            
            if pnl > trade.max_favorable:
                trade.max_favorable = pnl
            if pnl < 0 and abs(pnl) > trade.max_adverse:
                trade.max_adverse = abs(pnl)
            
            target_distance = trade.entry_price - trade.take_profit
            if pnl >= target_distance * 0.6 and not trade.trailing_active:
                trade.stop_loss = trade.entry_price - 0.1
                trade.trailing_active = True
                logger.info(f"Trailing stop activated - SL moved to breakeven")
            
            m15 = self._analysis.get("M15")
            if m15 and m15.trend == TrendDirection.BULLISH and pnl > 0:
                if pnl >= target_distance * 0.3:
                    return "M15 trend reversal - profit lock"
        
        return None

    # =========================================================================
    # TRADE MANAGEMENT
    # =========================================================================
    
    def _open_trade(self, setup: TradeSetup) -> None:
        """Open a new trade"""
        self.active_trade = ActiveTrade(
            signal_type=setup.signal_type,
            entry_price=setup.entry_price,
            entry_time=setup.timestamp,
            take_profit=setup.take_profit,
            stop_loss=setup.stop_loss,
            original_sl=setup.stop_loss,
        )
        self._last_trade_time = setup.timestamp
        
        logger.info(
            f"🚀 TRADE OPENED: {setup.signal_type.value} @ {setup.entry_price:.2f} | "
            f"TP: {setup.take_profit:.2f} ({setup.pip_target:.0f} pips) | "
            f"SL: {setup.stop_loss:.2f} ({setup.pip_risk:.0f} pips) | "
            f"RR: 1:{setup.risk_reward:.1f} | "
            f"Confidence: {setup.confidence:.0%}"
        )
        logger.info(f"   Factors: {', '.join(setup.confluence_factors)}")

    def _close_trade(self, tick: PriceTick, reason: str) -> dict:
        """Close the active trade"""
        if not self.active_trade:
            return {}
        
        trade = self.active_trade
        exit_price = tick.bid if trade.signal_type == SignalType.BUY else tick.ask
        pnl_points = trade.current_pnl_points(exit_price)
        pnl_pips = pnl_points * 10  # Convert to pips
        
        self.total_trades += 1
        self.total_pnl += pnl_points
        if pnl_points > 0:
            self.winning_trades += 1
            # Reset consecutive losses on win
            self._consecutive_losses = 0
            self._last_trade_won = True
        else:
            # Track consecutive losses
            self._consecutive_losses += 1
            self._last_trade_won = False
            if self._consecutive_losses >= self.max_consecutive_losses:
                logger.warning(f"⚠️ {self._consecutive_losses} consecutive losses - activating extended {self.extended_cooldown_minutes}min cooldown")
        
        duration = tick.timestamp - trade.entry_time
        
        trade_record = {
            "type": trade.signal_type.value,
            "entry_price": trade.entry_price,
            "exit_price": round(exit_price, 2),
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": tick.timestamp.isoformat(),
            "duration_minutes": int(duration.total_seconds() / 60),
            "pnl_points": round(pnl_points, 2),
            "pnl_pips": round(pnl_pips, 1),
            "pnl_usd": round(pnl_points * 10 * trade.lot_size / 0.01, 2),
            "max_favorable_pips": round(trade.max_favorable * 10, 1),
            "max_adverse_pips": round(trade.max_adverse * 10, 1),
            "reason": reason,
            "won": pnl_points > 0,
            "trailing_used": trade.trailing_active,
            "consecutive_losses": self._consecutive_losses,
        }
        
        self.trade_history.append(trade_record)
        
        emoji = "✅" if pnl_points > 0 else "❌"
        logger.info(
            f"{emoji} TRADE CLOSED: {trade.signal_type.value} @ {exit_price:.2f} | "
            f"PnL: {pnl_pips:+.1f} pips ({pnl_points:+.2f} pts) | "
            f"Duration: {duration.total_seconds()/60:.0f}m | {reason}"
        )
        
        if self._on_trade_close:
            self._on_trade_close(trade_record)
        
        self.active_trade = None
        return trade_record

    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    async def process_tick(self, tick: PriceTick) -> Optional[TradeSetup]:
        """Main tick processing - called on each new tick"""
        self._current_tick = tick
        
        # Check exit first if in trade
        if self.active_trade:
            exit_reason = self._check_exit_conditions(tick)
            if exit_reason:
                self._close_trade(tick, exit_reason)
                return None
        
        # Check for new entry
        setup = self._check_entry_conditions(tick)
        if setup:
            self._open_trade(setup)
            self.signal_history.append(setup)
            
            if self._on_signal:
                self._on_signal(setup)
            
            return setup
        
        return None

    async def update_and_analyze(self) -> Dict[str, TimeframeAnalysis]:
        """Update candles and run full analysis"""
        await self.update_all_candles()
        return self.analyze_all_timeframes()

    # =========================================================================
    # STATUS & STATISTICS
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get trading statistics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_pnl = (self.total_pnl / self.total_trades) if self.total_trades > 0 else 0
        
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.total_trades - self.winning_trades,
            "win_rate": round(win_rate, 1),
            "total_pnl_points": round(self.total_pnl, 2),
            "total_pnl_pips": round(self.total_pnl * 10, 1),
            "avg_pnl_pips": round(avg_pnl * 10, 1),
            "active_trade": self.active_trade is not None,
        }

    def get_current_indicators(self) -> Optional[dict]:
        """Get current indicator values for display - Sunrise Strategy"""
        m15 = self._analysis.get("M15")
        
        if not m15:
            return None
        
        return {
            # Sunrise EMA values
            "ema_fast": round(self._ema_values.get('fast', 0), 2),
            "ema_slow": round(self._ema_values.get('slow', 0), 2),
            "ema_filter": round(self._ema_values.get('filter', 0), 2),
            "atr": round(m15.atr, 2),
            "last_price": round(m15.last_close, 2),
            
            # State machine status
            "entry_state": self._entry_state,
            "pullback_count": self._pullback_candle_count,
            "signal_atr": round(self._signal_detection_atr, 2),
            
            # Window info (if active)
            "window_top": round(self._window_top_limit, 2) if self._window_top_limit > 0 else None,
            "window_bottom": round(self._window_bottom_limit, 2) if self._window_bottom_limit > 0 else None,
            
            # Trend summary
            "m15_trend": m15.trend.value,
            
            # Strategy name
            "strategy": "Sunrise",
        }

    def get_active_trade_info(self) -> Optional[dict]:
        """Get info about active trade"""
        if not self.active_trade or not self._current_tick:
            return None
        
        trade = self.active_trade
        tick = self._current_tick
        current_price = tick.bid if trade.signal_type == SignalType.BUY else tick.ask
        pnl = trade.current_pnl_points(current_price)
        
        return {
            "type": trade.signal_type.value,
            "entry_price": trade.entry_price,
            "current_price": round(current_price, 2),
            "take_profit": trade.take_profit,
            "stop_loss": trade.stop_loss,
            "pnl_points": round(pnl, 2),
            "pnl_pips": round(pnl * 10, 1),
            "duration_minutes": int((datetime.now(timezone.utc) - trade.entry_time).total_seconds() / 60),
            "trailing_active": trade.trailing_active,
        }

    def get_key_levels(self) -> List[dict]:
        """Get all key levels for charting"""
        levels = []
        for tf in ["H1", "M15"]:
            analysis = self._analysis.get(tf)
            if analysis:
                for level in analysis.key_levels[:5]:
                    levels.append({
                        "price": level.price,
                        "type": level.level_type,
                        "timeframe": level.timeframe,
                        "strength": level.strength,
                    })
        return levels

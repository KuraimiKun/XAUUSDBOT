"""
Backtesting Engine for Sunrise EMA Crossover Strategy

Runs historical backtests using MT5 historical data.
4-Phase entry system: SCANNING → ARMED → WINDOW_OPEN → BREAKOUT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from enum import Enum

import numpy as np
from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


class SignalType(Enum):
    """Signal type enum for backtesting"""
    BUY = "BUY"
    SELL = "SELL"
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
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class BacktestTrade:
    """A single trade in the backtest"""
    id: int
    signal_type: 'SignalType'
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    take_profit: float = 0.0
    stop_loss: float = 0.0
    pnl: float = 0.0  # In dollars
    pnl_pct: float = 0.0
    pnl_pips: float = 0.0
    won: bool = False
    exit_reason: str = ""
    duration_minutes: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results"""
    # Period info
    start_date: datetime
    end_date: datetime
    symbol: str
    
    # Balance tracking
    initial_balance: float = 10000.0
    final_balance: float = 10000.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # PnL statistics
    total_pnl_pips: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    avg_trade_duration: Optional[timedelta] = None
    
    # Monthly breakdown
    monthly_breakdown: Dict = field(default_factory=dict)
    
    # Comparison
    buy_hold_return: float = 0.0
    buy_hold_pct: float = 0.0
    
    # Trade list
    trades: List['BacktestTrade'] = field(default_factory=list)
    
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Backtesting engine that replays historical data through the Sunrise strategy.
    4-Phase State Machine: SCANNING → ARMED → WINDOW_OPEN → BREAKOUT
    """
    
    # State machine states
    STATE_SCANNING = "SCANNING"
    STATE_ARMED_LONG = "ARMED_LONG"
    STATE_ARMED_SHORT = "ARMED_SHORT"
    STATE_WINDOW_OPEN_LONG = "WINDOW_OPEN_LONG"
    STATE_WINDOW_OPEN_SHORT = "WINDOW_OPEN_SHORT"
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        settings = None,
        # === SUNRISE EMA PERIODS ===
        ema_confirm: int = 1,
        ema_fast: int = 14,
        ema_medium: int = 14,
        ema_slow: int = 24,
        ema_filter: int = 100,
        ema_exit: int = 25,
        ema_trend: int = 50,  # Legacy compatibility
        # === RSI (legacy) ===
        rsi_period: int = 14,
        rsi_ob: float = 70.0,
        rsi_os: float = 30.0,
        # === ATR SETTINGS ===
        atr_period: int = 14,
        atr_min: float = 0.0,
        atr_max: float = 999.0,
        atr_increment_filter: float = 0.5,
        atr_decrement_filter: float = -0.5,
        # === SL/TP MULTIPLIERS ===
        sl_mult_long: float = 1.2,    # Wider SL to avoid noise
        tp_mult_long: float = 2.4,    # 1:2 R:R ratio
        sl_mult_short: float = 1.2,   # Same for shorts
        tp_mult_short: float = 2.4,   # 1:2 R:R ratio
        # === PULLBACK SETTINGS ===
        pullback_candles_long: int = 2,   # Reduced from 3 for more signals
        pullback_candles_short: int = 2,
        # === ENTRY WINDOW ===
        window_period_long: int = 5,      # Increased from 1 - gives 5 bars to catch breakout
        window_period_short: int = 5,     # Reduced from 7 for consistency
        window_price_offset: float = 0.5, # Increased for gold volatility
        # === TIME FILTER (UTC) ===
        time_filter_start: int = 0,
        time_filter_end: int = 20,        # Expanded to 20 hours (00:00-20:00 UTC)
        # === ANGLE FILTER ===
        angle_filter_enabled: bool = False, # Disabled for more signals
        angle_threshold: float = 0.2,
        # === NEW FILTERS ===
        use_h1_trend_filter: bool = True,  # Only trade with H1 trend
        use_rsi_filter: bool = False,      # Disabled - reduces trades too much
        min_candle_body_atr: float = 0.3,  # Min candle body size (ATR %)
        max_consecutive_losses: int = 3,   # Max losses before extended cooldown
        extended_cooldown_minutes: int = 60,  # Cooldown after max losses
        # Legacy params
        min_tp_points: float = 3.0,
        max_sl_points: float = 2.0,
        min_rr_ratio: float = 1.0,
        min_confidence: float = 0.70,
        cooldown_minutes: int = 5,
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        
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
        
        # Pullback settings
        self.pullback_candles_long = pullback_candles_long
        self.pullback_candles_short = pullback_candles_short
        
        # Entry window
        self.window_period_long = window_period_long
        self.window_period_short = window_period_short
        self.window_price_offset = window_price_offset
        
        # Time filter
        self.time_filter_start = time_filter_start
        self.time_filter_end = time_filter_end
        
        # Angle filter
        self.angle_filter_enabled = angle_filter_enabled
        self.angle_threshold = angle_threshold
        
        # New filters
        self.use_h1_trend_filter = use_h1_trend_filter
        self.use_rsi_filter = use_rsi_filter
        self.min_candle_body_atr = min_candle_body_atr
        self.max_consecutive_losses = max_consecutive_losses
        self.extended_cooldown_minutes = extended_cooldown_minutes
        
        # Legacy
        self.min_tp_points = min_tp_points
        self.max_sl_points = max_sl_points
        self.min_rr_ratio = min_rr_ratio
        self.min_confidence = min_confidence
        self.cooldown_minutes = cooldown_minutes
        
        # Override with settings if provided
        if settings:
            self.ema_fast = getattr(settings, 'ema_fast', ema_fast)
            self.ema_slow = getattr(settings, 'ema_slow', ema_slow)
            self.ema_trend = getattr(settings, 'ema_trend', ema_trend)
            self.atr_period = getattr(settings, 'atr_period', atr_period)

    def fetch_historical_candles(
        self,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Candle]:
        """Fetch historical candles from MT5 (synchronous)"""
        if mt5 is None:
            logger.error("MT5 not available")
            return []

        # Initialize MT5 if needed
        if not mt5.initialize():
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return []

        rates = mt5.copy_rates_range(
            self.symbol,
            timeframe.mt5_tf,
            start_date,
            end_date
        )

        if rates is None or len(rates) == 0:
            logger.warning(f"No data for {timeframe.name_str} from {start_date} to {end_date}")
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
            ))
        return candles

    def _calculate_ema(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA series"""
        if len(closes) < period:
            return np.full_like(closes, np.nan, dtype=float)

        ema = np.zeros(len(closes), dtype=float)
        ema[:period-1] = np.nan
        multiplier = 2 / (period + 1)

        # SMA for first value
        ema[period-1] = np.mean(closes[:period])

        for i in range(period, len(closes)):
            ema[i] = (closes[i] * multiplier) + (ema[i-1] * (1 - multiplier))

        return ema

    def _calculate_rsi(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI series"""
        rsi = np.full(len(closes), 50.0)

        if len(closes) < period + 1:
            return rsi

        deltas = np.diff(closes)

        for i in range(period, len(closes)):
            gains = np.where(deltas[i-period:i] > 0, deltas[i-period:i], 0)
            losses = np.where(deltas[i-period:i] < 0, -deltas[i-period:i], 0)

            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)

            if avg_loss == 0:
                rsi[i] = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(self, candles: List[Candle], period: int) -> np.ndarray:
        """Calculate ATR series"""
        n = len(candles)
        atr = np.full(n, 0.5)  # Default

        if n < period + 1:
            return atr

        for i in range(period, n):
            tr_values = []
            for j in range(i - period, i):
                high_low = candles[j].high - candles[j].low
                high_close = abs(candles[j].high - candles[j-1].close)
                low_close = abs(candles[j].low - candles[j-1].close)
                tr_values.append(max(high_low, high_close, low_close))
            atr[i] = np.mean(tr_values)

        return atr

    def _is_near_swing_level(
        self,
        candles: List[Candle],
        idx: int,
        lookback: int,
        atr: float,
        direction: str,
    ) -> bool:
        """Ensure price is reacting near recent swing support/resistance"""
        if idx <= 0 or atr <= 0:
            return False

        start = max(0, idx - lookback)
        window = candles[start:idx]
        if len(window) < max(10, lookback // 2):
            return False

        recent_high = max(c.high for c in window)
        recent_low = min(c.low for c in window)
        price = candles[idx].close
        buffer = atr * 1.2
        safety_gap = atr * 0.8

        if direction == "BULLISH":
            dist_support = price - recent_low
            dist_resistance = recent_high - price
            return dist_support <= buffer and dist_resistance >= safety_gap
        elif direction == "BEARISH":
            dist_resistance = recent_high - price
            dist_support = price - recent_low
            return dist_resistance <= buffer and dist_support >= safety_gap

        return False

    def _get_trend(self, ema_fast: float, ema_slow: float, ema_trend: float) -> str:
        """Determine trend direction (legacy compatibility)"""
        if np.isnan(ema_fast) or np.isnan(ema_slow) or np.isnan(ema_trend):
            return "NEUTRAL"

        if ema_fast > ema_slow > ema_trend:
            return "BULLISH"
        elif ema_fast < ema_slow < ema_trend:
            return "BEARISH"
        return "NEUTRAL"
    
    def _check_time_filter(self, hour: int) -> bool:
        """Check if hour is within trading window"""
        if self.time_filter_start <= self.time_filter_end:
            return self.time_filter_start <= hour < self.time_filter_end
        else:
            return hour >= self.time_filter_start or hour < self.time_filter_end
    
    def _calculate_ema_angle(self, ema_values: np.ndarray, idx: int, lookback: int = 5) -> float:
        """Calculate EMA slope at given index"""
        if idx < lookback:
            return 0.0
        start_idx = idx - lookback
        return (ema_values[idx] - ema_values[start_idx]) / lookback
    
    def _check_atr_filter(self, atr: float, prev_atr: float) -> bool:
        """Check ATR volatility filters"""
        if atr < self.atr_min or atr > self.atr_max:
            return False
        if prev_atr > 0:
            atr_change = atr - prev_atr
            if atr_change > self.atr_increment_filter:
                return False
            if atr_change < self.atr_decrement_filter:
                return False
        return True
    
    def _detect_ema_crossover(
        self,
        idx: int,
        ema_fast: np.ndarray,
        ema_slow: np.ndarray,
        ema_filter: np.ndarray,
        closes: np.ndarray,
        candles: List[Candle],
    ) -> str:
        """
        Detect EMA trend alignment at given index.
        Returns "BULLISH", "BEARISH", or "NONE"
        Modified: Instead of requiring fresh crossover, detect sustained alignment
        """
        if idx < 2:
            return "NONE"
        
        cur_fast = ema_fast[idx]
        cur_slow = ema_slow[idx]
        cur_filter = ema_filter[idx]
        prev_fast = ema_fast[idx - 1]
        prev_slow = ema_slow[idx - 1]
        
        if np.isnan(cur_fast) or np.isnan(cur_slow) or np.isnan(cur_filter):
            return "NONE"
        
        current_candle = candles[idx]
        price = closes[idx]
        
        # Check for fresh crossover first
        bullish_cross = (prev_fast <= prev_slow) and (cur_fast > cur_slow)
        bearish_cross = (prev_fast >= prev_slow) and (cur_fast < cur_slow)
        
        # Bullish: fresh crossover OR sustained bullish alignment (fast > slow)
        if bullish_cross or (cur_fast > cur_slow):
            # Filter: price should be above filter EMA or fast above filter
            if (price > cur_filter) or (cur_fast > cur_filter):
                if self.angle_filter_enabled:
                    angle = self._calculate_ema_angle(ema_fast, idx)
                    if angle < self.angle_threshold:
                        return "NONE"
                return "BULLISH"
        
        # Bearish: fresh crossover OR sustained bearish alignment (fast < slow)
        if bearish_cross or (cur_fast < cur_slow):
            if (price < cur_filter) or (cur_fast < cur_filter):
                if self.angle_filter_enabled:
                    angle = self._calculate_ema_angle(ema_fast, idx)
                    if angle > -self.angle_threshold:
                        return "NONE"
                return "BEARISH"
        
        return "NONE"

    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        progress_callback=None,
    ) -> BacktestResult:
        """
        Run backtest over the specified period (synchronous).
        
        Uses M5 candles for entry/exit simulation, with M15 and H1 for trend analysis.
        """
        # Ensure dates are timezone-aware (UTC)
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
            
        logger.info(f"Starting backtest from {start_date} to {end_date}")

        # Fetch historical data for all timeframes
        # Add extra lookback for indicator calculation
        lookback_start = start_date - timedelta(days=7)

        if progress_callback:
            progress_callback(0.1, "Fetching M5 data...")
        m5_candles = self.fetch_historical_candles(Timeframe.M5, lookback_start, end_date)

        if progress_callback:
            progress_callback(0.2, "Fetching M15 data...")
        m15_candles = self.fetch_historical_candles(Timeframe.M15, lookback_start, end_date)

        if progress_callback:
            progress_callback(0.3, "Fetching H1 data...")
        h1_candles = self.fetch_historical_candles(Timeframe.H1, lookback_start, end_date)

        if not m5_candles or not m15_candles or not h1_candles:
            logger.error("Failed to fetch historical data")
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                symbol=self.symbol,
            )

        logger.info(f"Loaded {len(m5_candles)} M5, {len(m15_candles)} M15, {len(h1_candles)} H1 candles")

        if progress_callback:
            progress_callback(0.4, "Calculating indicators...")

        # Calculate indicators for each timeframe
        m5_closes = np.array([c.close for c in m5_candles])
        m15_closes = np.array([c.close for c in m15_candles])
        h1_closes = np.array([c.close for c in h1_candles])

        # M5 indicators
        m5_ema_fast = self._calculate_ema(m5_closes, self.ema_fast)
        m5_ema_slow = self._calculate_ema(m5_closes, self.ema_slow)
        m5_ema_trend = self._calculate_ema(m5_closes, self.ema_trend)
        m5_rsi = self._calculate_rsi(m5_closes, self.rsi_period)
        m5_atr = self._calculate_atr(m5_candles, self.atr_period)

        # M15 indicators - Sunrise strategy uses these primarily
        m15_ema_fast = self._calculate_ema(m15_closes, self.ema_fast)
        m15_ema_slow = self._calculate_ema(m15_closes, self.ema_slow)
        m15_ema_filter = self._calculate_ema(m15_closes, self.ema_filter)
        m15_ema_trend = self._calculate_ema(m15_closes, self.ema_trend)
        m15_atr = self._calculate_atr(m15_candles, self.atr_period)

        # H1 indicators (for reference)
        h1_ema_fast = self._calculate_ema(h1_closes, self.ema_fast)
        h1_ema_slow = self._calculate_ema(h1_closes, self.ema_slow)
        h1_ema_trend = self._calculate_ema(h1_closes, self.ema_trend)

        if progress_callback:
            progress_callback(0.5, "Running Sunrise strategy simulation...")

        # Create index maps for faster lookups
        m15_time_idx = {c.timestamp: i for i, c in enumerate(m15_candles)}
        h1_time_idx = {c.timestamp: i for i, c in enumerate(h1_candles)}

        # Simulation state
        trades: List[BacktestTrade] = []
        active_trade: Optional[BacktestTrade] = None
        last_trade_time: Optional[datetime] = None
        last_trade_was_loss: bool = False
        consecutive_losses: int = 0
        trade_id = 0
        equity_curve: List[float] = [self.initial_balance]
        current_balance = self.initial_balance
        
        # === SUNRISE STATE MACHINE VARIABLES ===
        entry_state = self.STATE_SCANNING
        pullback_candle_count = 0
        signal_detection_atr = 0.0
        window_top_limit = 0.0
        window_bottom_limit = 0.0
        window_expiry_bar = 0
        bar_count = 0

        # Find M15 candles within our test period (Sunrise uses M15 primarily)
        test_m15_indices = [
            i for i, c in enumerate(m15_candles)
            if start_date <= c.timestamp <= end_date
        ]

        if not test_m15_indices:
            logger.error("No M15 candles in test period")
            return BacktestResult(
                start_date=start_date, end_date=end_date, symbol=self.symbol,
                initial_balance=self.initial_balance, final_balance=self.initial_balance,
                equity_curve=[self.initial_balance],
            )

        total_bars = len(test_m15_indices)

        # Simulate bar by bar on M15 timeframe
        for bar_num, m15_idx in enumerate(test_m15_indices):
            candle = m15_candles[m15_idx]
            bar_count += 1

            # Progress update
            if progress_callback and bar_num % 50 == 0:
                progress = 0.5 + (bar_num / total_bars) * 0.45
                progress_callback(progress, f"Processing bar {bar_num}/{total_bars}...")

            # Time filter check
            hour = candle.timestamp.hour
            if not self._check_time_filter(hour):
                entry_state = self.STATE_SCANNING
                pullback_candle_count = 0
                continue

            atr = m15_atr[m15_idx]
            prev_atr = m15_atr[m15_idx - 1] if m15_idx > 0 else atr
            
            # ATR filter
            if not self._check_atr_filter(atr, prev_atr):
                entry_state = self.STATE_SCANNING
                pullback_candle_count = 0
                continue

            price = candle.close

            # If in trade, check for exit
            if active_trade:
                is_buy = active_trade.signal_type == SignalType.BUY
                
                # Track max favorable/adverse
                if is_buy:
                    favorable = candle.high - active_trade.entry_price
                    adverse = active_trade.entry_price - candle.low
                else:
                    favorable = active_trade.entry_price - candle.low
                    adverse = candle.high - active_trade.entry_price
                    
                if favorable > active_trade.max_favorable:
                    active_trade.max_favorable = favorable
                if adverse > active_trade.max_adverse:
                    active_trade.max_adverse = adverse

                pnl_points = 0.0
                trade_closed = False

                if is_buy:
                    if candle.high >= active_trade.take_profit:
                        active_trade.exit_time = candle.timestamp
                        active_trade.exit_price = active_trade.take_profit
                        pnl_points = active_trade.take_profit - active_trade.entry_price
                        active_trade.exit_reason = "Take Profit"
                        active_trade.won = True
                        trade_closed = True
                    elif candle.low <= active_trade.stop_loss:
                        active_trade.exit_time = candle.timestamp
                        active_trade.exit_price = active_trade.stop_loss
                        pnl_points = active_trade.stop_loss - active_trade.entry_price
                        active_trade.exit_reason = "Stop Loss"
                        active_trade.won = False
                        trade_closed = True
                else:  # SELL
                    if candle.low <= active_trade.take_profit:
                        active_trade.exit_time = candle.timestamp
                        active_trade.exit_price = active_trade.take_profit
                        pnl_points = active_trade.entry_price - active_trade.take_profit
                        active_trade.exit_reason = "Take Profit"
                        active_trade.won = True
                        trade_closed = True
                    elif candle.high >= active_trade.stop_loss:
                        active_trade.exit_time = candle.timestamp
                        active_trade.exit_price = active_trade.stop_loss
                        pnl_points = active_trade.entry_price - active_trade.stop_loss
                        active_trade.exit_reason = "Stop Loss"
                        active_trade.won = False
                        trade_closed = True

                if trade_closed:
                    active_trade.pnl_pips = pnl_points * 10
                    risk_amount = current_balance * self.risk_per_trade
                    sl_pips = abs(active_trade.entry_price - active_trade.stop_loss) * 10
                    if sl_pips > 0:
                        position_size = risk_amount / sl_pips
                        active_trade.pnl = active_trade.pnl_pips * position_size
                    else:
                        active_trade.pnl = active_trade.pnl_pips * 10
                    
                    active_trade.pnl_pct = (active_trade.pnl / current_balance) * 100
                    active_trade.duration_minutes = int(
                        (active_trade.exit_time - active_trade.entry_time).total_seconds() / 60
                    )
                    trades.append(active_trade)
                    current_balance += active_trade.pnl
                    equity_curve.append(current_balance)
                    last_trade_time = active_trade.exit_time
                    
                    # Track consecutive losses for extended cooldown
                    if active_trade.won:
                        consecutive_losses = 0
                        last_trade_was_loss = False
                    else:
                        consecutive_losses += 1
                        last_trade_was_loss = True
                    
                    active_trade = None
                    entry_state = self.STATE_SCANNING
                    pullback_candle_count = 0
                    continue

            # ================================================================
            # SIMPLIFIED EMA PULLBACK STRATEGY - Entry Logic
            # ================================================================
            
            if active_trade is not None:
                continue
            
            # Check cooldown - extended after consecutive losses
            if last_trade_time:
                if consecutive_losses >= self.max_consecutive_losses:
                    cooldown_end = last_trade_time + timedelta(minutes=self.extended_cooldown_minutes)
                else:
                    cooldown_end = last_trade_time + timedelta(minutes=self.cooldown_minutes)
                if candle.timestamp < cooldown_end:
                    continue

            # Get current EMA values
            cur_fast = m15_ema_fast[m15_idx]
            cur_slow = m15_ema_slow[m15_idx]
            cur_filter = m15_ema_filter[m15_idx]
            
            if np.isnan(cur_fast) or np.isnan(cur_slow) or np.isnan(cur_filter):
                continue
            
            # Get H1 trend for higher timeframe filter
            h1_idx = None
            candle_h1_time = candle.timestamp.replace(minute=0, second=0, microsecond=0)
            if candle_h1_time in h1_time_idx:
                h1_idx = h1_time_idx[candle_h1_time]
            elif len(h1_candles) > 0:
                # Find closest H1 candle
                for i, h1c in enumerate(h1_candles):
                    if h1c.timestamp <= candle.timestamp:
                        h1_idx = i
            
            h1_bullish = False
            h1_bearish = False
            if h1_idx is not None and h1_idx < len(h1_ema_fast):
                h1_fast = h1_ema_fast[h1_idx]
                h1_slow = h1_ema_slow[h1_idx]
                if not np.isnan(h1_fast) and not np.isnan(h1_slow):
                    h1_bullish = h1_fast > h1_slow
                    h1_bearish = h1_fast < h1_slow
            
            # Get RSI for overbought/oversold filter
            rsi = m5_rsi[m15_idx * 3] if m15_idx * 3 < len(m5_rsi) else 50  # Approximate M15 to M5 index
            
            # Minimum candle body size filter (avoid doji/indecision)
            min_body = atr * self.min_candle_body_atr
            
            # === LONG ENTRY CONDITIONS ===
            # 1. EMAs aligned bullish: fast > slow, and both above filter
            # 2. H1 trend must be bullish (if filter enabled)
            # 3. RSI not overbought (if filter enabled)
            # 4. Price pulled back to touch/near fast EMA (within 0.3%)
            # 5. Current candle is bullish with meaningful body
            if cur_fast > cur_slow > cur_filter:
                # H1 trend filter
                if self.use_h1_trend_filter and not h1_bullish:
                    continue
                
                # RSI filter - don't buy if overbought
                if self.use_rsi_filter and rsi > 70:
                    continue
                
                pullback_zone = cur_fast * 0.003  # 0.3% zone
                price_near_ema = abs(candle.low - cur_fast) <= pullback_zone
                has_good_body = candle.body_size >= min_body
                
                if price_near_ema and candle.is_bullish and has_good_body:
                    entry_price = candle.close
                    sl_distance = atr * self.sl_mult_long
                    tp_distance = atr * self.tp_mult_long
                    
                    trade_id += 1
                    active_trade = BacktestTrade(
                        id=trade_id,
                        signal_type=SignalType.BUY,
                        entry_time=candle.timestamp,
                        entry_price=entry_price,
                        take_profit=entry_price + tp_distance,
                        stop_loss=entry_price - sl_distance,
                    )
                    consecutive_losses = 0  # Reset on new trade
                    continue
            
            # === SHORT ENTRY CONDITIONS ===
            # 1. EMAs aligned bearish: fast < slow, and both below filter
            # 2. H1 trend must be bearish (if filter enabled)
            # 3. RSI not oversold (if filter enabled)
            # 4. Price pulled back to touch/near fast EMA (within 0.3%)
            # 5. Current candle is bearish with meaningful body
            if cur_fast < cur_slow < cur_filter:
                # H1 trend filter
                if self.use_h1_trend_filter and not h1_bearish:
                    continue
                
                # RSI filter - don't sell if oversold
                if self.use_rsi_filter and rsi < 30:
                    continue
                
                pullback_zone = cur_fast * 0.003  # 0.3% zone
                price_near_ema = abs(candle.high - cur_fast) <= pullback_zone
                has_good_body = candle.body_size >= min_body
                
                if price_near_ema and candle.is_bearish and has_good_body:
                    entry_price = candle.close
                    sl_distance = atr * self.sl_mult_short
                    tp_distance = atr * self.tp_mult_short
                    
                    trade_id += 1
                    active_trade = BacktestTrade(
                        id=trade_id,
                        signal_type=SignalType.SELL,
                        entry_time=candle.timestamp,
                        entry_price=entry_price,
                        take_profit=entry_price - tp_distance,
                        stop_loss=entry_price + sl_distance,
                    )
                    consecutive_losses = 0  # Reset on new trade
                    continue

        # Close any remaining trade at end of period
        if active_trade:
            last_candle = m15_candles[test_m15_indices[-1]]
            active_trade.exit_time = last_candle.timestamp
            active_trade.exit_price = last_candle.close
            if active_trade.signal_type == SignalType.BUY:
                pnl_points = last_candle.close - active_trade.entry_price
            else:
                pnl_points = active_trade.entry_price - last_candle.close
            active_trade.pnl_pips = pnl_points * 10
            active_trade.pnl = pnl_points * 10 * current_balance * self.risk_per_trade / 20
            active_trade.exit_reason = "End of Period"
            active_trade.won = pnl_points > 0
            active_trade.duration_minutes = int(
                (active_trade.exit_time - active_trade.entry_time).total_seconds() / 60
            )
            trades.append(active_trade)
            current_balance += active_trade.pnl
            equity_curve.append(current_balance)

        if progress_callback:
            progress_callback(0.95, "Calculating statistics...")

        # Get buy-and-hold comparison
        first_price = m15_candles[test_m15_indices[0]].close if test_m15_indices else 0
        last_price = m15_candles[test_m15_indices[-1]].close if test_m15_indices else 0
        buy_hold_return = last_price - first_price if first_price > 0 else 0
        buy_hold_pct = ((last_price / first_price) - 1) * 100 if first_price > 0 else 0

        # Calculate statistics
        result = self._calculate_statistics(
            trades, equity_curve, start_date, end_date,
            buy_hold_return, buy_hold_pct
        )

        if progress_callback:
            progress_callback(1.0, "Complete!")

        logger.info(
            f"Sunrise Backtest complete: {result.total_trades} trades, "
            f"{result.win_rate:.1f}% win rate, {result.total_pnl_pips:+.1f} pips"
        )

        return result

    def _calculate_statistics(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[float],
        start_date: datetime,
        end_date: datetime,
        buy_hold_return: float = 0.0,
        buy_hold_pct: float = 0.0,
    ) -> BacktestResult:
        """Calculate comprehensive backtest statistics"""
        final_balance = equity_curve[-1] if equity_curve else self.initial_balance
        total_return = final_balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            symbol=self.symbol,
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            total_return_pct=total_return_pct,
            trades=trades,
            equity_curve=equity_curve,
            buy_hold_return=buy_hold_return,
            buy_hold_pct=buy_hold_pct,
        )

        if not trades:
            return result

        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t.won)
        result.losing_trades = result.total_trades - result.winning_trades
        result.win_rate = (result.winning_trades / result.total_trades) * 100

        # PnL stats (in dollars)
        pnls = [t.pnl for t in trades]
        wins = [t.pnl for t in trades if t.won]
        losses = [t.pnl for t in trades if not t.won]

        result.total_pnl_pips = sum(t.pnl_pips for t in trades)
        result.avg_win = np.mean(wins) if wins else 0.0
        result.avg_loss = np.mean(losses) if losses else 0.0
        result.best_trade = max(pnls) if pnls else 0.0
        result.worst_trade = min(pnls) if pnls else 0.0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        if result.profit_factor == float('inf'):
            result.profit_factor = 99.99  # Cap for display

        # Max drawdown from equity curve
        peak = self.initial_balance
        max_dd = 0.0
        max_dd_pct = 0.0
        for balance in equity_curve:
            if balance > peak:
                peak = balance
            dd = peak - balance
            dd_pct = (dd / peak) * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        result.max_drawdown = max_dd
        result.max_drawdown_pct = max_dd_pct

        # Sharpe ratio (annualized)
        if len(pnls) > 1:
            returns = np.array(pnls)
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                # Approximate annualization (252 trading days)
                trades_per_year = 252 * 2  # Approximate
                result.sharpe_ratio = (avg_return / std_return) * np.sqrt(trades_per_year)
            else:
                result.sharpe_ratio = 0.0
        else:
            result.sharpe_ratio = 0.0

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        for t in trades:
            if t.won:
                current_wins += 1
                current_losses = 0
                if current_wins > max_consec_wins:
                    max_consec_wins = current_wins
            else:
                current_losses += 1
                current_wins = 0
                if current_losses > max_consec_losses:
                    max_consec_losses = current_losses

        result.consecutive_wins = max_consec_wins
        result.consecutive_losses = max_consec_losses

        # Average duration
        durations = [t.duration_minutes for t in trades if t.duration_minutes > 0]
        if durations:
            avg_dur = np.mean(durations)
            result.avg_trade_duration = timedelta(minutes=int(avg_dur))

        # Monthly breakdown
        monthly = {}
        for t in trades:
            month_key = t.entry_time.strftime("%Y-%m")
            if month_key not in monthly:
                monthly[month_key] = {"trades": 0, "wins": 0, "pnl": 0.0}
            monthly[month_key]["trades"] += 1
            monthly[month_key]["pnl"] += t.pnl
            if t.won:
                monthly[month_key]["wins"] += 1
        
        # Calculate monthly win rates
        for month_key in monthly:
            data = monthly[month_key]
            data["win_rate"] = (data["wins"] / data["trades"]) * 100 if data["trades"] > 0 else 0
        
        result.monthly_breakdown = monthly

        return result


# Singleton
_backtest_engine: Optional[BacktestEngine] = None


def get_backtest_engine(**kwargs) -> BacktestEngine:
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine(**kwargs)
    return _backtest_engine

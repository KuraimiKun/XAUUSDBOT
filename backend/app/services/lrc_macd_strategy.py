"""
LRC-MACD Strategy Backtesting Engine

A strategy using Linear Regression Candles and MACD
Based on: https://github.com/geraked - https://youtu.be/je6vGA30gcQ

Strategy Logic:
- LONG: MACD > 0, LRC close > open (bullish), LRC close crosses above signal
- SHORT: MACD < 0, LRC close < open (bearish), LRC close crosses below signal
- Pullback confirmation required before entry
- SL based on recent swing high/low
- TP based on SL distance * coefficient
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np
from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


class SignalType(Enum):
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


@dataclass
class LRCCandle:
    """Linear Regression Candle"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    signal: float  # SMA of LRC close
    color: int  # 1=bullish (green), -1=bearish (red)
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    @property
    def above_signal(self) -> bool:
        return self.close > self.signal
    
    @property
    def below_signal(self) -> bool:
        return self.close < self.signal


@dataclass
class BacktestTrade:
    """A single trade in the backtest"""
    id: int
    signal_type: SignalType
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    take_profit: float = 0.0
    stop_loss: float = 0.0
    pnl: float = 0.0
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
    start_date: datetime
    end_date: datetime
    symbol: str
    
    initial_balance: float = 10000.0
    final_balance: float = 10000.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl_pips: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    profit_factor: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    avg_trade_duration: Optional[timedelta] = None
    
    monthly_breakdown: Dict = field(default_factory=dict)
    buy_hold_return: float = 0.0
    buy_hold_pct: float = 0.0
    
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class LRCMACDStrategy:
    """
    LRC-MACD Strategy Backtester
    
    Uses Linear Regression Candles + MACD for entries.
    Pullback confirmation required before entering.
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        # === LRC Parameters ===
        lrc_period: int = 11,        # Linear Regression period
        lrc_signal_period: int = 5,  # LRC Signal (SMA) period
        # === MACD Parameters ===
        macd_fast: int = 34,         # MACD fast EMA
        macd_slow: int = 144,        # MACD slow EMA
        # === Trade Parameters ===
        tp_coefficient: float = 1.0,  # TP = SL distance * coefficient
        sl_lookback: int = 10,        # Bars to look back for SL level
        sl_deviation: float = 0.6,    # SL deviation in points (60 points = 6 pips for gold)
        pullback_lookback: int = 4,   # Bars to confirm pullback
        # === Risk Management ===
        trailing_stop_level: float = 0.5,  # Trailing stop at 50% of target
        use_trailing: bool = True,
        # === Time Filter ===
        time_filter_start: int = 0,
        time_filter_end: int = 23,
        # === Cooldown ===
        cooldown_minutes: int = 15,
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        
        # LRC params
        self.lrc_period = lrc_period
        self.lrc_signal_period = lrc_signal_period
        
        # MACD params
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        
        # Trade params
        self.tp_coefficient = tp_coefficient
        self.sl_lookback = sl_lookback
        self.sl_deviation = sl_deviation
        self.pullback_lookback = pullback_lookback
        
        # Risk management
        self.trailing_stop_level = trailing_stop_level
        self.use_trailing = use_trailing
        
        # Time filter
        self.time_filter_start = time_filter_start
        self.time_filter_end = time_filter_end
        
        # Cooldown
        self.cooldown_minutes = cooldown_minutes
        
        # Statistics
        self.consecutive_losses = 0
        self.last_trade_won = True
        
        logger.info(
            f"LRC-MACD Strategy initialized - "
            f"LRC: {lrc_period}/{lrc_signal_period} | "
            f"MACD: {macd_fast}/{macd_slow} | "
            f"TP Coef: {tp_coefficient} | SL Lookback: {sl_lookback}"
        )

    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_historical_candles(
        self,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Candle]:
        """Fetch historical candles from MT5"""
        if mt5 is None:
            logger.error("MetaTrader5 module not available")
            return []

        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return []

        rates = mt5.copy_rates_range(
            self.symbol,
            timeframe.mt5_tf,
            start_date,
            end_date,
        )

        if rates is None or len(rates) == 0:
            logger.warning(f"No data returned for {timeframe.name_str}")
            return []

        candles = []
        for r in rates:
            ts = datetime.fromtimestamp(r['time'], tz=timezone.utc)
            candles.append(Candle(
                timestamp=ts,
                open=float(r['open']),
                high=float(r['high']),
                low=float(r['low']),
                close=float(r['close']),
                volume=float(r['tick_volume']),
            ))

        return candles

    # =========================================================================
    # INDICATOR CALCULATIONS
    # =========================================================================
    
    def _calculate_linear_regression(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Linear Regression values for each point.
        Returns the predicted value at the end of each period.
        """
        result = np.full(len(data), np.nan)
        
        if len(data) < period:
            return result
        
        x = np.arange(period)
        x_mean = np.mean(x)
        x_var = np.sum((x - x_mean) ** 2)
        
        for i in range(period - 1, len(data)):
            y = data[i - period + 1:i + 1]
            y_mean = np.mean(y)
            
            # Calculate slope and intercept
            slope = np.sum((x - x_mean) * (y - y_mean)) / x_var
            intercept = y_mean - slope * x_mean
            
            # Predict value at the end of the period
            result[i] = intercept + slope * (period - 1)
        
        return result

    def _calculate_lrc_candles(self, candles: List[Candle]) -> List[LRCCandle]:
        """
        Calculate Linear Regression Candles.
        
        LRC applies linear regression to O, H, L, C separately,
        then creates smoothed candles with a signal line.
        """
        if len(candles) < self.lrc_period + self.lrc_signal_period:
            return []
        
        opens = np.array([c.open for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        
        # Calculate LR for each OHLC component
        lr_open = self._calculate_linear_regression(opens, self.lrc_period)
        lr_high = self._calculate_linear_regression(highs, self.lrc_period)
        lr_low = self._calculate_linear_regression(lows, self.lrc_period)
        lr_close = self._calculate_linear_regression(closes, self.lrc_period)
        
        # Calculate signal line (SMA of LRC close)
        signal = self._calculate_sma(lr_close, self.lrc_signal_period)
        
        lrc_candles = []
        start_idx = self.lrc_period + self.lrc_signal_period - 2
        
        for i in range(start_idx, len(candles)):
            if np.isnan(lr_open[i]) or np.isnan(signal[i]):
                continue
                
            lrc = LRCCandle(
                timestamp=candles[i].timestamp,
                open=lr_open[i],
                high=lr_high[i],
                low=lr_low[i],
                close=lr_close[i],
                signal=signal[i],
                color=1 if lr_close[i] > lr_open[i] else -1,
            )
            lrc_candles.append(lrc)
        
        return lrc_candles

    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        result = np.full(len(data), np.nan)
        if len(data) < period:
            return result
        
        for i in range(period - 1, len(data)):
            if not np.isnan(data[i - period + 1:i + 1]).any():
                result[i] = np.mean(data[i - period + 1:i + 1])
        
        return result

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        result = np.full(len(data), np.nan)
        if len(data) < period:
            return result
        
        multiplier = 2.0 / (period + 1)
        
        # Initialize with SMA
        result[period - 1] = np.mean(data[:period])
        
        # Calculate EMA
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        
        return result

    def _calculate_macd(self, closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate MACD line (fast EMA - slow EMA).
        Returns (macd_line, signal_line).
        Note: Original strategy only uses MACD line (no signal line).
        """
        ema_fast = self._calculate_ema(closes, self.macd_fast)
        ema_slow = self._calculate_ema(closes, self.macd_slow)
        
        macd_line = ema_fast - ema_slow
        
        # Signal line (9-period EMA of MACD) - not used in original but useful
        signal_line = self._calculate_ema(macd_line, 9)
        
        return macd_line, signal_line

    def _calculate_atr(self, candles: List[Candle], period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        if len(candles) < 2:
            return np.array([])
        
        tr = np.zeros(len(candles))
        
        for i in range(1, len(candles)):
            high_low = candles[i].high - candles[i].low
            high_prev_close = abs(candles[i].high - candles[i-1].close)
            low_prev_close = abs(candles[i].low - candles[i-1].close)
            tr[i] = max(high_low, high_prev_close, low_prev_close)
        
        tr[0] = candles[0].high - candles[0].low
        
        atr = np.zeros(len(candles))
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(candles)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr

    # =========================================================================
    # SIGNAL DETECTION
    # =========================================================================
    
    def _check_buy_signal(
        self,
        lrc_candles: List[LRCCandle],
        macd: np.ndarray,
        idx: int,
        lrc_offset: int,
    ) -> bool:
        """
        Check for BUY signal based on LRC-MACD strategy.
        
        Conditions:
        1. MACD[1] > 0 and MACD[2] > 0 (bullish momentum)
        2. LRC[1] is bullish (close > open)
        3. LRC[1] close > signal (above signal line)
        4. LRC[2] close <= signal (was below/at signal - crossover)
        5. LRC[2] is bullish (close > open)
        6. Pullback pattern confirmed in previous bars
        """
        lrc_idx = idx - lrc_offset
        
        if lrc_idx < self.pullback_lookback + 2:
            return False
        if idx < 2:
            return False
        
        lrc_1 = lrc_candles[lrc_idx]      # Current completed bar
        lrc_2 = lrc_candles[lrc_idx - 1]  # Previous bar
        
        md1 = macd[idx - 1] if idx >= 1 else 0
        md2 = macd[idx - 2] if idx >= 2 else 0
        
        # Check for NaN
        if np.isnan(md1) or np.isnan(md2):
            return False
        
        # Condition 1: MACD positive (bullish momentum)
        if not (md1 > 0 and md2 > 0):
            return False
        
        # Condition 2: Current LRC is bullish
        if not lrc_1.is_bullish:
            return False
        
        # Condition 3: Current LRC above signal
        if not lrc_1.above_signal:
            return False
        
        # Condition 4: Previous LRC at/below signal (crossover)
        if lrc_2.close > lrc_2.signal:
            return False
        
        # Condition 5: Previous LRC was bullish
        if not lrc_2.is_bullish:
            return False
        
        # Condition 6: Pullback pattern check
        # Find a bearish bar in the pullback lookback
        j = 0
        for i in range(2, self.pullback_lookback + 2):
            if lrc_idx - i < 0:
                return False
            
            macd_i = macd[idx - i] if idx >= i else 0
            if np.isnan(macd_i) or macd_i <= 0:
                return False
            
            lrc_i = lrc_candles[lrc_idx - i]
            if lrc_i.is_bearish:  # Found bearish candle (pullback)
                j = i
                break
        
        if j == 0:
            return False
        
        # Check that pullback continues (bearish candles)
        for i in range(j, min(j + self.pullback_lookback, lrc_idx)):
            if lrc_idx - i < 0:
                return False
            
            macd_i = macd[idx - i] if idx >= i else 0
            if np.isnan(macd_i) or macd_i <= 0:
                return False
            
            lrc_i = lrc_candles[lrc_idx - i]
            if lrc_i.is_bullish:  # Should be bearish during pullback
                return False
        
        return True

    def _check_sell_signal(
        self,
        lrc_candles: List[LRCCandle],
        macd: np.ndarray,
        idx: int,
        lrc_offset: int,
    ) -> bool:
        """
        Check for SELL signal based on LRC-MACD strategy.
        
        Conditions:
        1. MACD[1] < 0 and MACD[2] < 0 (bearish momentum)
        2. LRC[1] is bearish (close < open)
        3. LRC[1] close < signal (below signal line)
        4. LRC[2] close >= signal (was above/at signal - crossover)
        5. LRC[2] is bearish (close < open)
        6. Pullback pattern confirmed in previous bars
        """
        lrc_idx = idx - lrc_offset
        
        if lrc_idx < self.pullback_lookback + 2:
            return False
        if idx < 2:
            return False
        
        lrc_1 = lrc_candles[lrc_idx]
        lrc_2 = lrc_candles[lrc_idx - 1]
        
        md1 = macd[idx - 1] if idx >= 1 else 0
        md2 = macd[idx - 2] if idx >= 2 else 0
        
        if np.isnan(md1) or np.isnan(md2):
            return False
        
        # Condition 1: MACD negative (bearish momentum)
        if not (md1 < 0 and md2 < 0):
            return False
        
        # Condition 2: Current LRC is bearish
        if not lrc_1.is_bearish:
            return False
        
        # Condition 3: Current LRC below signal
        if not lrc_1.below_signal:
            return False
        
        # Condition 4: Previous LRC at/above signal (crossover)
        if lrc_2.close < lrc_2.signal:
            return False
        
        # Condition 5: Previous LRC was bearish
        if not lrc_2.is_bearish:
            return False
        
        # Condition 6: Pullback pattern check
        j = 0
        for i in range(2, self.pullback_lookback + 2):
            if lrc_idx - i < 0:
                return False
            
            macd_i = macd[idx - i] if idx >= i else 0
            if np.isnan(macd_i) or macd_i >= 0:
                return False
            
            lrc_i = lrc_candles[lrc_idx - i]
            if lrc_i.is_bullish:  # Found bullish candle (pullback)
                j = i
                break
        
        if j == 0:
            return False
        
        # Check that pullback continues (bullish candles)
        for i in range(j, min(j + self.pullback_lookback, lrc_idx)):
            if lrc_idx - i < 0:
                return False
            
            macd_i = macd[idx - i] if idx >= i else 0
            if np.isnan(macd_i) or macd_i >= 0:
                return False
            
            lrc_i = lrc_candles[lrc_idx - i]
            if lrc_i.is_bearish:
                return False
        
        return True

    # =========================================================================
    # BACKTEST EXECUTION
    # =========================================================================
    
    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        progress_callback=None,
    ) -> BacktestResult:
        """
        Run backtest over the specified period.
        Uses M15 timeframe for entries.
        """
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        logger.info(f"Starting LRC-MACD backtest from {start_date} to {end_date}")
        
        # Fetch data with extra lookback for indicators
        lookback_start = start_date - timedelta(days=14)
        
        if progress_callback:
            progress_callback(0.1, "Fetching M15 data...")
        
        candles = self.fetch_historical_candles(Timeframe.M15, lookback_start, end_date)
        
        if not candles:
            logger.error("Failed to fetch historical data")
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                symbol=self.symbol,
            )
        
        logger.info(f"Loaded {len(candles)} M15 candles")
        
        if progress_callback:
            progress_callback(0.3, "Calculating LRC indicators...")
        
        # Calculate indicators
        closes = np.array([c.close for c in candles])
        lrc_candles = self._calculate_lrc_candles(candles)
        macd, macd_signal = self._calculate_macd(closes)
        atr = self._calculate_atr(candles)
        
        if not lrc_candles:
            logger.error("Failed to calculate LRC candles")
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                symbol=self.symbol,
            )
        
        # Calculate offset between candle index and LRC candle index
        lrc_start_time = lrc_candles[0].timestamp
        lrc_offset = 0
        for i, c in enumerate(candles):
            if c.timestamp == lrc_start_time:
                lrc_offset = i
                break
        
        logger.info(f"Calculated {len(lrc_candles)} LRC candles (offset: {lrc_offset})")
        
        if progress_callback:
            progress_callback(0.5, "Running strategy simulation...")
        
        # Simulation state
        trades: List[BacktestTrade] = []
        active_trade: Optional[BacktestTrade] = None
        last_trade_time: Optional[datetime] = None
        trade_id = 0
        equity_curve: List[float] = [self.initial_balance]
        current_balance = self.initial_balance
        
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        # Find candles within test period
        test_indices = [
            i for i, c in enumerate(candles)
            if start_date <= c.timestamp <= end_date and i >= lrc_offset + self.pullback_lookback + 2
        ]
        
        if not test_indices:
            logger.error("No candles in test period")
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                symbol=self.symbol,
            )
        
        total_bars = len(test_indices)
        
        # Bar-by-bar simulation
        for bar_num, idx in enumerate(test_indices):
            candle = candles[idx]
            lrc_idx = idx - lrc_offset
            
            if progress_callback and bar_num % 100 == 0:
                progress = 0.5 + (bar_num / total_bars) * 0.45
                progress_callback(progress, f"Processing bar {bar_num}/{total_bars}...")
            
            # Time filter
            hour = candle.timestamp.hour
            if hour < self.time_filter_start or hour >= self.time_filter_end:
                continue
            
            current_atr = atr[idx]
            
            # Check active trade for exit
            if active_trade:
                is_buy = active_trade.signal_type == SignalType.BUY
                
                # Track max favorable/adverse
                if is_buy:
                    favorable = candle.high - active_trade.entry_price
                    adverse = active_trade.entry_price - candle.low
                else:
                    favorable = active_trade.entry_price - candle.low
                    adverse = candle.high - active_trade.entry_price
                
                active_trade.max_favorable = max(active_trade.max_favorable, favorable)
                active_trade.max_adverse = max(active_trade.max_adverse, adverse)
                
                trade_closed = False
                pnl_points = 0.0
                
                # Check TP/SL
                if is_buy:
                    # Trailing stop logic
                    if self.use_trailing and favorable > 0:
                        target_dist = active_trade.take_profit - active_trade.entry_price
                        if favorable >= target_dist * self.trailing_stop_level:
                            # Move SL to breakeven + small buffer
                            new_sl = active_trade.entry_price + (favorable * 0.3)
                            if new_sl > active_trade.stop_loss:
                                active_trade.stop_loss = new_sl
                    
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
                    if self.use_trailing and favorable > 0:
                        target_dist = active_trade.entry_price - active_trade.take_profit
                        if favorable >= target_dist * self.trailing_stop_level:
                            new_sl = active_trade.entry_price - (favorable * 0.3)
                            if new_sl < active_trade.stop_loss:
                                active_trade.stop_loss = new_sl
                    
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
                    active_trade.duration_minutes = int(
                        (active_trade.exit_time - active_trade.entry_time).total_seconds() / 60
                    )
                    
                    # Track consecutive wins/losses
                    if active_trade.won:
                        consecutive_wins += 1
                        consecutive_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    else:
                        consecutive_losses += 1
                        consecutive_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    
                    current_balance += pnl_points * 10  # Simplified PnL
                    equity_curve.append(current_balance)
                    trades.append(active_trade)
                    last_trade_time = candle.timestamp
                    active_trade = None
                    
                continue
            
            # Cooldown check
            if last_trade_time:
                elapsed = candle.timestamp - last_trade_time
                if elapsed < timedelta(minutes=self.cooldown_minutes):
                    continue
            
            # Check for entry signals
            if lrc_idx >= len(lrc_candles):
                continue
            
            # BUY Signal
            if self._check_buy_signal(lrc_candles, macd, idx, lrc_offset):
                entry_price = candle.close
                
                # SL = lowest low of last N bars - deviation
                lookback_lows = [candles[i].low for i in range(max(0, idx - self.sl_lookback), idx + 1)]
                sl_level = min(lookback_lows) - self.sl_deviation
                
                sl_distance = entry_price - sl_level
                tp_distance = sl_distance * self.tp_coefficient
                tp_level = entry_price + tp_distance
                
                trade_id += 1
                active_trade = BacktestTrade(
                    id=trade_id,
                    signal_type=SignalType.BUY,
                    entry_time=candle.timestamp,
                    entry_price=entry_price,
                    stop_loss=sl_level,
                    take_profit=tp_level,
                )
                
                logger.debug(
                    f"BUY @ {entry_price:.2f} | SL: {sl_level:.2f} | TP: {tp_level:.2f} | "
                    f"R:R = 1:{self.tp_coefficient:.1f}"
                )
                continue
            
            # SELL Signal
            if self._check_sell_signal(lrc_candles, macd, idx, lrc_offset):
                entry_price = candle.close
                
                # SL = highest high of last N bars + deviation
                lookback_highs = [candles[i].high for i in range(max(0, idx - self.sl_lookback), idx + 1)]
                sl_level = max(lookback_highs) + self.sl_deviation
                
                sl_distance = sl_level - entry_price
                tp_distance = sl_distance * self.tp_coefficient
                tp_level = entry_price - tp_distance
                
                trade_id += 1
                active_trade = BacktestTrade(
                    id=trade_id,
                    signal_type=SignalType.SELL,
                    entry_time=candle.timestamp,
                    entry_price=entry_price,
                    stop_loss=sl_level,
                    take_profit=tp_level,
                )
                
                logger.debug(
                    f"SELL @ {entry_price:.2f} | SL: {sl_level:.2f} | TP: {tp_level:.2f} | "
                    f"R:R = 1:{self.tp_coefficient:.1f}"
                )
        
        if progress_callback:
            progress_callback(0.95, "Calculating statistics...")
        
        # Calculate final statistics
        result = self._calculate_statistics(
            trades=trades,
            equity_curve=equity_curve,
            start_date=start_date,
            end_date=end_date,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            candles=candles,
        )
        
        logger.info(
            f"LRC-MACD Backtest complete: {result.total_trades} trades, "
            f"{result.win_rate:.1f}% win rate, {result.total_pnl_pips:+.1f} pips"
        )
        
        return result

    def _calculate_statistics(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[float],
        start_date: datetime,
        end_date: datetime,
        max_consecutive_wins: int,
        max_consecutive_losses: int,
        candles: List[Candle],
    ) -> BacktestResult:
        """Calculate backtest statistics"""
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.won)
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl_pips = sum(t.pnl_pips for t in trades)
        
        wins_pips = [t.pnl_pips for t in trades if t.won]
        losses_pips = [abs(t.pnl_pips) for t in trades if not t.won]
        
        avg_win = np.mean(wins_pips) if wins_pips else 0
        avg_loss = np.mean(losses_pips) if losses_pips else 0
        
        best_trade = max(wins_pips) if wins_pips else 0
        worst_trade = -max(losses_pips) if losses_pips else 0
        
        gross_profit = sum(wins_pips)
        gross_loss = sum(losses_pips)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown calculation
        peak = equity_curve[0]
        max_dd = 0
        max_dd_pct = 0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        # Average trade duration
        durations = [t.duration_minutes for t in trades if t.duration_minutes > 0]
        avg_duration = timedelta(minutes=np.mean(durations)) if durations else None
        
        final_balance = equity_curve[-1] if equity_curve else self.initial_balance
        total_return = final_balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100
        
        # Buy and hold comparison
        if candles:
            buy_hold_return = candles[-1].close - candles[0].close
            buy_hold_pct = (buy_hold_return / candles[0].close) * 100
        else:
            buy_hold_return = 0
            buy_hold_pct = 0
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            symbol=self.symbol,
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl_pips=total_pnl_pips,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            consecutive_wins=max_consecutive_wins,
            consecutive_losses=max_consecutive_losses,
            avg_trade_duration=avg_duration,
            buy_hold_return=buy_hold_return,
            buy_hold_pct=buy_hold_pct,
            trades=trades,
            equity_curve=equity_curve,
        )


# Quick test function
def test_strategy():
    """Quick test of the LRC-MACD strategy"""
    from datetime import datetime, timedelta
    
    strategy = LRCMACDStrategy(
        symbol="XAUUSD",
        lrc_period=11,
        lrc_signal_period=5,
        macd_fast=34,
        macd_slow=144,
        tp_coefficient=1.0,
        sl_lookback=10,
        sl_deviation=0.6,
        pullback_lookback=4,
    )
    
    end = datetime.now()
    start = end - timedelta(days=30)
    
    result = strategy.run_backtest(start, end)
    
    print("=" * 50)
    print("LRC-MACD STRATEGY BACKTEST RESULTS")
    print("=" * 50)
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Wins: {result.winning_trades}")
    print(f"Losses: {result.losing_trades}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Total PnL: {result.total_pnl_pips:+.1f} pips")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Max Drawdown: {result.max_drawdown_pct:.1f}%")
    print(f"Max Consecutive Losses: {result.consecutive_losses}")
    
    return result


if __name__ == "__main__":
    test_strategy()

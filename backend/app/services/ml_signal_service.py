"""
ML Signal Service - Daily Signal Generator
Provides ML-based trading signals for XAUUSD

Model: XGBoost with multi-timeframe features
Performance: 54% Win Rate, 17+ signals/day
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import threading

import numpy as np
import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False


@dataclass
class MLSignal:
    """ML-generated trading signal"""
    timestamp: datetime
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    h1_trend: int
    h4_trend: int
    atr: float
    features: Dict[str, float]
    
    @property
    def risk_pips(self) -> float:
        if self.direction == "LONG":
            return (self.entry_price - self.stop_loss) * 10
        return (self.stop_loss - self.entry_price) * 10
    
    @property
    def reward_pips(self) -> float:
        if self.direction == "LONG":
            return (self.take_profit - self.entry_price) * 10
        return (self.entry_price - self.take_profit) * 10
    
    @property
    def risk_reward(self) -> float:
        if self.risk_pips > 0:
            return self.reward_pips / self.risk_pips
        return 0


class MLSignalService:
    """
    Machine Learning Signal Service
    
    Generates trading signals using trained XGBoost model.
    Optimized for daily signals with 54%+ win rate.
    """
    
    # Model settings (from V8 optimization)
    PROB_THRESHOLD = 0.35
    SL_MULTIPLIER = 1.2
    TP_MULTIPLIER = 1.2
    MIN_GAP_BARS = 2  # 30 min between signals
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features: List[str] = []
        self.is_loaded = False
        
        # Data caches
        self._m15_df: Optional[pd.DataFrame] = None
        self._h1_df: Optional[pd.DataFrame] = None
        self._h4_df: Optional[pd.DataFrame] = None
        self._last_update: Optional[datetime] = None
        self._last_signal_time: Optional[datetime] = None
        
        # Signal history
        self.signal_history: List[MLSignal] = []
        
        # Paths
        self._model_path = Path(__file__).parent / "ml_model.pkl"
        self._scaler_path = Path(__file__).parent / "ml_scaler.pkl"
        self._features_path = Path(__file__).parent / "ml_features.txt"
        
        self._lock = threading.Lock()
        
    def load_model(self) -> bool:
        """Load trained model from disk"""
        if not JOBLIB_AVAILABLE:
            logger.error("joblib not available - cannot load model")
            return False
            
        try:
            if not self._model_path.exists():
                logger.error(f"Model file not found: {self._model_path}")
                return False
                
            self.model = joblib.load(self._model_path)
            self.scaler = joblib.load(self._scaler_path)
            
            with open(self._features_path, 'r') as f:
                self.features = [line.strip() for line in f.readlines()]
            
            self.is_loaded = True
            logger.info(f"ML model loaded: {len(self.features)} features")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return False
    
    def _ensure_mt5(self) -> bool:
        """Ensure MT5 is connected"""
        if not MT5_AVAILABLE:
            return False
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
        return True
    
    def _fetch_data(self) -> bool:
        """Fetch latest candle data from MT5"""
        if not self._ensure_mt5():
            return False
            
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 30 days of history
            
            # M15 data
            rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, start_date, end_date)
            if rates is None:
                return False
            self._m15_df = pd.DataFrame(rates)
            self._m15_df['time'] = pd.to_datetime(self._m15_df['time'], unit='s')
            self._m15_df.set_index('time', inplace=True)
            
            # H1 data
            rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H1, start_date, end_date)
            if rates is None:
                return False
            self._h1_df = pd.DataFrame(rates)
            self._h1_df['time'] = pd.to_datetime(self._h1_df['time'], unit='s')
            self._h1_df.set_index('time', inplace=True)
            
            # H4 data
            rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H4, start_date, end_date)
            if rates is None:
                return False
            self._h4_df = pd.DataFrame(rates)
            self._h4_df['time'] = pd.to_datetime(self._h4_df['time'], unit='s')
            self._h4_df.set_index('time', inplace=True)
            
            self._last_update = datetime.now()
            logger.debug(f"Data updated: M15={len(self._m15_df)}, H1={len(self._h1_df)}, H4={len(self._h4_df)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return False
    
    def _ema(self, s: pd.Series, p: int) -> pd.Series:
        return s.ewm(span=p, adjust=False).mean()
    
    def _rsi(self, s: pd.Series, p: int = 14) -> pd.Series:
        d = s.diff()
        g = d.where(d > 0, 0).rolling(p).mean()
        l = (-d.where(d < 0, 0)).rolling(p).mean()
        return 100 - 100 / (1 + g/l)
    
    def _atr(self, df: pd.DataFrame, p: int = 14) -> pd.Series:
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        return tr.rolling(p).mean()
    
    def _build_features(self) -> Optional[pd.DataFrame]:
        """Build feature matrix from current data"""
        if self._m15_df is None or self._h1_df is None or self._h4_df is None:
            return None
            
        df = self._m15_df.copy()
        
        # Basic
        df['ret'] = df['close'].pct_change()
        df['ret5'] = df['close'].pct_change(5)
        df['vol10'] = df['ret'].rolling(10).std()
        df['vol20'] = df['ret'].rolling(20).std()
        
        # Candles
        df['body'] = (df['close'] - df['open']) / df['open'] * 100
        df['range'] = df['high'] - df['low']
        df['bull'] = (df['close'] > df['open']).astype(int)
        
        # EMAs
        for p in [8, 14, 21, 34, 55, 100]:
            df[f'ema{p}'] = self._ema(df['close'], p)
            df[f'cvs{p}'] = (df['close'] - df[f'ema{p}']) / df[f'ema{p}'] * 100
        
        # EMA alignment
        df['ema_align'] = (
            (df['ema8'] > df['ema14']).astype(int) + 
            (df['ema14'] > df['ema21']).astype(int) +
            (df['ema21'] > df['ema55']).astype(int) +
            (df['ema55'] > df['ema100']).astype(int)
        ) - 2
        
        df['ema21_slope'] = df['ema21'].pct_change(3) * 100
        
        # RSI
        df['rsi14'] = self._rsi(df['close'], 14)
        df['rsi7'] = self._rsi(df['close'], 7)
        
        # ATR
        df['atr'] = self._atr(df, 14)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # MACD
        df['macd'] = self._ema(df['close'], 12) - self._ema(df['close'], 26)
        df['macd_sig'] = self._ema(df['macd'], 9)
        df['macd_hist'] = df['macd'] - df['macd_sig']
        
        # Stoch
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # S/R
        df['high20'] = df['high'].rolling(20).max()
        df['low20'] = df['low'].rolling(20).min()
        df['dist_hi'] = (df['high20'] - df['close']) / df['atr']
        df['dist_lo'] = (df['close'] - df['low20']) / df['atr']
        
        # Volume
        if 'tick_volume' in df.columns:
            df['vol_ma'] = df['tick_volume'].rolling(10).mean()
            df['vol_ratio'] = df['tick_volume'] / df['vol_ma']
        
        # Time features
        df['hour'] = df.index.hour
        df['dow'] = df.index.dayofweek
        df['london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        df['active'] = ((df['london'] == 1) | (df['ny'] == 1)).astype(int)
        
        # H1 features
        h1_df = self._h1_df.copy()
        h1_df['h1_ema21'] = self._ema(h1_df['close'], 21)
        h1_df['h1_ema55'] = self._ema(h1_df['close'], 55)
        h1_df['h1_rsi'] = self._rsi(h1_df['close'], 14)
        h1_df['h1_trend'] = np.where(h1_df['h1_ema21'] > h1_df['h1_ema55'], 1, -1)
        
        # H4 features
        h4_df = self._h4_df.copy()
        h4_df['h4_ema21'] = self._ema(h4_df['close'], 21)
        h4_df['h4_ema55'] = self._ema(h4_df['close'], 55)
        h4_df['h4_trend'] = np.where(h4_df['h4_ema21'] > h4_df['h4_ema55'], 1, -1)
        
        # Merge HTF features
        for c in ['h1_trend', 'h1_rsi']:
            df[c] = df.index.map(lambda x: h1_df.loc[:x, c].iloc[-1] if len(h1_df.loc[:x]) > 0 else 0)
        df['h4_trend'] = df.index.map(lambda x: h4_df.loc[:x, 'h4_trend'].iloc[-1] if len(h4_df.loc[:x]) > 0 else 0)
        
        # MTF alignment
        df['mtf'] = df['h1_trend'] + df['h4_trend']
        
        return df
    
    def check_signal(self) -> Optional[MLSignal]:
        """
        Check for new ML signal based on current market data.
        
        Returns MLSignal if conditions are met, None otherwise.
        """
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        with self._lock:
            # Fetch fresh data
            if not self._fetch_data():
                return None
            
            # Build features
            df = self._build_features()
            if df is None or len(df) < 100:
                return None
            
            # Check cooldown
            if self._last_signal_time:
                minutes_since = (datetime.now() - self._last_signal_time).total_seconds() / 60
                if minutes_since < 30:  # 30 min cooldown
                    return None
            
            # Get latest row
            latest = df.iloc[-1]
            
            # Skip if RSI extreme (light filter)
            if latest['rsi14'] > 75 or latest['rsi14'] < 25:
                return None
            
            # Prepare features
            try:
                X = df[self.features].iloc[-1:].values
                X_scaled = self.scaler.transform(X)
                proba = self.model.predict_proba(X_scaled)[0]
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return None
            
            # Check signal
            p_long = proba[1] if len(proba) > 1 else 0
            p_short = proba[2] if len(proba) > 2 else 0
            
            signal = None
            entry = latest['close']
            atr = latest['atr']
            
            if p_long >= self.PROB_THRESHOLD and latest['h1_trend'] > 0:
                signal = MLSignal(
                    timestamp=datetime.now(timezone.utc),
                    direction="LONG",
                    entry_price=entry,
                    stop_loss=entry - atr * self.SL_MULTIPLIER,
                    take_profit=entry + atr * self.TP_MULTIPLIER,
                    confidence=p_long,
                    h1_trend=int(latest['h1_trend']),
                    h4_trend=int(latest['h4_trend']),
                    atr=atr,
                    features={
                        'rsi14': latest['rsi14'],
                        'ema_align': latest['ema_align'],
                        'mtf': latest['mtf'],
                    }
                )
            elif p_short >= self.PROB_THRESHOLD and latest['h1_trend'] < 0:
                signal = MLSignal(
                    timestamp=datetime.now(timezone.utc),
                    direction="SHORT",
                    entry_price=entry,
                    stop_loss=entry + atr * self.SL_MULTIPLIER,
                    take_profit=entry - atr * self.TP_MULTIPLIER,
                    confidence=p_short,
                    h1_trend=int(latest['h1_trend']),
                    h4_trend=int(latest['h4_trend']),
                    atr=atr,
                    features={
                        'rsi14': latest['rsi14'],
                        'ema_align': latest['ema_align'],
                        'mtf': latest['mtf'],
                    }
                )
            
            if signal:
                self._last_signal_time = datetime.now()
                self.signal_history.append(signal)
                logger.info(f"ML Signal: {signal.direction} @ {signal.entry_price:.2f} "
                           f"(conf={signal.confidence:.2f}, TP={signal.take_profit:.2f}, SL={signal.stop_loss:.2f})")
            
            return signal
    
    def get_current_analysis(self) -> Dict:
        """Get current market analysis from ML perspective"""
        if self._m15_df is None:
            self._fetch_data()
        
        if self._m15_df is None:
            return {"error": "No data available"}
        
        df = self._build_features()
        if df is None:
            return {"error": "Failed to build features"}
        
        latest = df.iloc[-1]
        
        # Convert numpy types to Python native types for JSON serialization
        h1_trend_val = int(latest['h1_trend']) if not pd.isna(latest['h1_trend']) else 0
        h4_trend_val = int(latest['h4_trend']) if not pd.isna(latest['h4_trend']) else 0
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": float(latest['close']),
            "h1_trend": "BULLISH" if h1_trend_val > 0 else "BEARISH",
            "h4_trend": "BULLISH" if h4_trend_val > 0 else "BEARISH",
            "mtf_aligned": bool(h1_trend_val == h4_trend_val),
            "rsi14": float(latest['rsi14']) if not pd.isna(latest['rsi14']) else 50.0,
            "ema_alignment": int(latest['ema_align']) if not pd.isna(latest['ema_align']) else 0,
            "atr": float(latest['atr']) if not pd.isna(latest['atr']) else 0.0,
            "stoch_k": float(latest['stoch_k']) if not pd.isna(latest['stoch_k']) else 50.0,
            "session": {
                "london": bool(latest['london']) if not pd.isna(latest['london']) else False,
                "ny": bool(latest['ny']) if not pd.isna(latest['ny']) else False,
                "overlap": bool(latest['overlap']) if not pd.isna(latest['overlap']) else False,
            }
        }
    
    def get_signal_history(self, limit: int = 20) -> List[Dict]:
        """Get recent signal history"""
        signals = self.signal_history[-limit:]
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "direction": s.direction,
                "entry_price": float(s.entry_price),
                "stop_loss": float(s.stop_loss),
                "take_profit": float(s.take_profit),
                "confidence": float(s.confidence),
                "risk_pips": float(s.risk_pips),
                "reward_pips": float(s.reward_pips),
                "risk_reward": float(s.risk_reward),
            }
            for s in signals
        ]


# Singleton instance
_ml_service: Optional[MLSignalService] = None


def get_ml_signal_service() -> MLSignalService:
    """Get singleton ML signal service instance"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLSignalService()
        _ml_service.load_model()
    return _ml_service

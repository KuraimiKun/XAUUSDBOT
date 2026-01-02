"""
Machine Learning Trading Model for XAUUSD

Trains on historical data to predict profitable trade setups.
Uses features from price action, technical indicators, and market structure.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    from sklearn.utils import class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class TradeLabel:
    """Label for a potential trade setup"""
    timestamp: datetime
    direction: int  # 1=BUY, -1=SELL, 0=NO_TRADE
    entry_price: float
    outcome: int  # 1=WIN, 0=LOSS
    pnl_pips: float
    duration_minutes: int


@dataclass 
class MLPrediction:
    """Prediction from the ML model"""
    timestamp: datetime
    direction: int  # 1=BUY, -1=SELL, 0=NO_TRADE
    confidence: float
    entry_price: float
    predicted_sl: float
    predicted_tp: float


class MLTradingModel:
    """
    Machine Learning model for XAUUSD trading predictions.
    
    Features used:
    - Price action (returns, volatility, range)
    - Technical indicators (EMA, RSI, MACD, ATR)
    - Market structure (trend, support/resistance)
    - Time-based features (hour, day of week)
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        lookback_bars: int = 50,
        prediction_horizon: int = 12,  # Bars to look ahead for labeling
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
        min_rr_ratio: float = 2.0,
    ):
        self.symbol = symbol
        self.lookback_bars = lookback_bars
        self.prediction_horizon = prediction_horizon
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.min_rr_ratio = min_rr_ratio
        
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Model save path
        self.model_path = Path(__file__).parent / "ml_model.pkl"
        
        logger.info(f"ML Trading Model initialized for {symbol}")

    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_historical_data(
        self,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from MT5"""
        if mt5 is None:
            logger.error("MetaTrader5 not available")
            return pd.DataFrame()
        
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return pd.DataFrame()
        
        rates = mt5.copy_rates_range(
            self.symbol,
            timeframe,
            start_date,
            end_date,
        )
        
        if rates is None or len(rates) == 0:
            logger.warning("No data returned from MT5")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        logger.info(f"Fetched {len(df)} bars from {start_date} to {end_date}")
        return df

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = self._calculate_ema(series, fast)
        ema_slow = self._calculate_ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix from OHLCV data"""
        features = pd.DataFrame(index=df.index)
        
        # === PRICE ACTION FEATURES ===
        # Returns at different intervals
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}'] = df['close'].pct_change(period)
        
        # Volatility (rolling std of returns)
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        # Price range relative to close
        features['range_pct'] = (df['high'] - df['low']) / df['close']
        features['body_pct'] = (df['close'] - df['open']).abs() / df['close']
        features['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        features['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        
        # Candle direction
        features['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # Gap from previous close
        features['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # === MOVING AVERAGES ===
        for period in [8, 14, 21, 50, 100, 200]:
            ema = self._calculate_ema(df['close'], period)
            features[f'ema_{period}_dist'] = (df['close'] - ema) / df['close']
            features[f'ema_{period}_slope'] = ema.diff(3) / ema
        
        # EMA crossovers
        ema_8 = self._calculate_ema(df['close'], 8)
        ema_21 = self._calculate_ema(df['close'], 21)
        ema_50 = self._calculate_ema(df['close'], 50)
        features['ema_8_21_cross'] = (ema_8 > ema_21).astype(int)
        features['ema_21_50_cross'] = (ema_21 > ema_50).astype(int)
        
        # === RSI ===
        rsi = self._calculate_rsi(df['close'], 14)
        features['rsi'] = rsi
        features['rsi_ma'] = rsi.rolling(5).mean()
        features['rsi_overbought'] = (rsi > 70).astype(int)
        features['rsi_oversold'] = (rsi < 30).astype(int)
        
        # === MACD ===
        macd, signal, histogram = self._calculate_macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        features['macd_cross'] = (macd > signal).astype(int)
        features['macd_positive'] = (macd > 0).astype(int)
        
        # === ATR ===
        atr = self._calculate_atr(df)
        features['atr'] = atr
        features['atr_pct'] = atr / df['close']
        features['atr_change'] = atr.pct_change(5)
        
        # === BOLLINGER BANDS ===
        bb_upper, bb_mid, bb_lower = self._calculate_bollinger_bands(df['close'])
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_mid
        
        # === STOCHASTIC ===
        stoch_k, stoch_d = self._calculate_stochastic(df)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        features['stoch_cross'] = (stoch_k > stoch_d).astype(int)
        
        # === VOLUME FEATURES ===
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_change'] = df['volume'].pct_change()
        
        # === TIME FEATURES ===
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['is_london'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        features['is_ny'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
        features['is_overlap'] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(int)
        
        # === PRICE LEVEL FEATURES ===
        # Distance from recent high/low
        features['dist_from_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        features['dist_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        
        # Support/Resistance zones (simplified)
        features['near_resistance'] = (features['dist_from_high_20'] < 0.002).astype(int)
        features['near_support'] = (features['dist_from_low_20'] < 0.002).astype(int)
        
        # === MOMENTUM FEATURES ===
        # Rate of change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = df['close'].pct_change(period) * 100
        
        # Momentum
        features['momentum_10'] = df['close'] - df['close'].shift(10)
        features['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # === PATTERN FEATURES ===
        # Consecutive candles
        bullish = (df['close'] > df['open']).astype(int)
        features['consecutive_bullish'] = bullish.groupby((bullish != bullish.shift()).cumsum()).cumcount() + 1
        features['consecutive_bullish'] = features['consecutive_bullish'] * bullish
        
        bearish = (df['close'] < df['open']).astype(int)
        features['consecutive_bearish'] = bearish.groupby((bearish != bearish.shift()).cumsum()).cumcount() + 1
        features['consecutive_bearish'] = features['consecutive_bearish'] * bearish
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        return features

    # =========================================================================
    # LABELING (TARGET VARIABLE)
    # =========================================================================
    
    def create_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Create labels for training.
        
        Labels:
        - 1 = BUY signal (price goes up by TP before hitting SL)
        - -1 = SELL signal (price goes down by TP before hitting SL)
        - 0 = NO TRADE (neither TP nor SL hit, or mixed result)
        """
        atr = self._calculate_atr(df)
        labels = pd.Series(0, index=df.index)
        
        for i in range(len(df) - self.prediction_horizon):
            idx = df.index[i]
            entry_price = df['close'].iloc[i]
            current_atr = atr.iloc[i]
            
            if pd.isna(current_atr) or current_atr == 0:
                continue
            
            sl_dist = current_atr * self.sl_atr_mult
            tp_dist = current_atr * self.tp_atr_mult
            
            # Look ahead
            future_data = df.iloc[i+1:i+1+self.prediction_horizon]
            
            if len(future_data) < self.prediction_horizon:
                continue
            
            future_highs = future_data['high']
            future_lows = future_data['low']
            
            # Check LONG trade outcome
            long_sl = entry_price - sl_dist
            long_tp = entry_price + tp_dist
            
            long_sl_hit = future_lows.min() <= long_sl
            long_tp_hit = future_highs.max() >= long_tp
            
            # Check which was hit first
            if long_tp_hit and not long_sl_hit:
                labels.iloc[i] = 1  # BUY WIN
            elif not long_tp_hit and long_sl_hit:
                pass  # BUY LOSS - could be SELL opportunity
            elif long_tp_hit and long_sl_hit:
                # Both hit - check which first
                tp_bar = (future_highs >= long_tp).idxmax()
                sl_bar = (future_lows <= long_sl).idxmax()
                if tp_bar < sl_bar:
                    labels.iloc[i] = 1  # BUY WIN
            
            # Check SHORT trade outcome
            short_sl = entry_price + sl_dist
            short_tp = entry_price - tp_dist
            
            short_sl_hit = future_highs.max() >= short_sl
            short_tp_hit = future_lows.min() <= short_tp
            
            if short_tp_hit and not short_sl_hit:
                if labels.iloc[i] != 1:  # Don't overwrite BUY
                    labels.iloc[i] = -1  # SELL WIN
            elif short_tp_hit and short_sl_hit:
                tp_bar = (future_lows <= short_tp).idxmax()
                sl_bar = (future_highs >= short_sl).idxmax()
                if tp_bar < sl_bar and labels.iloc[i] != 1:
                    labels.iloc[i] = -1  # SELL WIN
        
        return labels

    # =========================================================================
    # TRAINING
    # =========================================================================
    
    def prepare_training_data(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M15 if mt5 else 15
        
        # Fetch data
        df = self.fetch_historical_data(timeframe, start_date, end_date)
        
        if df.empty:
            return np.array([]), np.array([])
        
        logger.info(f"Creating features for {len(df)} bars...")
        features = self.create_features(df)
        
        logger.info("Creating labels...")
        labels = self.create_labels(df, features)
        
        # Remove rows with NaN
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        # Remove last N rows (can't verify outcome)
        features = features.iloc[:-self.prediction_horizon]
        labels = labels.iloc[:-self.prediction_horizon]
        
        logger.info(f"Final dataset: {len(features)} samples")
        logger.info(f"Label distribution: BUY={sum(labels==1)}, SELL={sum(labels==-1)}, NO_TRADE={sum(labels==0)}")
        
        return features.values, labels.values

    def train(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe: int = None,
        test_size: float = 0.2,
        model_type: str = "random_forest",
    ) -> Dict:
        """Train the ML model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for training")
        
        X, y = self.prepare_training_data(start_date, end_date, timeframe)
        
        if len(X) == 0:
            logger.error("No training data available")
            return {}
        
        # Convert to binary classification (trade vs no trade) or keep multi-class
        # For simplicity, let's do: 1 = profitable trade opportunity, 0 = no trade
        # Combine BUY (1) and SELL (-1) as "trade", 0 as "no trade"
        y_binary = (y != 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, shuffle=False  # Time-series: no shuffle
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        logger.info(f"Training {model_type} model...")
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Select model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=20,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.is_trained = True
        
        results = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_)) if hasattr(self.model, 'feature_importances_') else {},
        }
        
        logger.info(f"Training complete!")
        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1: {test_f1:.4f}")
        logger.info(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Also train direction model
        self._train_direction_model(X, y, test_size)
        
        return results

    def _train_direction_model(self, X: np.ndarray, y: np.ndarray, test_size: float):
        """Train a separate model to predict direction (BUY vs SELL)"""
        # Only use samples where there's a trade signal
        trade_mask = y != 0
        X_trades = X[trade_mask]
        y_trades = y[trade_mask]
        
        if len(X_trades) < 100:
            logger.warning("Not enough trade samples for direction model")
            self.direction_model = None
            return
        
        # Convert to binary: 1=BUY, 0=SELL
        y_direction = (y_trades == 1).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_trades, y_direction, test_size=test_size, shuffle=False
        )
        
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.direction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
        )
        
        self.direction_model.fit(X_train_scaled, y_train)
        
        y_pred = self.direction_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Direction Model Accuracy: {accuracy:.4f}")

    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float, int]:
        """
        Make prediction on current market data.
        
        Returns:
            (trade_signal, confidence, direction)
            trade_signal: 1=trade, 0=no trade
            confidence: probability
            direction: 1=BUY, -1=SELL, 0=no direction
        """
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return 0, 0.0, 0
        
        features = self.create_features(df)
        
        if features.empty or features.iloc[-1].isna().any():
            return 0, 0.0, 0
        
        X = features.iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        # Predict trade opportunity
        trade_proba = self.model.predict_proba(X_scaled)[0]
        trade_signal = 1 if trade_proba[1] > 0.6 else 0  # Higher threshold
        confidence = trade_proba[1]
        
        # Predict direction
        direction = 0
        if trade_signal and hasattr(self, 'direction_model') and self.direction_model is not None:
            dir_pred = self.direction_model.predict(X_scaled)[0]
            dir_proba = self.direction_model.predict_proba(X_scaled)[0]
            direction = 1 if dir_pred == 1 else -1
            # Adjust confidence with direction confidence
            confidence = confidence * max(dir_proba)
        
        return trade_signal, confidence, direction

    # =========================================================================
    # BACKTESTING
    # =========================================================================
    
    def backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe: int = None,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        min_confidence: float = 0.6,
    ) -> Dict:
        """Backtest the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before backtesting")
        
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M15 if mt5 else 15
        
        df = self.fetch_historical_data(timeframe, start_date, end_date)
        
        if df.empty:
            return {}
        
        atr = self._calculate_atr(df)
        features = self.create_features(df)
        
        # Remove NaN
        valid_start = max(self.lookback_bars, 200)  # Need history for indicators
        
        trades = []
        balance = initial_balance
        equity_curve = [balance]
        
        in_trade = False
        trade_entry = None
        trade_direction = None
        trade_sl = None
        trade_tp = None
        trade_entry_time = None
        
        logger.info(f"Backtesting from {start_date} to {end_date}...")
        
        for i in range(valid_start, len(df) - 1):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            current_high = df['high'].iloc[i+1]  # Next bar
            current_low = df['low'].iloc[i+1]    # Next bar
            current_atr = atr.iloc[i]
            
            if pd.isna(current_atr):
                continue
            
            # Check existing trade
            if in_trade:
                if trade_direction == 1:  # LONG
                    if current_low <= trade_sl:
                        # Stop loss hit
                        pnl = trade_sl - trade_entry
                        trades.append({
                            'entry_time': trade_entry_time,
                            'exit_time': current_time,
                            'direction': 'BUY',
                            'entry': trade_entry,
                            'exit': trade_sl,
                            'pnl_points': pnl,
                            'pnl_pips': pnl * 10,
                            'won': False,
                        })
                        balance += pnl * 10
                        in_trade = False
                    elif current_high >= trade_tp:
                        # Take profit hit
                        pnl = trade_tp - trade_entry
                        trades.append({
                            'entry_time': trade_entry_time,
                            'exit_time': current_time,
                            'direction': 'BUY',
                            'entry': trade_entry,
                            'exit': trade_tp,
                            'pnl_points': pnl,
                            'pnl_pips': pnl * 10,
                            'won': True,
                        })
                        balance += pnl * 10
                        in_trade = False
                else:  # SHORT
                    if current_high >= trade_sl:
                        pnl = trade_entry - trade_sl
                        trades.append({
                            'entry_time': trade_entry_time,
                            'exit_time': current_time,
                            'direction': 'SELL',
                            'entry': trade_entry,
                            'exit': trade_sl,
                            'pnl_points': pnl,
                            'pnl_pips': pnl * 10,
                            'won': False,
                        })
                        balance += pnl * 10
                        in_trade = False
                    elif current_low <= trade_tp:
                        pnl = trade_entry - trade_tp
                        trades.append({
                            'entry_time': trade_entry_time,
                            'exit_time': current_time,
                            'direction': 'SELL',
                            'entry': trade_entry,
                            'exit': trade_tp,
                            'pnl_points': pnl,
                            'pnl_pips': pnl * 10,
                            'won': True,
                        })
                        balance += pnl * 10
                        in_trade = False
                
                equity_curve.append(balance)
                continue
            
            # Check for new trade
            if features.iloc[i].isna().any():
                continue
            
            X = features.iloc[i:i+1].values
            X_scaled = self.scaler.transform(X)
            
            trade_proba = self.model.predict_proba(X_scaled)[0]
            
            if trade_proba[1] < min_confidence:
                equity_curve.append(balance)
                continue
            
            # Get direction
            if hasattr(self, 'direction_model') and self.direction_model is not None:
                dir_pred = self.direction_model.predict(X_scaled)[0]
                direction = 1 if dir_pred == 1 else -1
            else:
                equity_curve.append(balance)
                continue
            
            # Enter trade
            trade_entry = current_price
            trade_direction = direction
            trade_entry_time = current_time
            
            if direction == 1:
                trade_sl = trade_entry - current_atr * self.sl_atr_mult
                trade_tp = trade_entry + current_atr * self.tp_atr_mult
            else:
                trade_sl = trade_entry + current_atr * self.sl_atr_mult
                trade_tp = trade_entry - current_atr * self.tp_atr_mult
            
            in_trade = True
            equity_curve.append(balance)
        
        # Calculate statistics
        if not trades:
            return {"error": "No trades generated"}
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['won'])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl_pips'] for t in trades)
        
        wins_pnl = [t['pnl_pips'] for t in trades if t['won']]
        losses_pnl = [abs(t['pnl_pips']) for t in trades if not t['won']]
        
        avg_win = np.mean(wins_pnl) if wins_pnl else 0
        avg_loss = np.mean(losses_pnl) if losses_pnl else 0
        
        profit_factor = sum(wins_pnl) / sum(losses_pnl) if losses_pnl and sum(losses_pnl) > 0 else float('inf')
        
        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl_pips': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_dd,
            'final_balance': balance,
            'return_pct': (balance - initial_balance) / initial_balance * 100,
            'trades': trades,
            'equity_curve': equity_curve,
        }
        
        logger.info(f"Backtest complete!")
        logger.info(f"Trades: {total_trades}, Win Rate: {win_rate:.1f}%")
        logger.info(f"Total PnL: {total_pnl:+.1f} pips")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        
        return results

    # =========================================================================
    # SAVE/LOAD
    # =========================================================================
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        save_path = Path(path) if path else self.model_path
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'direction_model': getattr(self, 'direction_model', None),
            'params': {
                'lookback_bars': self.lookback_bars,
                'prediction_horizon': self.prediction_horizon,
                'sl_atr_mult': self.sl_atr_mult,
                'tp_atr_mult': self.tp_atr_mult,
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: Optional[str] = None):
        """Load a trained model"""
        load_path = Path(path) if path else self.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.direction_model = data.get('direction_model')
        
        params = data.get('params', {})
        self.lookback_bars = params.get('lookback_bars', self.lookback_bars)
        self.prediction_horizon = params.get('prediction_horizon', self.prediction_horizon)
        self.sl_atr_mult = params.get('sl_atr_mult', self.sl_atr_mult)
        self.tp_atr_mult = params.get('tp_atr_mult', self.tp_atr_mult)
        
        self.is_trained = True
        logger.info(f"Model loaded from {load_path}")


def train_and_test():
    """Train and test the ML model"""
    model = MLTradingModel(
        symbol="XAUUSD",
        lookback_bars=50,
        prediction_horizon=12,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
    )
    
    # Train on last year
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365)
    
    print("="*60)
    print("ML TRADING MODEL - TRAINING")
    print("="*60)
    print(f"Training period: {start_date.date()} to {end_date.date()}")
    print()
    
    # Train
    results = model.train(
        start_date=start_date,
        end_date=end_date,
        timeframe=mt5.TIMEFRAME_M15 if mt5 else 15,
        model_type="random_forest",
    )
    
    if not results:
        print("Training failed!")
        return
    
    print()
    print("="*60)
    print("TOP 10 MOST IMPORTANT FEATURES:")
    print("="*60)
    
    if 'feature_importance' in results:
        sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10]
        for name, importance in sorted_features:
            print(f"  {name}: {importance:.4f}")
    
    # Backtest on last 60 days (out of sample)
    print()
    print("="*60)
    print("BACKTESTING (Last 60 days - Out of Sample)")
    print("="*60)
    
    backtest_start = end_date - timedelta(days=60)
    backtest_results = model.backtest(
        start_date=backtest_start,
        end_date=end_date,
        min_confidence=0.6,
    )
    
    if backtest_results:
        print()
        print(f"Total Trades: {backtest_results['total_trades']}")
        print(f"Win Rate: {backtest_results['win_rate']:.1f}%")
        print(f"Total PnL: {backtest_results['total_pnl_pips']:+.1f} pips")
        print(f"Profit Factor: {backtest_results['profit_factor']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown_pct']:.1f}%")
        print(f"Return: {backtest_results['return_pct']:+.1f}%")
    
    # Save model
    model.save_model()
    
    return model, results, backtest_results


if __name__ == "__main__":
    train_and_test()

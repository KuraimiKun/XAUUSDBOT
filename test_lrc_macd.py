"""Test LRC-MACD Strategy with different configurations"""
from backend.app.services.lrc_macd_strategy import LRCMACDStrategy
from backend.app.services.backtesting import BacktestEngine
from datetime import datetime, timedelta

# Test with different TP coefficients and configurations
configs = [
    {'tp_coef': 3.0, 'sl_lookback': 20, 'trailing': True, 'pullback': 3},
    {'tp_coef': 3.5, 'sl_lookback': 20, 'trailing': True, 'pullback': 3},
    {'tp_coef': 4.0, 'sl_lookback': 25, 'trailing': True, 'pullback': 3},
    {'tp_coef': 5.0, 'sl_lookback': 30, 'trailing': True, 'pullback': 2},
]

end = datetime.now()
start = end - timedelta(days=30)

print('Testing LRC-MACD with optimized configurations...')
print('='*60)

best_pnl = -float('inf')
best_config = None
best_result = None

for cfg in configs:
    strategy = LRCMACDStrategy(
        symbol='XAUUSD',
        lrc_period=11,
        lrc_signal_period=5,
        macd_fast=34,
        macd_slow=144,
        tp_coefficient=cfg['tp_coef'],
        sl_lookback=cfg['sl_lookback'],
        sl_deviation=0.6,
        pullback_lookback=cfg['pullback'],
        use_trailing=cfg['trailing'],
        trailing_stop_level=0.5,
    )
    
    result = strategy.run_backtest(start, end)
    
    print(f"TP Coef={cfg['tp_coef']}, SL LB={cfg['sl_lookback']}, PB={cfg['pullback']}: ")
    print(f"  Trades={result.total_trades}, WR={result.win_rate:.1f}%, PnL={result.total_pnl_pips:+.0f} pips, PF={result.profit_factor:.2f}")
    
    if result.total_pnl_pips > best_pnl:
        best_pnl = result.total_pnl_pips
        best_config = cfg
        best_result = result
    print()

print('='*60)
print('BEST LRC-MACD CONFIG:')
print(f"  TP Coef={best_config['tp_coef']}, SL LB={best_config['sl_lookback']}, PB={best_config['pullback']}")
print(f"  Trades={best_result.total_trades}, WR={best_result.win_rate:.1f}%, PnL={best_result.total_pnl_pips:+.0f} pips")
print()

# Compare with EMA Pullback strategy
print('='*60)
print('COMPARISON WITH EMA PULLBACK STRATEGY:')
print('='*60)

ema_strategy = BacktestEngine(
    symbol='XAUUSD',
    ema_fast=14,
    ema_slow=24,
    ema_filter=100,
    sl_mult_long=1.2,
    sl_mult_short=1.2,
    tp_mult_long=2.4,
    tp_mult_short=2.4,
    use_h1_trend_filter=True,
)

ema_result = ema_strategy.run_backtest(start, end)

print(f"EMA Pullback: Trades={ema_result.total_trades}, WR={ema_result.win_rate:.1f}%, PnL={ema_result.total_pnl_pips:+.0f} pips, PF={ema_result.profit_factor:.2f}")
print(f"LRC-MACD:     Trades={best_result.total_trades}, WR={best_result.win_rate:.1f}%, PnL={best_result.total_pnl_pips:+.0f} pips, PF={best_result.profit_factor:.2f}")
print()

if best_result.total_pnl_pips > ema_result.total_pnl_pips:
    print(">>> LRC-MACD performs better!")
else:
    print(">>> EMA Pullback performs better!")

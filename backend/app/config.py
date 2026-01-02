from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find .env file - check current dir, parent dirs
def find_env_file() -> str:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent.parent.parent / ".env",  # sistemcerdas/.env
        Path.cwd().parent / ".env",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=find_env_file(), env_file_encoding="utf-8", extra="allow")

    app_name: str = "XAUUSD Scalping Signal System"
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/xauusd",
        alias="DATABASE_URL",
    )
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    fetch_interval_seconds: float = Field(default=0.5, alias="FETCH_INTERVAL_SECONDS")

    # MTF Scalping engine settings - tuned for XAUUSD
    # EMA periods
    ema_fast: int = Field(default=9, alias="EMA_FAST")
    ema_slow: int = Field(default=21, alias="EMA_SLOW")
    ema_trend: int = Field(default=50, alias="EMA_TREND")
    
    # RSI settings
    rsi_period: int = Field(default=14, alias="RSI_PERIOD")
    rsi_overbought: float = Field(default=70.0, alias="RSI_OVERBOUGHT")
    rsi_oversold: float = Field(default=30.0, alias="RSI_OVERSOLD")
    
    # ATR for volatility
    atr_period: int = Field(default=14, alias="ATR_PERIOD")
    
    # Target sizing (in POINTS - 1 point = 10 pips for XAUUSD)
    min_tp_points: float = Field(default=3.0, alias="MIN_TP_POINTS")  # 30 pips min target
    max_sl_points: float = Field(default=2.0, alias="MAX_SL_POINTS")  # 20 pips max risk
    min_rr_ratio: float = Field(default=1.5, alias="MIN_RR_RATIO")  # Minimum risk:reward ratio
    
    # Trade management
    max_spread: float = Field(default=0.50, alias="MAX_SPREAD")
    min_signal_confidence: float = Field(default=0.70, alias="MIN_SIGNAL_CONFIDENCE")
    cooldown_minutes: int = Field(default=15, alias="COOLDOWN_MINUTES")  # Minutes between trades
    
    # Legacy (kept for compatibility)
    atr_tp_multiplier: float = Field(default=2.5, alias="ATR_TP_MULTIPLIER")
    atr_sl_multiplier: float = Field(default=1.5, alias="ATR_SL_MULTIPLIER")
    cooldown_ticks: int = Field(default=50, alias="COOLDOWN_TICKS")
    
    # Candle analysis settings
    candle_lookback: int = Field(default=100, alias="CANDLE_LOOKBACK")
    candle_update_seconds: int = Field(default=10, alias="CANDLE_UPDATE_SECONDS")

    mt5_login: int | None = Field(default=None, alias="MT5_LOGIN")
    mt5_password: str | None = Field(default=None, alias="MT5_PASSWORD")
    mt5_server: str | None = Field(default=None, alias="MT5_SERVER")
    mt5_path: str | None = Field(default=None, alias="MT5_PATH")

    backend_base_url: str = Field(default="http://localhost:8000", alias="BACKEND_BASE_URL")

    @property
    def alerts_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

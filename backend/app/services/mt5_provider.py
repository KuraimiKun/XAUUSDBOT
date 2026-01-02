"""
Simplified MT5 Data Provider for Scalping
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from ..config import settings
from .scalping_engine import PriceTick


class MT5Provider:
    def __init__(
        self,
        symbol: str = "XAUUSD",
        login: Optional[int] = settings.mt5_login,
        password: Optional[str] = settings.mt5_password,
        server: Optional[str] = settings.mt5_server,
        path: Optional[str] = settings.mt5_path,
    ) -> None:
        self.symbol = symbol
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self._lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not available")

        async with self._lock:
            if self._connected:
                return

            def _init():
                initialized = mt5.initialize(path=self.path)
                if not initialized:
                    raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
                if self.login and self.password:
                    authorized = mt5.login(self.login, password=self.password, server=self.server)
                    if not authorized:
                        raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
                info = mt5.symbol_info(self.symbol)
                if info is None:
                    raise RuntimeError(f"Symbol {self.symbol} not found")
                if not info.visible:
                    mt5.symbol_select(self.symbol, True)

            await asyncio.to_thread(_init)
            self._connected = True
            logger.info(f"MT5 connected for {self.symbol}")

    async def disconnect(self) -> None:
        if mt5 is None:
            return
        async with self._lock:
            if self._connected:
                await asyncio.to_thread(mt5.shutdown)
                self._connected = False
                logger.info("MT5 disconnected")

    async def fetch_tick(self) -> PriceTick:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 not installed")

        def _fetch():
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                raise RuntimeError("Failed to get tick")
            return PriceTick(
                timestamp=datetime.fromtimestamp(tick.time, tz=timezone.utc),
                bid=tick.bid,
                ask=tick.ask,
                volume=tick.volume,
            )

        return await asyncio.to_thread(_fetch)

    @property
    def is_connected(self) -> bool:
        return self._connected

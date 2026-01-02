"""
Telegram Alert Service for Scalping Signals
"""
from __future__ import annotations

import asyncio
from loguru import logger

try:
    from telegram import Bot
    from telegram.error import TelegramError
except ImportError:
    Bot = None
    TelegramError = Exception

from ..config import settings


class TelegramAlerts:
    def __init__(self) -> None:
        self.bot_token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self._bot = None
        if Bot and self.bot_token:
            self._bot = Bot(token=self.bot_token)

    @property
    def enabled(self) -> bool:
        return bool(self._bot and self.chat_id)

    async def send_message(self, message: str) -> bool:
        """Send a formatted message to Telegram."""
        if not self.enabled:
            return False
        
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="Markdown",
            )
            logger.info("Telegram message sent")
            return True
        except TelegramError as e:
            logger.error(f"Telegram send failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    async def send_signal(self, signal) -> bool:
        if not self.enabled:
            return False

        # Format message based on signal type
        if signal.signal_type == SignalType.BUY:
            emoji = "🟢"
            action = "BUY SIGNAL"
            msg = (
                f"{emoji} *{action}*\n\n"
                f"💰 Entry: `{signal.entry_price:.2f}`\n"
                f"🎯 Take Profit: `{signal.take_profit:.2f}`\n"
                f"🛑 Stop Loss: `{signal.stop_loss:.2f}`\n"
                f"📊 Confidence: `{signal.confidence*100:.0f}%`\n"
                f"📝 {signal.reason}"
            )
        elif signal.signal_type == SignalType.SELL:
            emoji = "🔴"
            action = "SELL SIGNAL"
            msg = (
                f"{emoji} *{action}*\n\n"
                f"💰 Entry: `{signal.entry_price:.2f}`\n"
                f"🎯 Take Profit: `{signal.take_profit:.2f}`\n"
                f"🛑 Stop Loss: `{signal.stop_loss:.2f}`\n"
                f"📊 Confidence: `{signal.confidence*100:.0f}%`\n"
                f"📝 {signal.reason}"
            )
        elif signal.signal_type in (SignalType.EXIT_BUY, SignalType.EXIT_SELL):
            emoji = "⚪"
            action = "EXIT SIGNAL"
            msg = (
                f"{emoji} *{action}*\n\n"
                f"💰 Exit Price: `{signal.entry_price:.2f}`\n"
                f"📝 {signal.reason}"
            )
        else:
            return False

        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode="Markdown",
            )
            logger.info(f"Telegram alert sent: {action}")
            return True
        except TelegramError as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def send_trade_result(self, trade: dict) -> bool:
        if not self.enabled:
            return False

        emoji = "✅" if trade["won"] else "❌"
        pnl_emoji = "📈" if trade["pnl_points"] > 0 else "📉"

        msg = (
            f"{emoji} *TRADE CLOSED*\n\n"
            f"📌 Type: `{trade['type']}`\n"
            f"💰 Entry: `{trade['entry_price']:.2f}`\n"
            f"💰 Exit: `{trade['exit_price']:.2f}`\n"
            f"{pnl_emoji} PnL: `{trade['pnl_points']:+.2f} pts` (`${trade['pnl_usd']:+.2f}`)\n"
            f"📝 {trade['reason']}"
        )

        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode="Markdown",
            )
            return True
        except TelegramError as e:
            logger.error(f"Telegram send failed: {e}")
            return False


# Singleton instance
_alerts: TelegramAlerts | None = None


def get_telegram_alerts() -> TelegramAlerts:
    global _alerts
    if _alerts is None:
        _alerts = TelegramAlerts()
    return _alerts

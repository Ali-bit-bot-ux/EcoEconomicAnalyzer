"""
Daryn — modules/telegram_bot.py
================================
Скрипт для мгновенных Push-оповещений через Telegram Bot API.
Автоматически отправляет уведомления при фиксации критических аномалий 
(падение влажности, эко-стресс, ранний сигнал засухи).
"""

import sys
import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
from loguru import logger
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class DarynTelegramBot:
    def __init__(self):
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def is_configured(self):
        """Проверяет, настроен ли бот в .env файле."""
        return bool(self.token and self.chat_id)

    def send_drought_alert(self, region: str, psi_val: float, smai_val: float, date: str):
        """
        Отправляет предупреждение об эко-стрессе.
        """
        if not self.is_configured():
            logger.warning("Telegram Bot не настроен (отсутствует токен или chat_id в .env). Оповещение пропущено.")
            return False

        message = (
            f"🚨 <b>ВНИМАНИЕ: УГРОЗА ЗАСУХИ</b>\n\n"
            f"📍 <b>Регион:</b> {region}\n"
            f"📅 <b>Дата фиксации:</b> {date}\n\n"
            f"🌱 <b>PhenoShift Index (PSI):</b> {psi_val:+.2f} (<i>Ниже нормы!</i>)\n"
            f"💧 <b>HydroRisk (SMAI):</b> {smai_val:.2f}\n\n"
            f"⚠️ <b>Прогноз:</b> Выявлен корневой вододефицит. Алгоритм прогнозирует снижение вегетации "
            f"и падение урожайности. Рекомендуется превентивный перевод логистики."
        )

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }

        try:
            # Сессия с автоматическими ретраями
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(max_retries=3)
            session.mount('https://', adapter)
            
            response = session.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.success("Push-уведомление успешно отправлено в Telegram.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Сетевая ошибка Telegram API: {e}")
            return False

if __name__ == "__main__":
    # Тестовый запуск
    bot = DarynTelegramBot()
    bot.send_drought_alert("Северо-Казахстанская обл.", -1.8, -1.2, datetime.date.today().strftime("%Y-%m-%d"))

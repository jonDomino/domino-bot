"""
Telegram API helpers using raw HTTP requests.
"""

import requests
import time
from typing import Optional, Dict, Any

from utils import config


def send_telegram_message(msg: str, chat_id: Optional[str] = None):
    """
    Send a Telegram message.
    If chat_id is provided, send to that chat. Otherwise use config.TELEGRAM_CHAT_ID.
    """
    if not config.TELEGRAM_BOT_TOKEN:
        print("❌ Telegram not configured (missing bot token)")
        return
    
    target_chat_id = chat_id or config.TELEGRAM_CHAT_ID
    if not target_chat_id:
        print("❌ Telegram not configured (missing chat ID)")
        return
    
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": target_chat_id, "text": msg}
    
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")


def poll_updates(bot_token: str, offset: Optional[int] = None) -> Dict[str, Any]:
    """
    Poll Telegram for updates using getUpdates.
    Returns the JSON response.
    """
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    params = {"timeout": 30}
    
    if offset is not None:
        params["offset"] = offset
    
    try:
        resp = requests.get(url, params=params, timeout=35)
        print("Telegram raw response:", resp.text)
        return resp.json()
    except Exception as e:
        print(f"❌ Failed to poll Telegram updates: {e}")
        return {"ok": False, "result": []}


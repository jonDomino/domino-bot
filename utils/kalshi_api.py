"""
Kalshi API helpers with RSA-PSS + SHA256 signing.
"""

import time
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from typing import Dict, Any, Optional

from utils import config


def load_creds():
    """
    Load Kalshi API credentials from files.
    Returns (api_key_id, private_key_pem).
    """
    try:
        with open(config.API_KEY_ID_FILE, "r") as f:
            api_key_id = f.read().strip()
        
        with open(config.PRIVATE_KEY_FILE, "r") as f:
            private_key_pem = f.read()
        
        return api_key_id, private_key_pem
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing Kalshi credentials file: {e.filename}")


def sign_request(private_key_pem: str, message: str) -> str:
    """
    Sign a message using RSA-PSS + SHA256.
    Returns base64-encoded signature.
    """
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem,
        password=None,
        backend=default_backend()
    )
    
    signature = private_key.sign(
        message.encode('utf-8'),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    
    import base64
    return base64.b64encode(signature).decode('utf-8')


def make_request(
    api_key_id: str,
    private_key_pem: str,
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Make an authenticated Kalshi API request.
    """
    timestamp = str(int(time.time() * 1000))
    
    # Ensure path starts with /trade-api/v2
    sign_path = path if path.startswith("/trade-api/v2") else "/trade-api/v2" + path
    message = timestamp + method.upper() + sign_path
    
    signature = sign_request(private_key_pem, message)
    
    headers = {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json",
    }
    
    url = config.KALSHI_BASE_URL + path
    
    try:
        if method.upper() == "GET":
            resp = requests.get(url, headers=headers, params=body, timeout=20)
        elif method.upper() == "POST":
            resp = requests.post(url, headers=headers, json=body, timeout=20)
        elif method.upper() == "DELETE":
            resp = requests.delete(url, headers=headers, timeout=20)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        resp.raise_for_status()
        
        # Handle 204 No Content (some DELETE endpoints return this)
        if resp.status_code == 204:
            return {}
        
        # Try to parse JSON, return empty dict if not JSON
        try:
            return resp.json()
        except ValueError:
            return {}
    except requests.exceptions.HTTPError as e:
        print(f"❌ Kalshi API error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        raise
    except Exception as e:
        print(f"❌ Kalshi API request failed: {e}")
        raise


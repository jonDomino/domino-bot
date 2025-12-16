"""
Temporary debug script to inspect raw Kalshi orderbook JSON structure.

This script fetches and prints the complete orderbook response for
moneyline markets to understand the actual field structure.

Will be deleted after diagnostic use.
"""

import json
from utils.kalshi_api import load_creds, make_request

# Hardcoded market tickers to inspect
MARKET_TICKERS = [
    "KXNCAAMBGAME-25DEC15NIAGVCU-VCU",
    "KXNCAAMBGAME-25DEC15NIAGVCU-NIAG"
]


def fetch_orderbook(api_key_id: str, private_key_pem: str, market_ticker: str) -> dict:
    """
    Fetch raw orderbook JSON for a market ticker.
    Uses the same endpoint and method as main.py.
    """
    path = f"/markets/{market_ticker}/orderbook"
    
    try:
        resp = make_request(api_key_id, private_key_pem, "GET", path)
        return resp
    except Exception as e:
        print(f"❌ Error fetching orderbook for {market_ticker}: {e}")
        return {}


def inspect_field(field_name: str, data: any) -> None:
    """
    Inspect and print details about a specific field.
    """
    print(f"  {field_name}:")
    
    if field_name not in data:
        print(f"    Status: FIELD NOT PRESENT")
        return
    
    value = data[field_name]
    value_type = type(value).__name__
    
    print(f"    Status: PRESENT")
    print(f"    Type: {value_type}")
    
    if value is None:
        print(f"    Value: None")
    elif isinstance(value, list):
        print(f"    Length: {len(value)}")
        if len(value) > 0:
            print(f"    First 3 entries: {value[:3]}")
        else:
            print(f"    First 3 entries: (empty list)")
    elif isinstance(value, (int, float)):
        print(f"    Value: {value}")
    elif isinstance(value, str):
        print(f"    Value: {value}")
    else:
        print(f"    Value: {value}")


def print_orderbook_analysis(market_ticker: str, orderbook_data: dict) -> None:
    """
    Print complete analysis of orderbook structure.
    """
    print("=" * 80)
    print(f"MARKET TICKER: {market_ticker}")
    print("=" * 80)
    print()
    
    # Extract orderbook dict (response may be wrapped)
    if "orderbook" in orderbook_data:
        orderbook = orderbook_data["orderbook"]
        print("📦 Full Response Structure:")
        print(json.dumps(orderbook_data, indent=2))
        print()
        print("📊 Orderbook Object:")
        print(json.dumps(orderbook, indent=2))
    else:
        orderbook = orderbook_data
        print("📦 Full Response (no 'orderbook' wrapper):")
        print(json.dumps(orderbook_data, indent=2))
        print()
        print("📊 Orderbook Object (same as above):")
        print(json.dumps(orderbook, indent=2))
    
    print()
    print("-" * 80)
    print("FIELD INSPECTION")
    print("-" * 80)
    print()
    
    # Inspect all key fields
    print("YES Side Fields:")
    inspect_field("yes_ask", orderbook)
    inspect_field("yes_bid", orderbook)
    inspect_field("yes_asks", orderbook)
    inspect_field("yes_bids", orderbook)
    print()
    
    print("NO Side Fields:")
    inspect_field("no_ask", orderbook)
    inspect_field("no_bid", orderbook)
    inspect_field("no_asks", orderbook)
    inspect_field("no_bids", orderbook)
    print()
    
    # Also check for any other unexpected fields
    all_keys = set(orderbook.keys())
    expected_keys = {"yes_ask", "yes_bid", "yes_asks", "yes_bids", "no_ask", "no_bid", "no_asks", "no_bids", "yes", "no"}
    unexpected_keys = all_keys - expected_keys
    
    if unexpected_keys:
        print("⚠️  Additional Fields Found:")
        for key in sorted(unexpected_keys):
            value = orderbook[key]
            value_type = type(value).__name__
            print(f"  {key}: {value_type} = {value}")
        print()
    
    print("=" * 80)
    print()


def main():
    """
    Main execution: fetch and analyze orderbooks for both markets.
    """
    print("🔍 Kalshi Orderbook Structure Inspector")
    print("=" * 80)
    print()
    print("This script fetches raw orderbook JSON to understand field structure.")
    print("No inference, no transformation, raw data only.")
    print()
    
    # Load Kalshi credentials
    try:
        api_key_id, private_key_pem = load_creds()
        print("✓ Kalshi credentials loaded")
        print()
    except Exception as e:
        print(f"❌ Failed to load Kalshi credentials: {e}")
        return
    
    # Fetch and analyze each market
    for market_ticker in MARKET_TICKERS:
        print(f"Fetching orderbook for: {market_ticker}")
        orderbook_data = fetch_orderbook(api_key_id, private_key_pem, market_ticker)
        
        if not orderbook_data:
            print(f"⚠️  No data returned for {market_ticker}")
            print()
            continue
        
        print_orderbook_analysis(market_ticker, orderbook_data)
    
    print("✅ Inspection complete")
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Review the output above to determine:")
    print("  • Which fields Kalshi actually provides")
    print("  • Whether YES asks exist independently of NO bids")
    print("  • Whether array-based fields (*_asks, *_bids) are populated")
    print("  • Whether maker logic should infer asks or use direct fields")
    print()


if __name__ == "__main__":
    main()


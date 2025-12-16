"""
Temporary debug script to inspect Kalshi totals market structure.

This script fetches all markets for a known CBB event and prints
detailed information about each market to understand how totals
markets are represented.

This script will be deleted after use.
"""

from utils.kalshi_api import load_creds, make_request
from utils import config


def fetch_all_markets_for_event(event_ticker: str):
    """
    Fetch all markets for a Kalshi event.
    
    Args:
        event_ticker: Event ticker (e.g., "KXNCAAMBGAME-25DEC15NIAGVCU")
    
    Returns:
        List of market dicts
    """
    api_key_id, private_key_pem = load_creds()
    
    path = "/markets"
    params = {
        "event_ticker": event_ticker,
        "status": "open"
    }
    
    try:
        resp = make_request(api_key_id, private_key_pem, "GET", path, params)
        return resp.get("markets", [])
    except Exception as e:
        print(f"❌ Failed to fetch markets: {e}")
        return []


def contains_totals_keywords(market: dict) -> bool:
    """
    Check if market contains keywords suggesting it's a totals market.
    
    Returns True if "total", "over", or "under" appears in any text field.
    """
    text_fields = [
        market.get("ticker", ""),
        market.get("title", ""),
        market.get("subtitle", ""),
        market.get("rules_primary", ""),
        market.get("rules_secondary", ""),
    ]
    
    combined_text = " ".join(str(f) for f in text_fields).upper()
    keywords = ["TOTAL", "OVER", "UNDER"]
    
    return any(keyword in combined_text for keyword in keywords)


def extract_numeric_values(text: str) -> list:
    """
    Extract all numeric values from text (for debugging line detection).
    Returns list of numeric strings found.
    """
    import re
    # Find all numbers (integers and decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    return numbers


def print_market_details(market: dict, index: int):
    """
    Print detailed information about a single market.
    """
    print("=" * 80)
    print(f"MARKET #{index + 1}")
    print("=" * 80)
    
    # Basic fields
    print(f"TICKER: {market.get('ticker', 'N/A')}")
    print(f"market_type: {market.get('market_type', 'N/A')}")
    print(f"title: {market.get('title', 'N/A')}")
    print(f"subtitle: {market.get('subtitle', 'N/A')}")
    
    # Rules
    rules_primary = market.get("rules_primary", "")
    rules_secondary = market.get("rules_secondary", "")
    
    if rules_primary:
        print(f"rules_primary: {rules_primary}")
    if rules_secondary:
        print(f"rules_secondary: {rules_secondary}")
    
    # Outcome labels
    outcome_labels = market.get("outcome_labels", [])
    if outcome_labels:
        print(f"outcome_labels: {outcome_labels}")
    
    # Numeric values in text fields
    all_text = " ".join([
        market.get("ticker", ""),
        market.get("title", ""),
        market.get("subtitle", ""),
        market.get("rules_primary", ""),
        market.get("rules_secondary", ""),
    ])
    numeric_values = extract_numeric_values(all_text)
    if numeric_values:
        print(f"Numeric values found in text: {numeric_values}")
    
    # Additional fields that might be relevant
    if "line" in market:
        print(f"line: {market.get('line')}")
    if "strike" in market:
        print(f"strike: {market.get('strike')}")
    if "strike_price" in market:
        print(f"strike_price: {market.get('strike_price')}")
    
    # Check for totals keywords
    is_possible_totals = contains_totals_keywords(market)
    if is_possible_totals:
        print("⚠️  POSSIBLE_TOTALS_MARKET = True")
    else:
        print("POSSIBLE_TOTALS_MARKET = False")
    
    # Print all keys for reference
    all_keys = list(market.keys())
    print(f"\nAll available fields: {', '.join(sorted(all_keys))}")
    
    print()


def main():
    """
    Main debug function - fetches and prints market structure.
    """
    # Hardcoded known event ticker
    EVENT_TICKER = "KXNCAAMBGAME-25DEC15NIAGVCU"
    
    print("=" * 80)
    print("KALSHI TOTALS MARKET STRUCTURE DEBUG")
    print("=" * 80)
    print()
    print(f"Fetching all markets for event: {EVENT_TICKER}")
    print()
    
    markets = fetch_all_markets_for_event(EVENT_TICKER)
    
    if not markets:
        print("❌ No markets found or API error occurred")
        return
    
    print(f"✓ Found {len(markets)} total markets")
    print()
    
    # Print details for each market
    for i, market in enumerate(markets):
        print_market_details(market, i)
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    totals_count = sum(1 for m in markets if contains_totals_keywords(m))
    print(f"Total markets: {len(markets)}")
    print(f"Markets with totals keywords: {totals_count}")
    print()
    print("Inspect the output above to determine:")
    print("1. Which fields contain the total line value")
    print("2. How the line is formatted")
    print("3. What differentiates totals from moneyline markets")
    print()


if __name__ == "__main__":
    main()


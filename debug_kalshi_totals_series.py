"""
Temporary debug script to discover Kalshi totals series structure.

This script fetches all CBB events and searches for totals-related events
to identify the totals series ticker and event structure.

This script will be deleted after use.
"""

from utils.kalshi_api import load_creds, make_request
from utils import config
import re


def fetch_events_for_series(series_ticker: str):
    """
    Fetch all open events for a given series, WITH nested markets.
    
    Args:
        series_ticker: Series ticker to fetch events for
    
    Returns:
        List of event dicts (with nested markets)
    """
    api_key_id, private_key_pem = load_creds()
    
    path = "/events"
    params = {
        "series_ticker": series_ticker,
        "status": "open",
        "with_nested_markets": "true"  # We need markets to inspect
    }
    
    try:
        resp = make_request(api_key_id, private_key_pem, "GET", path, params)
        return resp.get("events", [])
    except Exception as e:
        print(f"  ⚠️  Failed to fetch events for {series_ticker}: {e}")
        return []


def search_totals_series():
    """
    Search for totals-related series by fetching all series.
    """
    api_key_id, private_key_pem = load_creds()
    
    path = "/series"
    params = {
        "status": "open"
    }
    
    try:
        resp = make_request(api_key_id, private_key_pem, "GET", path, params)
        return resp.get("series", [])
    except Exception as e:
        print(f"❌ Failed to fetch series: {e}")
        return []


def contains_totals_keywords(text: str) -> bool:
    """
    Check if text contains totals-related keywords.
    Any one keyword is sufficient to flag as potential totals.
    """
    if not text:
        return False
    text_upper = str(text).upper()
    keywords = [
        "OVER",
        "UNDER",
        "POINTS SCORED",
        "POINTS",
        "TOTAL",
        "O/U"
    ]
    return any(keyword in text_upper for keyword in keywords)


def extract_numeric_values(text: str) -> list:
    """
    Extract numeric values from text, prioritizing decimal totals lines.
    Focuses on values between 100.0 and 200.0 (typical CBB totals range).
    """
    if not text:
        return []
    
    numbers = []
    
    # First, find decimal numbers (like 141.5) - these are most likely totals lines
    decimals = re.findall(r'\d+\.\d+', text)
    for dec in decimals:
        try:
            val = float(dec)
            if 100.0 <= val <= 200.0:
                numbers.append(dec)
        except ValueError:
            pass
    
    # Then find integers in reasonable range (100-200 for totals)
    integers = re.findall(r'\b(\d{3})\b', text)
    for i in integers:
        try:
            num = int(i)
            if 100 <= num <= 200:
                # Prefer decimals, so only add integer if no decimals found nearby
                if not decimals:
                    numbers.append(i)
        except ValueError:
            pass
    
    return numbers


def detect_over_under(text: str) -> dict:
    """
    Detect if "OVER" or "UNDER" appears in text.
    Returns dict with flags.
    """
    if not text:
        return {"has_over": False, "has_under": False}
    
    text_upper = str(text).upper()
    return {
        "has_over": "OVER" in text_upper,
        "has_under": "UNDER" in text_upper
    }


def print_totals_market(market: dict, event: dict, series_ticker: str):
    """
    Print a confirmed totals market in a clear, structured format.
    """
    print("=" * 80)
    print("🎯 CONFIRMED TOTALS MARKET FOUND")
    print("=" * 80)
    print()
    
    event_ticker = event.get("event_ticker", "N/A")
    market_ticker = market.get("ticker", "N/A")
    market_title = market.get("title", "N/A")
    custom_strike = market.get("custom_strike")
    
    print(f"series_ticker: {series_ticker}")
    print(f"event_ticker: {event_ticker}")
    print(f"market_ticker: {market_ticker}")
    print(f"market title: {market_title}")
    
    if custom_strike is not None:
        print(f"custom_strike: {custom_strike}")
    else:
        print(f"custom_strike: (not present)")
    
    print()
    print("=" * 80)


def main():
    """
    Main debug function - searches across ALL series for totals events.
    """
    print("=" * 80)
    print("KALSHI TOTALS SERIES DISCOVERY (ALL SERIES)")
    print("=" * 80)
    print()
    
    # Step 1: Fetch all open series
    print("Step 1: Fetching all open series...")
    print()
    all_series = search_totals_series()
    print(f"✓ Found {len(all_series)} total open series")
    print()
    
    # Step 2: For each series, fetch events with markets and check markets for totals
    print("=" * 80)
    print("Step 2: Searching markets across all series for totals...")
    print()
    
    series_checked = 0
    
    for series in all_series:
        series_ticker = series.get("ticker", "")  # Fixed: use "ticker" not "series_ticker"
        if not series_ticker:
            continue
        
        # Restrict search to CBB-related series only (much faster)
        if "NCAAMB" not in series_ticker.upper():
            continue
        
        series_checked += 1
        if series_checked % 10 == 0:
            print(f"  Progress: Checked {series_checked} CBB-related series...")
        
        # Fetch events for this series (with nested markets)
        events = fetch_events_for_series(series_ticker)
        
        if not events:
            continue
        
        # Check each event's markets for totals keywords
        for event in events:
            markets = event.get("markets", [])
            
            if not markets:
                continue
            
            # Check each market for totals keywords
            for market in markets:
                market_ticker = market.get("ticker", "")
                market_title = market.get("title", "")
                market_subtitle = market.get("subtitle", "")
                
                # Skip spread markets
                market_text_upper = f"{market_ticker} {market_title} {market_subtitle}".upper()
                if "SPREAD" in market_text_upper:
                    continue
                
                # Check market-level fields for totals keywords
                market_text = f"{market_ticker} {market_title} {market_subtitle}"
                
                if contains_totals_keywords(market_text):
                    # Found a confirmed totals market - print and stop
                    print_totals_market(market, event, series_ticker)
                    print()
                    print("✓ Stopping after first confirmed totals market discovery")
                    return
    
    print(f"\n✓ Checked {series_checked} series")
    print("⚠️  No totals markets found across all series")
    print()


if __name__ == "__main__":
    main()


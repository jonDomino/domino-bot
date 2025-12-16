"""
One-time helper script to extract unique CBB team names from Unabated
and team codes from Kalshi for manual team_xref_cbb.csv creation.

This script:
1. Fetches all unique CBB team names from Unabated (lg4: sections only)
2. Fetches all unique CBB team code suffixes from Kalshi (KXNCAAMBGAME)
3. Prints both lists for manual CSV creation
"""

import requests
import re
from utils import config
from utils.kalshi_api import load_creds, make_request

MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}


def fetch_unabated_cbb_teams():
    """Fetch all unique CBB team names from Unabated snapshot (lg4: sections only)."""
    if not config.UNABATED_API_KEY:
        raise ValueError("Unabated API key not configured")
    
    url = "https://partner-api.unabated.com/api/markets/gameOdds"
    resp = requests.get(url, headers={"x-api-key": config.UNABATED_API_KEY}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    
    # Build team dictionary (authoritative)
    teams = {
        str(k): v.get("name")
        for k, v in (data.get("teams") or {}).items()
        if isinstance(v, dict) and v.get("name")
    }
    
    team_names = set()
    
    # Only CBB sections (leagueId = 4, prefix: lg4:)
    for section_key, games in (data.get("gameOddsEvents") or {}).items():
        # Only process CBB sections
        if not section_key.startswith("lg4:"):
            continue
        
        if not isinstance(games, list):
            continue
        
        for game in games:
            event_teams = game.get("eventTeams", {})
            if not isinstance(event_teams, dict):
                continue
            
            for slot in ("0", "1"):
                team_info = event_teams.get(slot)
                if not isinstance(team_info, dict):
                    continue
                
                team_id = str(team_info.get("id"))
                team_name = teams.get(team_id)
                if team_name:
                    team_names.add(team_name)
    
    return sorted(team_names)


def parse_kalshi_event_ticker(event_ticker: str) -> tuple:
    """
    Parse Kalshi event ticker to extract team code suffix.
    Example: "KXNCAAMBGAME-25DEC14CHSLCHI" -> "CHSLCHI"
    """
    try:
        if "-" not in event_ticker:
            return None
        
        token = event_ticker.split("-")[1]  # e.g., "25DEC14CHSLCHI"
        
        if len(token) < 7:
            return None
        
        # Extract date part (first 7 chars: YYMMMDD)
        date_part = token[:7]
        # Rest is team codes
        team_suffix = token[7:]
        
        return team_suffix
    except Exception:
        return None


def fetch_kalshi_cbb_team_codes():
    """Fetch all unique CBB team code suffixes from Kalshi events."""
    api_key_id, private_key_pem = load_creds()
    
    path = "/events"
    params = {
        "series_ticker": "KXNCAAMBGAME",
        "status": "open",
        "with_nested_markets": "true"
    }
    
    resp = make_request(api_key_id, private_key_pem, "GET", path, params)
    events = resp.get("events", [])
    
    team_code_suffixes = set()
    
    for event in events:
        event_ticker = event.get("event_ticker", "")
        suffix = parse_kalshi_event_ticker(event_ticker)
        if suffix:
            team_code_suffixes.add(suffix)
    
    return sorted(team_code_suffixes)


def main():
    print("=" * 60)
    print("CBB Team Xref Helper")
    print("=" * 60)
    print()
    
    print("Fetching Unabated CBB teams (lg4: sections only)...")
    try:
        unabated_teams = fetch_unabated_cbb_teams()
        print(f"✓ Found {len(unabated_teams)} unique CBB team names from Unabated")
        print()
        print("Unabated CBB Team Names (sorted):")
        print("-" * 60)
        for team in unabated_teams:
            print(team)
        print()
    except Exception as e:
        print(f"❌ Error fetching Unabated teams: {e}")
        return
    
    print("=" * 60)
    print()
    
    print("Fetching Kalshi CBB team code suffixes (KXNCAAMBGAME)...")
    try:
        kalshi_suffixes = fetch_kalshi_cbb_team_codes()
        print(f"✓ Found {len(kalshi_suffixes)} unique team code suffixes from Kalshi")
        print()
        print("Kalshi Team Code Suffixes (sorted):")
        print("-" * 60)
        for suffix in kalshi_suffixes:
            print(suffix)
        print()
    except Exception as e:
        print(f"❌ Error fetching Kalshi team codes: {e}")
        return
    
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Manually create team_xref_cbb.csv with format:")
    print("   league,unabated_name,kalshi_code")
    print("2. Map Unabated team names to Kalshi codes")
    print("3. Note: Kalshi suffixes may contain multiple team codes (e.g., CHSLCHI)")
    print("   You'll need to split them using the xref table during matching")


if __name__ == "__main__":
    main()


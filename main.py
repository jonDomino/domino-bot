"""
Kalshi Telegram Turn-In Executor

Main execution script that polls Telegram for turn-in messages,
resolves games via Unabated, maps to Kalshi markets, and executes trades.
"""

import re
import csv
import math
import uuid
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import requests
try:
    from zoneinfo import ZoneInfo
    USE_PYTZ = False
except ImportError:
    # Python < 3.9 fallback - use pytz
    import pytz
    USE_PYTZ = True

from utils import config
from utils.telegram_api import send_telegram_message, poll_updates
from utils.kalshi_api import load_creds, make_request

# League-scoped constants (imported from config)
LEAGUE = config.LEAGUE
UNABATED_LEAGUE_PREFIX = f"lg{config.UNABATED_LEAGUE_ID}:"
KALSHI_SERIES_TICKER = config.KALSHI_SERIES_TICKER
TEAM_XREF_FILE = config.TEAM_XREF_FILE


# ============================================================================
# Constants
# ============================================================================

MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

# ============================================================================
# Caching (Performance Optimization)
# ============================================================================

# Unabated cache
UNABATED_CACHE = {
    "snapshot": None,
    "teams": {},
    "roto_to_game": {},
    "fetched_at": 0
}

UNABATED_TTL = 60  # seconds

# Kalshi events cache (GAME series)
KALSHI_EVENTS_CACHE = {
    "by_key": {},
    "fetched_at": 0
}

# Kalshi totals events cache (TOTAL series) - parallel to GAME cache
KALSHI_TOTALS_EVENTS_CACHE = {
    "by_key": {},
    "fetched_at": 0
}

KALSHI_EVENTS_TTL = 120  # seconds


# ============================================================================
# Kill Command Parsing
# ============================================================================

def parse_totals_command(msg: str) -> Optional[Dict[str, Any]]:
    """
    Parse a totals read-only command.
    
    Format: totals <rotation> <line>
    Example: "totals 891 141.5"
    
    Returns dict with:
    - command_type: "totals_view"
    - rotation_number: int
    - line: float
    Returns None if not a totals command.
    """
    msg = msg.strip()
    msg_lower = msg.lower()
    
    # Check for "totals" command
    if not msg_lower.startswith("totals "):
        return None
    
    remaining = msg[7:].strip()  # Remove "totals " prefix
    
    # Parse: <rotation> <line>
    totals_pattern = r'^(\d+)\s+(\d+\.?\d*)$'
    totals_match = re.match(totals_pattern, remaining)
    
    if totals_match:
        rotation_number = int(totals_match.group(1))
        line = float(totals_match.group(2))
        
        return {
            "command_type": "totals_view",
            "rotation_number": rotation_number,
            "line": line
        }
    
    return None


def parse_kill_command(msg: str) -> Optional[Dict[str, Any]]:
    """
    Parse a kill command message.
    
    Supported formats:
    - "kill" → cancel all open resting orders
    - "kill {rotation_number}" → cancel all open resting orders for that roto
    
    Returns dict with:
    - command_type: "kill_all" or "kill_roto"
    - rotation_number: int (only for kill_roto)
    Returns None if not a kill command.
    """
    msg = msg.strip()
    msg_lower = msg.lower()
    
    # Check for "kill" command
    if msg_lower == "kill":
        return {
            "command_type": "kill_all"
        }
    
    # Check for "kill {roto}" format
    kill_roto_pattern = r'^kill\s+(\d+)$'
    kill_roto_match = re.match(kill_roto_pattern, msg_lower)
    
    if kill_roto_match:
        rotation_number = int(kill_roto_match.group(1))
        return {
            "command_type": "kill_roto",
            "rotation_number": rotation_number
        }
    
    return None


# ============================================================================
# Turn-In Parsing
# ============================================================================

def parse_turnin(msg: str) -> Optional[Dict[str, Any]]:
    """
    Parse a turn-in message (moneyline or totals).
    
    Taker format (unchanged):
    - Moneyline: {rotation_number} {american_odds}, {budget}
      Example: "892 -900, 0.001"
    - Totals: {rotation_number} {over|under} {line} {american_odds}, {budget}
      Example: "891 over 141.5 -110, 0.001"
    
    Maker format (explicit):
    - Moneyline: make {rotation_number} {american_odds}, {budget}
      Example: "make 892 -900, 0.001"
    - Totals: make {rotation_number} {over|under} {line} {american_odds}, {budget}
      Example: "make 891 over 141.5 -110, 0.001"
    
    Make both format (totals only, tick-based):
    - make both {rotation_number} {line}, {offset_ticks} {budget}
      Example: "make both 891 141.5, 1 0.01"
    
    Returns dict with:
    - For moneyline: rotation_number, juice, amount, market_type="moneyline", execution_mode="taker"|"maker"
    - For totals: rotation_number, side, line, juice, amount, market_type="totals", execution_mode="taker"|"maker"
    - For make both: rotation_number, line, offset_ticks, amount, market_type="totals", execution_mode="make_both"
    Returns None if parsing fails.
    """
    msg = msg.strip()
    
    # Check for "make both" command first (totals only, tick-based)
    msg_upper = msg.upper()
    if msg_upper.startswith("MAKE BOTH "):
        # Parse: make both <rotation> <line>, <offset_ticks> <budget>
        # Example: "make both 891 141.5, 1 0.01"
        remaining = msg[10:].strip()  # Remove "make both " prefix
        make_both_pattern = r'^(\d+)\s+(\d+\.?\d*),\s*(\d+)\s+(\d+\.?\d*)$'
        make_both_match = re.match(make_both_pattern, remaining)
        
        if make_both_match:
            rotation_number = int(make_both_match.group(1))
            line = float(make_both_match.group(2))
            offset_ticks = int(make_both_match.group(3))
            amount = float(make_both_match.group(4))
            
            return {
                "rotation_number": rotation_number,
                "market_type": "totals",
                "line": line,
                "offset_ticks": offset_ticks,
                "amount": amount,
                "execution_mode": "make_both"
            }
        else:
            # Invalid make both format
            return None
    
    # Check if first token is "make" (case-insensitive)
    # This determines execution mode
    execution_mode = "taker"  # Default
    
    if msg_upper.startswith("MAKE "):
        execution_mode = "maker"
        msg = msg[5:].strip()  # Remove "make " prefix
        print(f"DEBUG: execution_mode=maker (detected via 'make' prefix)")
    elif msg_upper.startswith("MAKER "):
        # Reject "maker" as first token (must use "make")
        return None
    elif msg_upper.endswith(" MAKER") or " MAKER " in msg_upper:
        # Reject trailing "maker" keyword (legacy format not allowed)
        return None
    
    # Try totals format first (more specific)
    totals_pattern = r'^(\d+)\s+(over|under)\s+(\d+\.?\d*)\s+(-?\d+),\s*(\d+\.?\d*)$'
    totals_match = re.match(totals_pattern, msg, re.IGNORECASE)
    
    if totals_match:
        rotation_number = int(totals_match.group(1))
        side = totals_match.group(2).lower()
        line = float(totals_match.group(3))
        juice = int(totals_match.group(4))
        amount = float(totals_match.group(5))
        
        return {
            "rotation_number": rotation_number,
            "market_type": "totals",
            "side": side,
            "line": line,
            "juice": juice,
            "amount": amount,
            "execution_mode": execution_mode
        }
    
    # Try moneyline format
    moneyline_pattern = r'^(\d+)\s+(-?\d+),\s*(\d+\.?\d*)$'
    moneyline_match = re.match(moneyline_pattern, msg)
    
    if moneyline_match:
        rotation_number = int(moneyline_match.group(1))
        juice = int(moneyline_match.group(2))
        amount = float(moneyline_match.group(3))
        
        return {
            "rotation_number": rotation_number,
            "market_type": "moneyline",
            "juice": juice,
            "amount": amount,
            "execution_mode": execution_mode
        }
    
    return None


# ============================================================================
# Team Name Normalization
# ============================================================================

def load_team_xref(path: str = None) -> Dict[Tuple[str, str], str]:
    """
    Load team name cross-reference CSV.
    Uses config.TEAM_XREF_FILE if path not provided.
    """
    if path is None:
        path = TEAM_XREF_FILE
    """
    Load team name cross-reference CSV.
    Returns dict mapping (league, unabated_name_lower) -> kalshi_code.
    """
    xref = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                league_key = row["league"].strip().upper()
                unabated_name = row["unabated_name"].strip().lower()
                kalshi_code = row["kalshi_code"].strip().upper()
                xref[(league_key, unabated_name)] = kalshi_code
    except FileNotFoundError:
        print(f"❌ Team xref file not found: {path}")
    return xref


def team_to_kalshi_code(league: str, team_raw: str) -> Optional[str]:
    """
    Look up Kalshi code for a team name.
    Returns None if not found.
    Uses module-level TEAM_XREF.
    """
    key = (league.upper(), team_raw.strip().lower())
    return TEAM_XREF.get(key)


# Load team xref once at module level (after function definition)
TEAM_XREF = load_team_xref()


# ============================================================================
# Unabated Integration
# ============================================================================

def fetch_unabated_snapshot() -> Dict[str, Any]:
    """
    Fetch Unabated game odds snapshot.
    """
    if not config.UNABATED_API_KEY:
        raise ValueError("Unabated API key not configured")
    
    url = f"{config.UNABATED_PROD_URL}?x-api-key={config.UNABATED_API_KEY}"
    
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise Exception(f"Failed to fetch Unabated snapshot: {e}")


def refresh_unabated_cache():
    """
    Always fetch and rebuild Unabated cache.
    Called by maybe_refresh_unabated_cache().
    """
    snapshot = fetch_unabated_snapshot()
    UNABATED_CACHE["snapshot"] = snapshot
    UNABATED_CACHE["teams"] = snapshot.get("teams", {})
    
    # Precompute roto → game mapping (CBB only: lg4: sections)
    roto_to_game = {}
    for section_key, games in snapshot.get("gameOddsEvents", {}).items():
        # Only process CBB sections (league ID 4)
        if not section_key.startswith(UNABATED_LEAGUE_PREFIX):
            continue
        
        if not isinstance(games, list):
            continue
        
        for game in games:
            event_teams = game.get("eventTeams", {})
            
            # Normalize eventTeams
            if isinstance(event_teams, dict):
                teams_iter = event_teams.values()
            elif isinstance(event_teams, list):
                teams_iter = event_teams
            else:
                continue
            
            for team in teams_iter:
                if not isinstance(team, dict):
                    continue
                
                roto = team.get("rotationNumber")
                if roto is not None:
                    roto_to_game[roto] = game
    
    UNABATED_CACHE["roto_to_game"] = roto_to_game
    UNABATED_CACHE["fetched_at"] = time.time()


def maybe_refresh_unabated_cache():
    """
    Refresh Unabated cache if it's older than TTL.
    Uses last good cache on failure.
    """
    if time.time() - UNABATED_CACHE["fetched_at"] > UNABATED_TTL:
        try:
            refresh_unabated_cache()
        except Exception:
            # If refresh fails, continue using last good cache
            if UNABATED_CACHE["snapshot"] is None:
                raise  # Nothing to fall back to
            # else: log and continue (logging can be added later)


def find_unabated_game_by_roto(roto: int) -> Optional[Dict[str, Any]]:
    """
    Find a game by rotation number using cached roto_to_game mapping.
    Returns None if not found.
    """
    return UNABATED_CACHE["roto_to_game"].get(roto)


# ============================================================================
# Date and Key Building
# ============================================================================

def unabated_event_to_kalshi_date(event_start: str) -> str:
    """
    Convert Unabated UTC eventStart to Kalshi local date (US/Eastern).
    
    Kalshi uses US Eastern local dates in tickers, while Unabated uses UTC.
    For evening games (6-9 PM ET), this can cross the UTC midnight boundary,
    causing date mismatches if we use UTC directly.
    
    Example:
        Unabated: 2025-12-16T00:00:00Z (midnight UTC = 7 PM ET on Dec 15)
        Kalshi: 25DEC15 (Dec 15 local)
        Returns: "20251215"
    """
    if USE_PYTZ:
        import pytz
        utc = pytz.UTC
        eastern = pytz.timezone("US/Eastern")
    else:
        utc = ZoneInfo("UTC")
        eastern = ZoneInfo("US/Eastern")
    
    dt_utc = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=utc)
    else:
        dt_utc = dt_utc.astimezone(utc)
    
    dt_local = dt_utc.astimezone(eastern)
    return dt_local.strftime("%Y%m%d")


def build_canonical_key(league: str, event_start: str, team_a: str, team_b: str) -> str:
    """
    Build canonical game key: {LEAGUE}_{YYYYMMDD}_{TEAM_A}_{TEAM_B}
    Teams are sorted alphabetically.
    Uses US Eastern local date (not UTC) to match Kalshi ticker dates.
    """
    date_str = unabated_event_to_kalshi_date(event_start)
    teams_sorted = sorted([team_a.upper(), team_b.upper()])
    return f"{league}_{date_str}_{teams_sorted[0]}_{teams_sorted[1]}"


# ============================================================================
# Kalshi Event Matching
# ============================================================================

def parse_kalshi_event_ticker(event_ticker: str) -> Optional[Tuple[str, str]]:
    """
    Parse Kalshi event ticker to extract date (YYYYMMDD) and team codes.
    Example: "KXNCAAMBGAME-25DEC14CHSLCHI" -> ("20251214", "CHSLCHI")
    Returns (date_str, team_codes_str) or None if parsing fails.
    """
    try:
        if "-" not in event_ticker:
            return None
        
        token = event_ticker.split("-")[1]  # e.g., "25DEC14VANNJ"
        
        if len(token) < 7:
            return None
        
        yy = token[0:2]
        mmm = token[2:5].upper()
        dd = token[5:7]
        rest = token[7:]  # Team codes
        
        if mmm not in MONTHS:
            return None
        
        yyyy = "20" + yy
        mm = MONTHS[mmm]
        yyyymmdd = f"{yyyy}{mm}{dd}"
        
        return (yyyymmdd, rest)
    except Exception:
        return None


def fetch_kalshi_events(api_key_id: str, private_key_pem: str, series_ticker: str = None) -> List[Dict[str, Any]]:
    """
    Fetch open Kalshi events for a given series.
    
    Args:
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
        series_ticker: Series ticker (defaults to KALSHI_SERIES_TICKER for GAME events)
    
    Returns:
        List of event dicts
    """
    if series_ticker is None:
        series_ticker = KALSHI_SERIES_TICKER
    
    path = "/events"
    params = {
        "series_ticker": series_ticker,
        "status": "open",
        "with_nested_markets": "true"
    }
    
    resp = make_request(api_key_id, private_key_pem, "GET", path, params)
    return resp.get("events", [])


def refresh_kalshi_events_cache(api_key_id: str, private_key_pem: str):
    """
    Always fetch and rebuild Kalshi GAME events cache.
    Called by maybe_refresh_kalshi_events_cache().
    """
    events = fetch_kalshi_events(api_key_id, private_key_pem, KALSHI_SERIES_TICKER)
    
    # Build event_ticker → event index
    # We'll match by parsing tickers during lookup
    by_key = {}
    for event in events:
        event_ticker = event.get("event_ticker", "")
        if event_ticker:
            by_key[event_ticker] = event
    
    KALSHI_EVENTS_CACHE["by_key"] = by_key
    KALSHI_EVENTS_CACHE["fetched_at"] = time.time()


def refresh_kalshi_totals_events_cache(api_key_id: str, private_key_pem: str):
    """
    Always fetch and rebuild Kalshi TOTALS events cache.
    Called by maybe_refresh_kalshi_totals_events_cache().
    """
    totals_series_ticker = "KXNCAAMBTOTAL"
    events = fetch_kalshi_events(api_key_id, private_key_pem, totals_series_ticker)
    
    # Build event_ticker → event index
    # We'll match by parsing tickers during lookup
    by_key = {}
    for event in events:
        event_ticker = event.get("event_ticker", "")
        if event_ticker:
            by_key[event_ticker] = event
    
    KALSHI_TOTALS_EVENTS_CACHE["by_key"] = by_key
    KALSHI_TOTALS_EVENTS_CACHE["fetched_at"] = time.time()


def maybe_refresh_kalshi_totals_events_cache(api_key_id: str, private_key_pem: str):
    """
    Refresh Kalshi totals events cache if it's older than TTL.
    Uses last good cache on failure.
    """
    if time.time() - KALSHI_TOTALS_EVENTS_CACHE["fetched_at"] > KALSHI_EVENTS_TTL:
        try:
            refresh_kalshi_totals_events_cache(api_key_id, private_key_pem)
        except Exception:
            # If refresh fails, continue using last good cache
            if KALSHI_TOTALS_EVENTS_CACHE["by_key"] == {}:
                raise  # Nothing to fall back to
            # else: log and continue (logging can be added later)


def maybe_refresh_kalshi_events_cache(api_key_id: str, private_key_pem: str):
    """
    Refresh Kalshi events cache if it's older than TTL.
    Uses last good cache on failure.
    """
    if time.time() - KALSHI_EVENTS_CACHE["fetched_at"] > KALSHI_EVENTS_TTL:
        try:
            refresh_kalshi_events_cache(api_key_id, private_key_pem)
        except Exception:
            # If refresh fails, continue using last good cache
            if KALSHI_EVENTS_CACHE["by_key"] == {}:
                raise  # Nothing to fall back to
            # else: log and continue (logging can be added later)


def match_kalshi_event(
    canonical_key: str,
    team_codes: Tuple[str, str],
    use_totals_cache: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Match a Kalshi event by canonical key and team codes using cached events.
    
    Args:
        canonical_key: Canonical key (e.g., "CBB_20251215_NIAG_VCU")
        team_codes: Tuple of (team_a, team_b) codes
        use_totals_cache: If True, search totals events cache; otherwise search GAME events cache
    
    Returns:
        Event dict if found, None otherwise
    """
    # Parse canonical key
    parts = canonical_key.split("_")
    if len(parts) != 4:
        return None
    
    league, date_str, team_a, team_b = parts
    
    # Select which cache to search
    cache = KALSHI_TOTALS_EVENTS_CACHE if use_totals_cache else KALSHI_EVENTS_CACHE
    
    # Search through cached events
    for event_ticker, event in cache["by_key"].items():
        parsed = parse_kalshi_event_ticker(event_ticker)
        
        if not parsed:
            continue
        
        event_date, event_team_codes = parsed
        
        # Check date match
        if event_date != date_str:
            continue
        
        # Check if both team codes appear in the ticker suffix
        # Note: This assumes team codes are substrings in the ticker suffix.
        # For CBB, this works because codes are short (e.g., "NIAG", "VCU") and
        # appear distinctly in tickers like "NIAGVCU". Do not "improve" this
        # with fuzzy matching or ordering assumptions.
        code_a, code_b = team_codes
        if code_a in event_team_codes and code_b in event_team_codes:
            return event
    
    return None


# ============================================================================
# Price Conversion
# ============================================================================

def cents_to_american(price_cents: int) -> int:
    """
    Convert Kalshi price (cents) to American odds.
    
    Args:
        price_cents: Price in cents (0-100)
    
    Returns:
        American odds (e.g., -110, +150)
    """
    if price_cents <= 0 or price_cents >= 100:
        return 0  # Invalid
    
    P = price_cents / 100.0
    
    if P >= 0.5:
        # Favorite (negative odds)
        odds = int(round(-100.0 * P / (1.0 - P)))
    else:
        # Underdog (positive odds)
        odds = int(round(100.0 * (1.0 - P) / P))
    
    return odds


def cents_to_american(price_cents: int) -> int:
    """
    Convert Kalshi price (cents) to American odds.
    
    Args:
        price_cents: Price in cents (0-100)
    
    Returns:
        American odds (e.g., -110, +150)
    """
    if price_cents <= 0 or price_cents >= 100:
        return 0  # Invalid
    
    P = price_cents / 100.0
    
    if P >= 0.5:
        # Favorite (negative odds)
        odds = int(round(-100.0 * P / (1.0 - P)))
    else:
        # Underdog (positive odds)
        odds = int(round(100.0 * (1.0 - P) / P))
    
    return odds


def american_to_cents(odds: int) -> int:
    """
    Convert American odds to Kalshi price in cents.
    """
    if odds < 0:
        p = (-odds) / ((-odds) + 100.0)
    else:
        p = 100.0 / (odds + 100.0)
    
    return int(round(p * 100))


# ============================================================================
# Fee Calculation
# ============================================================================

def fee_dollars(contracts: int, price_cents: int) -> float:
    """
    Calculate trading fees: round up to next cent of 0.07 * C * P * (1-P),
    where P is price in dollars.
    """
    P = price_cents / 100.0
    raw = config.FEE_RATE * contracts * P * (1.0 - P)
    return math.ceil(raw * 100.0) / 100.0


def maker_fee_cents(price_cents: int, contracts: int = 1) -> int:
    """
    Calculate maker fee in cents: ceil(0.0175 * C * P * (1 - P) * 100).
    
    Args:
        price_cents: Fill price in cents
        contracts: Number of contracts (default 1 for calibration)
    
    Returns:
        Maker fee in cents (rounded up)
    """
    P = price_cents / 100.0
    raw_fee_dollars = 0.0175 * contracts * P * (1.0 - P)
    fee_cents = math.ceil(raw_fee_dollars * 100.0)
    return int(fee_cents)


def adjust_maker_price_for_fees(limit_price_cents: int) -> Optional[int]:
    """
    Adjust maker price downward to account for fees.
    
    User's limit_price_cents is interpreted as "max effective price after fees".
    This function finds the highest postable price such that:
        post_price_cents + maker_fee(post_price_cents, C=1) <= limit_price_cents
    
    Args:
        limit_price_cents: Maximum effective price (post-fee) in cents
    
    Returns:
        Highest valid post_price_cents, or None if no valid price exists
    
    Example:
        limit_price_cents = 90 (user wants -900, i.e. 90¢ net)
        Returns ~89 (post at 89¢, fee = 1¢, effective = 90¢)
    """
    # Edge case: very low prices
    if limit_price_cents <= 2:
        return None
    
    # Search downward from limit to find highest valid post price
    # Start at limit - 1 to ensure we're below (since fee will add)
    for post_price in range(limit_price_cents - 1, 0, -1):
        fee_cents = maker_fee_cents(post_price, contracts=1)
        effective_price = post_price + fee_cents
        
        if effective_price <= limit_price_cents:
            return post_price
    
    # No valid price found (shouldn't happen for reasonable limits)
    return None


def level_all_in_cost(contracts: int, price_cents: int) -> float:
    """
    Calculate total cost (contracts * price + fees) for a price level.
    """
    contract_cost = contracts * (price_cents / 100.0)
    fees = fee_dollars(contracts, price_cents)
    return contract_cost + fees


def max_affordable_contracts(remaining: float, price_cents: int, available: int) -> int:
    """
    Find maximum number of contracts affordable at a given price level.
    """
    for c in range(available, 0, -1):
        if level_all_in_cost(c, price_cents) <= remaining + 1e-9:
            return c
    return 0


# ============================================================================
# Orderbook and Execution
# ============================================================================

def fetch_orderbook(api_key_id: str, private_key_pem: str, market_ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch Kalshi orderbook for a market.
    """
    path = f"/markets/{market_ticker}/orderbook"
    
    try:
        resp = make_request(api_key_id, private_key_pem, "GET", path)
        return resp.get("orderbook", {})
    except Exception:
        return None


def fetch_kalshi_markets_for_event(
    api_key_id: str,
    private_key_pem: str,
    event_ticker: str
) -> List[Dict[str, Any]]:
    """
    Fetch all markets for a Kalshi event (including totals, moneylines, etc.).
    
    Args:
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
        event_ticker: Event ticker (e.g., "KXNCAAMBGAME-25DEC15NIAGVCU")
    
    Returns:
        List of market dicts for the event
    """
    path = "/markets"
    params = {
        "event_ticker": event_ticker,
        "status": "open"
    }
    
    try:
        resp = make_request(api_key_id, private_key_pem, "GET", path, params)
        return resp.get("markets", [])
    except Exception as e:
        print(f"❌ Failed to fetch markets for event {event_ticker}: {e}")
        return []


def derive_implied_yes_asks(no_bids: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Derive implied YES asks from NO bids.
    Returns list of (price_cents, qty) sorted by price ascending.
    """
    if not no_bids:
        return []
    
    # Best bid is last element
    implied_asks = []
    for no_price, no_qty in no_bids:
        yes_ask = 100 - no_price
        implied_asks.append((yes_ask, no_qty))
    
    # Sort by price ascending (lowest first)
    implied_asks.sort(key=lambda x: x[0])
    return implied_asks


def derive_implied_no_asks(yes_bids: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Derive implied NO asks from YES bids.
    Returns list of (price_cents, qty) sorted by price ascending.
    """
    if not yes_bids:
        return []
    
    # Best bid is last element
    implied_asks = []
    for yes_price, yes_qty in yes_bids:
        no_ask = 100 - yes_price
        implied_asks.append((no_ask, yes_qty))
    
    # Sort by price ascending (lowest first)
    implied_asks.sort(key=lambda x: x[0])
    return implied_asks


def determine_execution_prices(
    orderbook: Dict[str, Any],
    side: str,
    max_price_cents: int
) -> List[Tuple[int, int]]:
    """
    Determine execution prices from orderbook for taker-style execution.
    
    Args:
        orderbook: Kalshi orderbook dict with "yes" and "no" bid lists
        side: "yes" or "no"
        max_price_cents: Maximum acceptable price in cents
    
    Returns:
        List of (price_cents, available_qty) tuples sorted by price ascending.
        Only includes prices <= max_price_cents.
    """
    if side == "yes":
        no_bids = orderbook.get("no") or []  # Normalize None to []
        implied_asks = derive_implied_yes_asks(no_bids)
    elif side == "no":
        yes_bids = orderbook.get("yes") or []  # Normalize None to []
        implied_asks = derive_implied_no_asks(yes_bids)
    else:
        return []
    
    # Filter by max price
    return [(p, q) for p, q in implied_asks if p <= max_price_cents]


def determine_execution_price_maker(
    orderbook: Dict[str, Any],
    side: str,
    limit_price_cents: int
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Determine if maker order price can be posted without immediately matching.
    
    Maker orders are rejected ONLY if they would cross the implied ask.
    Implied asks are derived from OPPOSITE-SIDE bids, not same-side bids.
    Bids do not trade with bids - only bids trade with implied asks.
    
    Args:
        orderbook: Kalshi orderbook dict with "yes" and "no" bid lists (bid-only per side)
        side: "yes" or "no"
        limit_price_cents: Proposed limit price in cents
    
    Returns:
        (is_valid, implied_ask_cents, error_message)
        is_valid: True if price can be posted without crossing, False otherwise
        implied_ask_cents: Current implied ask price (for error messages)
        error_message: None if valid, error message if invalid
    
    Rules:
    - YES maker: Compute implied YES ask from NO bids. Reject if limit_price >= implied_yes_ask
    - NO maker: Compute implied NO ask from YES bids. Reject if limit_price >= implied_no_ask
    - If opposite-side bids don't exist: implied ask = 100 (no sellers, always safe)
    - Never compare against same-side bids
    - Maker orders are permissive by default (allowed to invent the market)
    """
    if side == "yes":
        # Read NO bids (OPPOSITE side) to compute implied YES ask
        no_bids = orderbook.get("no") or []  # Normalize None to []
        
        if not no_bids:
            # No NO bids exist - no sellers, implied YES ask = 100
            implied_yes_ask = 100
            best_no_bid = None
            print(f"DEBUG(maker): side=yes limit={limit_price_cents} best_opposite_bid=None implied_ask={implied_yes_ask} -> allow_post=True (no opposite bids)")
        else:
            # Compute implied YES ask from best NO bid
            best_no_bid = no_bids[-1][0]  # Last element is best bid
            implied_yes_ask = 100 - best_no_bid
            print(f"DEBUG(maker): side=yes limit={limit_price_cents} best_opposite_bid={best_no_bid} implied_ask={implied_yes_ask} -> ", end="")
        
        # Validation: limit price must be < implied ask to avoid crossing
        if limit_price_cents >= implied_yes_ask:
            print(f"allow_post=False (would cross)")
            return False, implied_yes_ask, f"Limit price {limit_price_cents}¢ would cross the book. Implied YES ask: {implied_yes_ask}¢"
        
        if no_bids:
            print(f"allow_post=True")
        return True, implied_yes_ask, None
    
    elif side == "no":
        # Read YES bids (OPPOSITE side) to compute implied NO ask
        yes_bids = orderbook.get("yes") or []  # Normalize None to []
        
        if not yes_bids:
            # No YES bids exist - no sellers, implied NO ask = 100
            implied_no_ask = 100
            best_yes_bid = None
            print(f"DEBUG(maker): side=no limit={limit_price_cents} best_opposite_bid=None implied_ask={implied_no_ask} -> allow_post=True (no opposite bids)")
        else:
            # Compute implied NO ask from best YES bid
            best_yes_bid = yes_bids[-1][0]  # Last element is best bid
            implied_no_ask = 100 - best_yes_bid
            print(f"DEBUG(maker): side=no limit={limit_price_cents} best_opposite_bid={best_yes_bid} implied_ask={implied_no_ask} -> ", end="")
        
        # Validation: limit price must be < implied ask to avoid crossing
        if limit_price_cents >= implied_no_ask:
            print(f"allow_post=False (would cross)")
            return False, implied_no_ask, f"Limit price {limit_price_cents}¢ would cross the book. Implied NO ask: {implied_no_ask}¢"
        
        if yes_bids:
            print(f"allow_post=True")
        return True, implied_no_ask, None
    
    else:
        return False, None, f"Invalid side: {side}"


def place_kalshi_order(
    api_key_id: str,
    private_key_pem: str,
    market_ticker: str,
    count: int,
    price_cents: int,
    side: str = "yes",
    execution_mode: str = "taker"
) -> Optional[Dict[str, Any]]:
    """
    Place a limit buy order (taker or maker).
    
    Args:
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
        market_ticker: Market ticker
        count: Number of contracts
        price_cents: Price in cents
        side: "yes" for moneyline/totals over, "no" for totals under
        execution_mode: "taker" (crosses immediately) or "maker" (post-only)
    
    Returns response dict or None on error.
    """
    path = "/portfolio/orders"
    
    # Build order body based on side
    body = {
        "ticker": market_ticker,
        "type": "limit",
        "action": "buy",
        "side": side,
        "count": count,
        "client_order_id": str(uuid.uuid4())
    }
    
    # Set price field based on side
    if side == "yes":
        body["yes_price"] = price_cents
    elif side == "no":
        body["no_price"] = price_cents
    else:
        print(f"❌ Invalid side: {side}")
        return None
    
    # Maker mode: post-only (do not cross)
    if execution_mode == "maker":
        body["post_only"] = True
    
    try:
        resp = make_request(api_key_id, private_key_pem, "POST", path, body)
        return resp
    except Exception as e:
        print(f"❌ Failed to place order: {e}")
        return None


# ============================================================================
# Order Management (Kill Command)
# ============================================================================

def fetch_open_orders(api_key_id: str, private_key_pem: str) -> List[Dict[str, Any]]:
    """
    Fetch all resting orders for the authenticated user.
    Always queries Kalshi API live (no session cache).
    Supports pagination.
    
    Args:
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
    
    Returns:
        List of order dicts with status="resting"
    """
    path = "/portfolio/orders"
    all_orders = []
    cursor = None
    limit = 200
    
    try:
        while True:
            params = {
                "status": "resting",
                "limit": limit
            }
            if cursor:
                params["cursor"] = cursor
            
            resp = make_request(api_key_id, private_key_pem, "GET", path, params)
            
            # Primary response structure: resp["orders"]
            orders = resp.get("orders", [])
            if orders:
                all_orders.extend(orders)
            
            # Check for pagination cursor
            cursor = resp.get("cursor") or resp.get("next_cursor")
            if not cursor:
                break
        
        # Debug: print response keys if structure differs
        if not all_orders and resp:
            print(f"DEBUG: Response keys: {list(resp.keys())}")
            if "orders" not in resp:
                print(f"DEBUG: Full response structure: {resp}")
        
        return all_orders
    except Exception as e:
        print(f"❌ Failed to fetch resting orders: {e}")
        return []


def cancel_order(api_key_id: str, private_key_pem: str, order_id: str) -> bool:
    """
    Cancel a single resting order by order_id.
    Uses DELETE /portfolio/orders/{order_id}.
    Cancels remaining resting quantity (safe for partially-filled orders).
    
    Args:
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
        order_id: Order ID to cancel
    
    Returns:
        True if canceled successfully, False otherwise
    """
    path = f"/portfolio/orders/{order_id}"
    
    try:
        resp = make_request(api_key_id, private_key_pem, "DELETE", path)
        
        # Success: HTTP 200 with JSON or HTTP 204
        # Log response details if present
        if resp.get("order"):
            order_info = resp.get("order", {})
            reduced_by = order_info.get("reduced_by")
            if reduced_by is not None:
                print(f"DEBUG: Order {order_id} reduced_by: {reduced_by}")
        
        # Request succeeded (200 or 204)
        return True
    except Exception as e:
        print(f"❌ Failed to cancel order {order_id}: {e}")
        # Retry once on transient failure
        try:
            time.sleep(0.5)
            resp = make_request(api_key_id, private_key_pem, "DELETE", path)
            return True
        except Exception:
            pass
        return False


def resolve_rotation_from_market_ticker(
    api_key_id: str,
    private_key_pem: str,
    market_ticker: str
) -> Optional[int]:
    """
    Resolve rotation number from a market ticker.
    
    This performs reverse lookup:
    1. Extract event ticker from market ticker
    2. Parse event ticker to get date and team codes
    3. Build canonical key
    4. Find matching game in Unabated cache by canonical key
    
    Args:
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
        market_ticker: Market ticker (e.g., "KXNCAAMBGAME-25DEC15NIAGVCU-VCU")
    
    Returns:
        Rotation number if found, None otherwise
    """
    # Extract event ticker from market ticker
    # Format: {event_ticker}-{suffix}
    # Example: KXNCAAMBGAME-25DEC15NIAGVCU-VCU -> KXNCAAMBGAME-25DEC15NIAGVCU
    if "-" not in market_ticker:
        return None
    
    parts = market_ticker.split("-")
    if len(parts) < 2:
        return None
    
    # For moneyline: KXNCAAMBGAME-25DEC15NIAGVCU-VCU -> event is KXNCAAMBGAME-25DEC15NIAGVCU
    # For totals: KXNCAAMBTOTAL-25DEC15NIAGVCU-141 -> event is KXNCAAMBTOTAL-25DEC15NIAGVCU
    event_ticker = None
    
    # Try moneyline event (remove last part)
    if len(parts) >= 3:
        candidate = "-".join(parts[:-1])
        # Check if this event exists in cache
        if candidate in KALSHI_EVENTS_CACHE["by_key"]:
            event_ticker = candidate
    
    # Try totals event (remove last two parts if last part is a number)
    if not event_ticker and len(parts) >= 3:
        try:
            int(parts[-1])  # Check if last part is a number (totals line)
            candidate = "-".join(parts[:-2])
            if candidate in KALSHI_TOTALS_EVENTS_CACHE["by_key"]:
                event_ticker = candidate
        except ValueError:
            pass
    
    if not event_ticker:
        return None
    
    # Parse event ticker to get date and team codes
    parsed = parse_kalshi_event_ticker(event_ticker)
    if not parsed:
        return None
    
    event_date, team_codes_str = parsed
    
    # Lazy refresh Unabated cache if needed
    maybe_refresh_unabated_cache()
    
    # Iterate through Unabated games to find matching rotation
    # Build canonical key for each game and compare
    for roto, game in UNABATED_CACHE["roto_to_game"].items():
        event_start = game.get("eventStart")
        if not event_start:
            continue
        
        # Build canonical key from Unabated game
        event_teams_raw = game.get("eventTeams", {})
        if isinstance(event_teams_raw, dict):
            event_teams = list(event_teams_raw.values())
        elif isinstance(event_teams_raw, list):
            event_teams = event_teams_raw
        else:
            continue
        
        if len(event_teams) != 2:
            continue
        
        # Resolve team names via teams lookup
        teams_lookup = UNABATED_CACHE["teams"]
        team_codes = []
        for t in event_teams:
            if not isinstance(t, dict):
                continue
            team_id = t.get("id")
            if team_id is None:
                continue
            team_info = teams_lookup.get(str(team_id), {})
            team_name = team_info.get("name", "").strip()
            if not team_name:
                continue
            code = team_to_kalshi_code(LEAGUE, team_name)
            if code:
                team_codes.append(code)
        
        if len(team_codes) != 2:
            continue
        
        # Build canonical key
        canonical_key = build_canonical_key(LEAGUE, event_start, team_codes[0], team_codes[1])
        
        # Check if this canonical key matches the event date and team codes
        key_parts = canonical_key.split("_")
        if len(key_parts) == 4:
            key_league, key_date, key_team_a, key_team_b = key_parts
            if key_date == event_date:
                # Check if both team codes appear in the ticker suffix
                if key_team_a in team_codes_str and key_team_b in team_codes_str:
                    return int(roto)
    
    return None


def execute_kill(
    kill_command: Dict[str, Any],
    api_key_id: str,
    private_key_pem: str,
    chat_id: Optional[str] = None,
    cli_mode: bool = False
) -> None:
    """
    Execute kill command to cancel resting orders.
    
    Args:
        kill_command: Parsed kill command dict
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
        chat_id: Telegram chat ID (for Telegram mode)
        cli_mode: If True, print to stdout instead of Telegram
    """
    command_type = kill_command.get("command_type")
    target_roto = kill_command.get("rotation_number")
    
    # Fetch all resting orders (always queries API live)
    resting_orders = fetch_open_orders(api_key_id, private_key_pem)
    
    # Defensive logging: print summary before canceling
    print(f"Found {len(resting_orders)} resting orders")
    
    if not resting_orders:
        msg = "No resting orders found."
        print(f"DEBUG: Query params used: status=resting, limit=200")
        send_response(msg, chat_id, cli_mode)
        return
    
    # Print first 5 orders for debugging
    print("First orders found:")
    for i, order in enumerate(resting_orders[:5]):
        order_id = order.get("order_id") or order.get("id", "unknown")
        ticker = order.get("ticker") or order.get("market_ticker", "unknown")
        status = order.get("status", "unknown")
        price = order.get("yes_price") or order.get("no_price") or order.get("price", 0)
        remaining = order.get("remaining_count") or order.get("remaining_quantity") or order.get("count", 0)
        side = order.get("side", "unknown")
        print(f"  [{i+1}] {order_id} | {ticker} | {side.upper()} @ {price}¢ | remaining={remaining} | status={status}")
    
    # Filter orders by roto if specified
    orders_to_cancel = []
    skipped_orders = []
    
    if command_type == "kill_all":
        orders_to_cancel = resting_orders
    else:  # kill_roto
        # Resolve rotation for each order
        for order in resting_orders:
            market_ticker = order.get("ticker") or order.get("market_ticker", "")
            if not market_ticker:
                skipped_orders.append((order, "no market ticker"))
                continue
            
            resolved_roto = resolve_rotation_from_market_ticker(
                api_key_id, private_key_pem, market_ticker
            )
            
            if resolved_roto is None:
                skipped_orders.append((order, "could not resolve roto"))
                continue
            
            if resolved_roto == target_roto:
                orders_to_cancel.append(order)
            else:
                skipped_orders.append((order, f"different roto (R{resolved_roto})"))
    
    # Safety check: print summary before canceling
    if command_type == "kill_all":
        header = "KILL ALL ORDERS"
    else:
        header = f"KILL ROTATION {target_roto}"
    
    summary_msg = f"{header}\nFound {len(resting_orders)} resting orders"
    
    if command_type == "kill_roto":
        summary_msg += f"\nMatched {len(orders_to_cancel)} orders for rotation {target_roto}"
        if skipped_orders:
            summary_msg += f"\nSkipped {len(skipped_orders)} orders (different roto)"
    
    if not orders_to_cancel:
        summary_msg += "\nNo orders to cancel."
        send_response(summary_msg, chat_id, cli_mode)
        return
    
    summary_msg += f"\nCanceling {len(orders_to_cancel)} orders..."
    send_response(summary_msg, chat_id, cli_mode)
    
    # Cancel each order
    canceled_count = 0
    failed_count = 0
    
    for order in orders_to_cancel:
        order_id = order.get("order_id") or order.get("id")
        if not order_id:
            failed_count += 1
            continue
        
        market_ticker = order.get("ticker") or order.get("market_ticker", "")
        side = order.get("side", "unknown")
        price_cents = order.get("yes_price") or order.get("no_price") or order.get("price", 0)
        remaining = order.get("remaining_count") or order.get("remaining_quantity") or order.get("count", 0)
        
        success = cancel_order(api_key_id, private_key_pem, order_id)
        
        if success:
            canceled_count += 1
            status_msg = f"✓ Canceled order {order_id} ({market_ticker} @ {price_cents}¢ {side.upper()}, remaining={remaining})"
            send_response(status_msg, chat_id, cli_mode)
        else:
            failed_count += 1
            status_msg = f"✗ Failed to cancel order {order_id}"
            send_response(status_msg, chat_id, cli_mode)
    
    # Final summary
    summary = f"Done.\nCanceled: {canceled_count}\nFailed: {failed_count}"
    if skipped_orders and command_type == "kill_roto":
        summary += f"\nSkipped: {len(skipped_orders)}"
    send_response(summary, chat_id, cli_mode)


# ============================================================================
# Market Selection
# ============================================================================

def validate_rotation_parity(rotation_number: int, side: str) -> tuple[bool, Optional[str]]:
    """
    Validate rotation parity matches side (safety check).
    
    Rules:
    - Odd rotation number → Over
    - Even rotation number → Under
    
    Args:
        rotation_number: Rotation number from turn-in
        side: "over" or "under"
    
    Returns:
        (is_valid, error_message)
        is_valid: True if parity matches side, False otherwise
        error_message: None if valid, error message if invalid
    """
    is_odd = (rotation_number % 2) == 1
    expected_side = "over" if is_odd else "under"
    
    if side.lower() != expected_side:
        error = (
            f"Rotation parity mismatch: Rotation {rotation_number} is "
            f"{'odd' if is_odd else 'even'} (expects {expected_side}), "
            f"but turn-in specified {side}. No bet placed."
        )
        return False, error
    
    return True, None


def find_totals_market(
    markets: List[Dict[str, Any]],
    side: str,
    line: float
) -> Optional[Dict[str, Any]]:
    """
    Find totals market matching side and line.
    
    Args:
        markets: List of ALL market dicts for the totals event
        side: "over" or "under" (used for validation only - side is determined by YES/NO in market)
        line: Total points line (e.g., 141.5)
    
    Returns:
        Market dict if found, None otherwise
    
    Note: Kalshi encodes X.5 totals as integers in the market ticker suffix.
    Example: 141.5 → market suffix "-141", 165.5 → market suffix "-165"
    Rule: kalshi_line = int(user_line) (no rounding, no tolerance, no guessing)
    Inside a totals market, YES = Over, NO = Under.
    """
    # Convert line to Kalshi encoding: int(line) for suffix matching
    # Example: 141.5 → 141, 165.5 → 165
    kalshi_line_suffix = int(line)
    target_suffix = f"-{kalshi_line_suffix}"
    
    for market in markets:
        ticker = market.get("ticker", "")
        
        # Match by exact suffix (e.g., "-141" for line 141.5)
        if ticker.endswith(target_suffix):
            # Found the correct totals market for this line
            # Side is determined by YES/NO when placing order, not market selection
            return market
    
    return None


# ============================================================================
# Response Helper
# ============================================================================

def send_response(msg: str, chat_id: Optional[str] = None, cli_mode: bool = False) -> None:
    """
    Send response message via Telegram or print to stdout (CLI mode).
    """
    if cli_mode:
        print(msg)
    else:
        send_telegram_message(msg, chat_id)


# ============================================================================
# Execution Functions
# ============================================================================

def execute_taker_orders(
    api_key_id: str,
    private_key_pem: str,
    market_ticker: str,
    execution_prices: List[Tuple[int, int]],
    order_side: str,
    remaining_budget: float,
    chat_id: Optional[str] = None,
    cli_mode: bool = False
) -> Dict[str, Any]:
    """
    Execute taker-style orders (cross the book immediately).
    
    Args:
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
        market_ticker: Market ticker
        execution_prices: List of (price_cents, available_qty) tuples
        order_side: "yes" or "no"
        remaining_budget: Remaining budget in dollars
        chat_id: Telegram chat ID (for error messages)
        cli_mode: If True, print to stdout instead of Telegram
    
    Returns:
        Dict with standardized execution result:
        {
            "filled_contracts": int,
            "total_cost": float,
            "total_fees": float,
            "total_price_weighted": float,
            "orders_placed": bool,
            "status": "filled" | "partial" | "none" | "error"
        }
    
    """
    # Execution tracking
    total_contracts = 0
    total_cost = 0.0
    total_fees = 0.0
    total_price_weighted = 0.0  # For VWAP calculation
    orders_placed = False
    remaining = remaining_budget
    
    # Execute: buy up orderbook
    for price_cents, available_qty in execution_prices:
        orders_placed = True
        
        # Check if we can afford any contracts at this level
        affordable = max_affordable_contracts(remaining, price_cents, available_qty)
        if affordable == 0:
            continue
        
        # Place order (taker-style: crosses immediately)
        order_resp = place_kalshi_order(
            api_key_id=api_key_id,
            private_key_pem=private_key_pem,
            market_ticker=market_ticker,
            count=affordable,
            price_cents=price_cents,
            side=order_side,
            execution_mode="taker"
        )
        
        if not order_resp:
            # Order failed, stop execution
            break
        
        # Check for errors in response
        if not order_resp.get("order", {}):
            send_response("Order rejected by Kalshi. Execution stopped.", chat_id, cli_mode)
            return {
                "requested_contracts": sum(q for _, q in execution_prices),
                "filled_contracts": total_contracts,
                "avg_price_cents": None,
                "total_spend": total_cost + total_fees,
                "fees": total_fees,
                "order_id": None,
                "status": "error"
            }
        
        # Extract filled count (assume requested if not provided)
        filled_count = affordable  # Default assumption
        # TODO: Extract actual filled_count from order_resp if available
        
        # Calculate cost and fees
        level_cost = level_all_in_cost(filled_count, price_cents)
        level_fees = fee_dollars(filled_count, price_cents)
        
        # Update totals
        total_contracts += filled_count
        total_cost += filled_count * (price_cents / 100.0)
        total_fees += level_fees
        total_price_weighted += filled_count * price_cents  # For VWAP
        remaining -= level_cost
        
        # Stop if budget exhausted
        if remaining <= 0:
            break
    
    # Determine status
    if total_contracts == 0:
        status = "none"
    elif remaining <= 0 or not orders_placed:
        status = "filled"  # Budget exhausted or no more liquidity
    else:
        status = "partial"  # Some filled but budget remaining
    
    return {
        "requested_contracts": sum(q for _, q in execution_prices),
        "filled_contracts": total_contracts,
        "avg_price_cents": int(total_price_weighted / total_contracts) if total_contracts > 0 else None,
        "total_spend": total_cost + total_fees,
        "fees": total_fees,
        "order_id": None,  # Taker orders don't return order_id
        "status": status
    }


def execute_maker_moneyline(
    api_key_id: str,
    private_key_pem: str,
    market_ticker: str,
    order_side: str,
    limit_price_cents: int,
    budget_dollars: float,
    bet_team_name: Optional[str],
    bet_team_code: str,
    rotation_number: int,
    chat_id: Optional[str] = None,
    cli_mode: bool = False
) -> Dict[str, Any]:
    """
    Execute maker-style moneyline order (post-only, fire-and-forget).
    
    This function is hard-separated from taker logic:
    - Does NOT fetch orderbook for liquidity checks
    - Does NOT check liquidity
    - Does NOT reference yes_bids or no_bids
    - Only validates price improvement (fetches orderbook internally for that)
    - Posts resting order and returns immediately
    
    Args:
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
        market_ticker: Market ticker
        order_side: "yes" or "no"
        limit_price_cents: Limit price in cents (must improve book)
        budget_dollars: Budget in dollars
        bet_team_name: Team name for error messages
        bet_team_code: Team code for error messages
        rotation_number: Rotation number for error messages
        chat_id: Telegram chat ID (for error messages)
        cli_mode: If True, print to stdout instead of Telegram
    
    Returns:
        Dict with standardized execution result:
        {
            "requested_contracts": int,
            "filled_contracts": int (always 0 for maker),
            "avg_price_cents": Optional[int] (None for maker),
            "total_spend": float (0.0 for maker),
            "fees": float (0.0 for maker),
            "order_id": Optional[str] (populated if posted),
            "status": "none" | "error"
        }
    """
    # Adjust limit price for maker fees (user's limit is post-fee requirement)
    # limit_price_cents is the max effective price after fees
    # We need to post at a lower price to account for fees
    post_price_cents = adjust_maker_price_for_fees(limit_price_cents)
    
    if post_price_cents is None:
        send_response(f"Maker order rejected: Cannot find valid post price for limit {limit_price_cents}¢ (too low or fee calculation failed)", chat_id, cli_mode)
        return {
            "requested_contracts": 0,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "status": "error"
        }
    
    # Calculate fee for debug logging
    fee_cents = maker_fee_cents(post_price_cents, contracts=1)
    effective_price = post_price_cents + fee_cents
    print(f"DEBUG(maker): user_limit={limit_price_cents} post_price={post_price_cents} fee={fee_cents} effective={effective_price}")
    
    # Validate price improvement (fetches orderbook internally, but doesn't expose it)
    orderbook = fetch_orderbook(api_key_id, private_key_pem, market_ticker)
    if not orderbook:
        send_response("Failed to fetch orderbook for price validation", chat_id, cli_mode)
        return {
            "requested_contracts": 0,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "status": "error"
        }
    
    # Use adjusted post_price for validation (must not cross book)
    is_valid, best_price, error_msg = determine_execution_price_maker(
        orderbook, order_side, post_price_cents
    )
    
    if not is_valid:
        # Price does not improve book - abort
        warning = f"Maker order rejected: {error_msg}"
        if best_price is not None:
            warning += f" Current best: {best_price}¢"
        send_response(warning, chat_id, cli_mode)
        return {
            "requested_contracts": 0,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "status": "error"
        }
    
    # Calculate contracts from budget
    # Use effective price (limit_price_cents) for contract calculation since that's what user pays
    effective_price_dollars = limit_price_cents / 100.0
    contracts = int(budget_dollars / effective_price_dollars)
    
    if contracts == 0:
        send_response("Budget too small for maker order (minimum 1 contract)", chat_id, cli_mode)
        return {
            "requested_contracts": 0,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "status": "error"
        }
    
    # Place single limit order (post-only) using adjusted post_price
    order_resp = place_kalshi_order(
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
        market_ticker=market_ticker,
        count=contracts,
        price_cents=post_price_cents,  # Use adjusted price, not limit_price
        side=order_side,
        execution_mode="maker"
    )
    
    if not order_resp:
        send_response("Failed to place maker order", chat_id, cli_mode)
        return {
            "requested_contracts": contracts,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "status": "error"
        }
    
    # Check for errors in response
    order_data = order_resp.get("order", {})
    if not order_data:
        send_response("Maker order rejected by Kalshi", chat_id, cli_mode)
        return {
            "requested_contracts": contracts,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "status": "error"
        }
    
    # Extract order ID
    order_id = order_data.get("order_id") or order_data.get("id")
    
    return {
        "requested_contracts": contracts,
        "filled_contracts": 0,  # Maker orders don't fill immediately
        "avg_price_cents": None,  # No fill price yet
        "total_spend": 0.0,  # No cost until filled
        "fees": 0.0,  # No fees until filled
        "order_id": order_id,
        "status": "none"  # Maker orders are posted, not filled
    }


def determine_execution_price_maker_totals(
    orderbook: Dict[str, Any],
    side: str,
    limit_price_cents: int
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Determine if maker totals order price can be posted without immediately matching.
    
    For totals maker orders, we check SAME-SIDE bids (not opposite-side).
    This is different from moneyline maker logic.
    
    Args:
        orderbook: Kalshi orderbook dict with "yes" and "no" bid lists (bid-only per side)
        side: "yes" (OVER) or "no" (UNDER)
        limit_price_cents: Proposed limit price in cents
    
    Returns:
        (is_valid, best_same_side_bid, error_message)
        is_valid: True if price can be posted without crossing, False otherwise
        best_same_side_bid: Current best same-side bid (for error messages)
        error_message: None if valid, error message if invalid
    
    Rules:
    - OVER (YES): Read YES bids. Reject if post_price >= best_yes_bid
    - UNDER (NO): Read NO bids. Reject if post_price >= best_no_bid
    - If same-side bids don't exist: always allow posting (inventing the market)
    - Never require opposite-side liquidity
    """
    if side == "yes":
        # Read YES bids (SAME side) for OVER
        yes_bids = orderbook.get("yes") or []  # Normalize None to []
        
        if not yes_bids:
            # No YES bids exist - always allow posting
            best_yes_bid = None
            print(f"DEBUG(maker totals): side=YES limit={limit_price_cents} best_same_side_bid=None -> allow_post=True (no same-side bids)")
            return True, None, None
        else:
            # Get best YES bid (last element is best bid)
            best_yes_bid = yes_bids[-1][0]
            print(f"DEBUG(maker totals): side=YES limit={limit_price_cents} best_same_side_bid={best_yes_bid} -> ", end="")
        
        # Validation: limit price must be < best same-side bid to avoid crossing
        if limit_price_cents >= best_yes_bid:
            print(f"allow_post=False (would cross)")
            return False, best_yes_bid, f"Limit price {limit_price_cents}¢ would cross the book. Best YES bid: {best_yes_bid}¢"
        
        print(f"allow_post=True")
        return True, best_yes_bid, None
    
    elif side == "no":
        # Read NO bids (SAME side) for UNDER
        no_bids = orderbook.get("no") or []  # Normalize None to []
        
        if not no_bids:
            # No NO bids exist - always allow posting
            best_no_bid = None
            print(f"DEBUG(maker totals): side=NO limit={limit_price_cents} best_same_side_bid=None -> allow_post=True (no same-side bids)")
            return True, None, None
        else:
            # Get best NO bid (last element is best bid)
            best_no_bid = no_bids[-1][0]
            print(f"DEBUG(maker totals): side=NO limit={limit_price_cents} best_same_side_bid={best_no_bid} -> ", end="")
        
        # Validation: limit price must be < best same-side bid to avoid crossing
        if limit_price_cents >= best_no_bid:
            print(f"allow_post=False (would cross)")
            return False, best_no_bid, f"Limit price {limit_price_cents}¢ would cross the book. Best NO bid: {best_no_bid}¢"
        
        print(f"allow_post=True")
        return True, best_no_bid, None
    
    else:
        return False, None, f"Invalid side: {side}"


def execute_maker_totals(
    api_key_id: str,
    private_key_pem: str,
    market_ticker: str,
    order_side: str,
    limit_price_cents: int,
    budget_dollars: float,
    side: str,
    line: float,
    rotation_number: int,
    chat_id: Optional[str] = None,
    cli_mode: bool = False
) -> Dict[str, Any]:
    """
    Execute maker-style totals order (post-only, fire-and-forget).
    
    This function mirrors execute_maker_moneyline but for totals markets.
    Uses same-side bid checking (not opposite-side like moneyline).
    
    Args:
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
        market_ticker: Market ticker
        order_side: "yes" (OVER) or "no" (UNDER)
        limit_price_cents: Limit price in cents (must improve book, after fee adjustment)
        budget_dollars: Budget in dollars
        side: "over" or "under" (for display)
        line: Total points line (e.g., 141.5)
        rotation_number: Rotation number for error messages
        chat_id: Telegram chat ID (for error messages)
        cli_mode: If True, print to stdout instead of Telegram
    
    Returns:
        Dict with standardized execution result:
        {
            "requested_contracts": int,
            "filled_contracts": int (always 0 for maker),
            "avg_price_cents": Optional[int] (None for maker),
            "total_spend": float (0.0 for maker),
            "fees": float (0.0 for maker),
            "order_id": Optional[str] (populated if posted),
            "post_price_cents": int (adjusted price),
            "effective_price_cents": int (post_price + fee),
            "status": "none" | "error"
        }
    """
    # Adjust limit price for maker fees (user's limit is post-fee requirement)
    post_price_cents = adjust_maker_price_for_fees(limit_price_cents)
    
    if post_price_cents is None:
        send_response(f"Maker totals order rejected: Cannot find valid post price for limit {limit_price_cents}¢ (too low or fee calculation failed)", chat_id, cli_mode)
        return {
            "requested_contracts": 0,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "post_price_cents": None,
            "effective_price_cents": None,
            "status": "error"
        }
    
    # Calculate fee for debug logging
    fee_cents = maker_fee_cents(post_price_cents, contracts=1)
    effective_price_cents = post_price_cents + fee_cents
    print(f"DEBUG(maker totals): side={order_side.upper()} line={line} user_limit={limit_price_cents} post={post_price_cents} fee={fee_cents} effective={effective_price_cents}")
    
    # Validate price improvement (fetches orderbook internally)
    orderbook = fetch_orderbook(api_key_id, private_key_pem, market_ticker)
    if not orderbook:
        send_response("Failed to fetch orderbook for price validation", chat_id, cli_mode)
        return {
            "requested_contracts": 0,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "post_price_cents": None,
            "effective_price_cents": None,
            "status": "error"
        }
    
    # Use adjusted post_price for validation (must not cross book)
    is_valid, best_price, error_msg = determine_execution_price_maker_totals(
        orderbook, order_side, post_price_cents
    )
    
    if not is_valid:
        # Price does not improve book - abort
        warning = f"Maker totals order rejected: {error_msg}"
        if best_price is not None:
            warning += f" Current best: {best_price}¢"
        send_response(warning, chat_id, cli_mode)
        return {
            "requested_contracts": 0,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "post_price_cents": None,
            "effective_price_cents": None,
            "status": "error"
        }
    
    # Calculate contracts from budget
    # Use effective price (limit_price_cents) for contract calculation since that's what user pays
    effective_price_dollars = limit_price_cents / 100.0
    contracts = int(budget_dollars / effective_price_dollars)
    
    if contracts == 0:
        send_response("Budget too small for maker totals order (minimum 1 contract)", chat_id, cli_mode)
        return {
            "requested_contracts": 0,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "post_price_cents": None,
            "effective_price_cents": None,
            "status": "error"
        }
    
    # Place single limit order (post-only) using adjusted post_price
    order_resp = place_kalshi_order(
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
        market_ticker=market_ticker,
        count=contracts,
        price_cents=post_price_cents,  # Use adjusted price, not limit_price
        side=order_side,
        execution_mode="maker"
    )
    
    if not order_resp:
        send_response("Failed to place maker totals order", chat_id, cli_mode)
        return {
            "requested_contracts": contracts,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "post_price_cents": post_price_cents,
            "effective_price_cents": effective_price_cents,
            "status": "error"
        }
    
    # Check for errors in response
    order_data = order_resp.get("order", {})
    if not order_data:
        send_response("Maker totals order rejected by Kalshi", chat_id, cli_mode)
        return {
            "requested_contracts": contracts,
            "filled_contracts": 0,
            "avg_price_cents": None,
            "total_spend": 0.0,
            "fees": 0.0,
            "order_id": None,
            "post_price_cents": post_price_cents,
            "effective_price_cents": effective_price_cents,
            "status": "error"
        }
    
    # Extract order ID
    order_id = order_data.get("order_id") or order_data.get("id")
    
    return {
        "requested_contracts": contracts,
        "filled_contracts": 0,
        "avg_price_cents": None,
        "total_spend": 0.0,
        "fees": 0.0,
        "order_id": order_id,
        "post_price_cents": post_price_cents,
        "effective_price_cents": effective_price_cents,
        "status": "none"
    }


def execute_make_both_totals(
    api_key_id: str,
    private_key_pem: str,
    market_ticker: str,
    line: float,
    offset_ticks: int,
    budget_dollars: float,
    rotation_number: int,
    chat_id: Optional[str] = None,
    cli_mode: bool = False
) -> Dict[str, Any]:
    """
    Execute make both totals: post passive maker orders on both sides of the same totals line.
    
    Uses tick offsets from current best bids (not American odds).
    Each side is independent - no netting, no linking.
    
    This function is fully decoupled from juice-based logic.
    Pricing is pure cents, derived from orderbook best bids and offset_ticks.
    Fees apply only at fill time and do not constrain posting.
    
    Args:
        api_key_id: Kalshi API key ID
        private_key_pem: Kalshi private key PEM
        market_ticker: Market ticker for the totals line
        line: Total points line (e.g., 141.5)
        offset_ticks: Number of ticks behind best bid on each side
        budget_dollars: Total budget (split evenly across YES and NO)
        rotation_number: Rotation number for error messages
        chat_id: Telegram chat ID (for error messages)
        cli_mode: If True, print to stdout instead of Telegram
    
    Returns:
        Dict with execution result:
        {
            "yes": {"posted": bool, "price_cents": int, "contracts": int, "order_id": str, "reason": str},
            "no": {"posted": bool, "price_cents": int, "contracts": int, "order_id": str, "reason": str},
            "status": "both_posted" | "one_posted" | "none_posted" | "error"
        }
    """
    # Fetch orderbook
    orderbook = fetch_orderbook(api_key_id, private_key_pem, market_ticker)
    if not orderbook:
        send_response("Failed to fetch orderbook for make both", chat_id, cli_mode)
        return {
            "yes": {"posted": False, "price_cents": 0, "contracts": 0, "order_id": None, "reason": "orderbook fetch failed"},
            "no": {"posted": False, "price_cents": 0, "contracts": 0, "order_id": None, "reason": "orderbook fetch failed"},
            "status": "error"
        }
    
    # Get best bids (same-side only, never infer from opposite side)
    yes_bids = orderbook.get("yes") or []
    no_bids = orderbook.get("no") or []
    
    best_yes_bid = yes_bids[-1][0] if yes_bids else None
    best_no_bid = no_bids[-1][0] if no_bids else None
    
    print(f"DEBUG(make both): line={line}, offset={offset_ticks}, best_yes_bid={best_yes_bid}, best_no_bid={best_no_bid}")
    
    # Split budget evenly
    budget_per_side = budget_dollars / 2.0
    
    # Process YES side
    yes_result = {"posted": False, "price_cents": 0, "contracts": 0, "order_id": None, "reason": ""}
    
    if best_yes_bid is not None:
        # Compute posting price: best_bid - offset_ticks
        yes_post_price = best_yes_bid - offset_ticks
        
        # Validate: must not cross (post_price < best_bid)
        if yes_post_price >= best_yes_bid:
            yes_result["reason"] = f"would cross best bid ({best_yes_bid}¢)"
        elif yes_post_price < 1:
            yes_result["reason"] = "computed price too low (< 1¢)"
        else:
            # Calculate contracts from budget
            # Note: fees apply on fill, but we post at the computed price
            yes_price_dollars = yes_post_price / 100.0
            yes_contracts = int(budget_per_side / yes_price_dollars)
            
            if yes_contracts > 0:
                # Place YES order
                yes_order_resp = place_kalshi_order(
                    api_key_id=api_key_id,
                    private_key_pem=private_key_pem,
                    market_ticker=market_ticker,
                    count=yes_contracts,
                    price_cents=yes_post_price,
                    side="yes",
                    execution_mode="maker"
                )
                
                if yes_order_resp and yes_order_resp.get("order"):
                    yes_order_data = yes_order_resp.get("order", {})
                    yes_order_id = yes_order_data.get("order_id") or yes_order_data.get("id")
                    yes_result = {
                        "posted": True,
                        "price_cents": yes_post_price,
                        "contracts": yes_contracts,
                        "order_id": yes_order_id,
                        "reason": ""
                    }
                    print(f"DEBUG(make both): YES order posted @ {yes_post_price}¢, {yes_contracts} contracts")
                else:
                    yes_result["reason"] = "Kalshi rejected order"
            else:
                yes_result["reason"] = "budget too small for YES side"
    else:
        # Market invention: no YES bids exist
        # Post at a reasonable price (use offset from a default, or minimum)
        # For simplicity, use offset from 50¢ (midpoint) or minimum 1¢
        yes_post_price = max(1, 50 - offset_ticks)
        yes_price_dollars = yes_post_price / 100.0
        yes_contracts = int(budget_per_side / yes_price_dollars)
        
        if yes_contracts > 0:
            yes_order_resp = place_kalshi_order(
                api_key_id=api_key_id,
                private_key_pem=private_key_pem,
                market_ticker=market_ticker,
                count=yes_contracts,
                price_cents=yes_post_price,
                side="yes",
                execution_mode="maker"
            )
            
            if yes_order_resp and yes_order_resp.get("order"):
                yes_order_data = yes_order_resp.get("order", {})
                yes_order_id = yes_order_data.get("order_id") or yes_order_data.get("id")
                yes_result = {
                    "posted": True,
                    "price_cents": yes_post_price,
                    "contracts": yes_contracts,
                    "order_id": yes_order_id,
                    "reason": "market invented (no bids)"
                }
                print(f"DEBUG(make both): YES order posted @ {yes_post_price}¢ (market invented), {yes_contracts} contracts")
            else:
                yes_result["reason"] = "Kalshi rejected YES order"
        else:
            yes_result["reason"] = "budget too small for YES side"
    
    # Process NO side (same logic)
    no_result = {"posted": False, "price_cents": 0, "contracts": 0, "order_id": None, "reason": ""}
    
    if best_no_bid is not None:
        # Compute posting price: best_bid - offset_ticks
        no_post_price = best_no_bid - offset_ticks
        
        # Validate: must not cross (post_price < best_bid)
        if no_post_price >= best_no_bid:
            no_result["reason"] = f"would cross best bid ({best_no_bid}¢)"
        elif no_post_price < 1:
            no_result["reason"] = "computed price too low (< 1¢)"
        else:
            # Calculate contracts from budget
            # Note: fees apply on fill, but we post at the computed price
            no_price_dollars = no_post_price / 100.0
            no_contracts = int(budget_per_side / no_price_dollars)
            
            if no_contracts > 0:
                # Place NO order
                no_order_resp = place_kalshi_order(
                    api_key_id=api_key_id,
                    private_key_pem=private_key_pem,
                    market_ticker=market_ticker,
                    count=no_contracts,
                    price_cents=no_post_price,
                    side="no",
                    execution_mode="maker"
                )
                
                if no_order_resp and no_order_resp.get("order"):
                    no_order_data = no_order_resp.get("order", {})
                    no_order_id = no_order_data.get("order_id") or no_order_data.get("id")
                    no_result = {
                        "posted": True,
                        "price_cents": no_post_price,
                        "contracts": no_contracts,
                        "order_id": no_order_id,
                        "reason": ""
                    }
                    print(f"DEBUG(make both): NO order posted @ {no_post_price}¢, {no_contracts} contracts")
                else:
                    no_result["reason"] = "Kalshi rejected order"
            else:
                no_result["reason"] = "budget too small for NO side"
    else:
        # Market invention: no NO bids exist
        no_post_price = max(1, 50 - offset_ticks)
        no_price_dollars = no_post_price / 100.0
        no_contracts = int(budget_per_side / no_price_dollars)
        
        if no_contracts > 0:
            no_order_resp = place_kalshi_order(
                api_key_id=api_key_id,
                private_key_pem=private_key_pem,
                market_ticker=market_ticker,
                count=no_contracts,
                price_cents=no_post_price,
                side="no",
                execution_mode="maker"
            )
            
            if no_order_resp and no_order_resp.get("order"):
                no_order_data = no_order_resp.get("order", {})
                no_order_id = no_order_data.get("order_id") or no_order_data.get("id")
                no_result = {
                    "posted": True,
                    "price_cents": no_post_price,
                    "contracts": no_contracts,
                    "order_id": no_order_id,
                    "reason": "market invented (no bids)"
                }
                print(f"DEBUG(make both): NO order posted @ {no_post_price}¢ (market invented), {no_contracts} contracts")
            else:
                no_result["reason"] = "Kalshi rejected NO order"
        else:
            no_result["reason"] = "budget too small for NO side"
    
    # Determine status
    if yes_result["posted"] and no_result["posted"]:
        status = "both_posted"
    elif yes_result["posted"] or no_result["posted"]:
        status = "one_posted"
    else:
        status = "none_posted"
    
    return {
        "yes": yes_result,
        "no": no_result,
        "status": status
    }


def execute_totals_view(
    totals_command: Dict[str, Any],
    chat_id: Optional[str] = None,
    cli_mode: bool = False
) -> None:
    """
    Execute totals read-only view command.
    
    Fetches orderbook and displays TAKE/MAKE prices for both OVER and UNDER.
    
    Args:
        totals_command: Parsed totals command dict
        chat_id: Telegram chat ID (for Telegram mode, but this is CLI-only)
        cli_mode: If True, print to stdout
    """
    rotation_number = totals_command["rotation_number"]
    line = totals_command["line"]
    
    # Lazy refresh Unabated cache if stale
    maybe_refresh_unabated_cache()
    
    # Find game by rotation number
    game = find_unabated_game_by_roto(rotation_number)
    if not game:
        send_response(f"Rotation number {rotation_number} did not resolve to any game", chat_id, cli_mode)
        return
    
    # Extract game info
    event_start = game.get("eventStart")
    event_teams_raw = game.get("eventTeams", {})
    
    # Normalize eventTeams
    if isinstance(event_teams_raw, dict):
        event_teams = list(event_teams_raw.values())
    elif isinstance(event_teams_raw, list):
        event_teams = event_teams_raw
    else:
        send_response("Invalid Unabated eventTeams format", chat_id, cli_mode)
        return
    
    if not event_start or len(event_teams) != 2:
        send_response("Could not reliably identify both teams from Unabated data", chat_id, cli_mode)
        return
    
    # Resolve team names via teams lookup
    teams_lookup = UNABATED_CACHE["teams"]
    team_codes = []
    
    for t in event_teams:
        if not isinstance(t, dict):
            continue
        team_id = t.get("id")
        if team_id is None:
            continue
        team_info = teams_lookup.get(str(team_id), {})
        team_name = team_info.get("name", "").strip()
        if not team_name:
            continue
        code = team_to_kalshi_code(LEAGUE, team_name)
        if code:
            team_codes.append(code)
    
    if len(team_codes) != 2:
        send_response("Could not map both teams to Kalshi codes", chat_id, cli_mode)
        return
    
    # Build canonical key
    canonical_key = build_canonical_key(LEAGUE, event_start, team_codes[0], team_codes[1])
    
    # Load Kalshi credentials
    try:
        api_key_id, private_key_pem = load_creds()
    except FileNotFoundError as e:
        send_response(f"Missing Kalshi credentials: {e}", chat_id, cli_mode)
        return
    
    # Lazy refresh Kalshi totals events cache
    maybe_refresh_kalshi_totals_events_cache(api_key_id, private_key_pem)
    
    # Match Kalshi totals event
    matched_event = match_kalshi_event(canonical_key, tuple(team_codes), use_totals_cache=True)
    
    if not matched_event:
        send_response(f"Totals event not found for rotation {rotation_number}, line {line}", chat_id, cli_mode)
        return
    
    # Fetch all markets for the totals event
    event_ticker = matched_event.get("event_ticker")
    if not event_ticker:
        send_response("Event ticker not found", chat_id, cli_mode)
        return
    
    all_markets = fetch_kalshi_markets_for_event(api_key_id, private_key_pem, event_ticker)
    
    # Find totals market by line
    kalshi_line_suffix = int(line)
    target_suffix = f"-{kalshi_line_suffix}"
    
    selected_market = None
    for market in all_markets:
        ticker = market.get("ticker", "")
        if ticker.endswith(target_suffix):
            selected_market = market
            break
    
    if not selected_market:
        send_response(f"Totals market not found for line {line} (expected suffix: {target_suffix})", chat_id, cli_mode)
        return
    
    market_ticker = selected_market.get("ticker")
    if not market_ticker:
        send_response("Market ticker not found", chat_id, cli_mode)
        return
    
    # Fetch orderbook
    orderbook = fetch_orderbook(api_key_id, private_key_pem, market_ticker)
    if not orderbook:
        send_response("Failed to fetch orderbook or orderbook is empty", chat_id, cli_mode)
        return
    
    # Extract bids (Kalshi orderbook has bids only per side)
    yes_bids = orderbook.get("yes") or []
    no_bids = orderbook.get("no") or []
    
    # Derive asks from opposite-side bids
    # YES asks = 100 - NO bids
    # NO asks = 100 - YES bids
    yes_asks = []
    no_asks = []
    
    if no_bids:
        for no_price, no_qty in no_bids:
            yes_ask = 100 - no_price
            yes_asks.append((yes_ask, no_qty))
        yes_asks.sort(key=lambda x: x[0])  # Sort ascending (lowest ask first)
    
    if yes_bids:
        for yes_price, yes_qty in yes_bids:
            no_ask = 100 - yes_price
            no_asks.append((no_ask, yes_qty))
        no_asks.sort(key=lambda x: x[0])  # Sort ascending (lowest ask first)
    
    # Compute OVER (YES) values
    over_take_odds = None
    over_take_liquidity = None
    over_make_odds = None
    
    if yes_asks:
        # TAKE: best YES ask (lowest ask)
        best_yes_ask_cents, best_yes_ask_qty = yes_asks[0]
        
        # Apply taker fee: effective price = ask_price + taker_fee
        taker_fee_cents = fee_dollars(1, best_yes_ask_cents) * 100  # Convert to cents
        effective_take_price_cents = best_yes_ask_cents + int(taker_fee_cents)
        
        # Convert to American odds
        over_take_odds = cents_to_american(effective_take_price_cents)
        
        # Compute liquidity in dollars: contracts × P × (1 - P)
        P = best_yes_ask_cents / 100.0
        liquidity_dollars = best_yes_ask_qty * P * (1.0 - P)
        over_take_liquidity = liquidity_dollars
    
    if yes_bids:
        # MAKE: best YES bid + 1 tick
        best_yes_bid_cents = yes_bids[-1][0]  # Last element is best bid
        make_price_cents = best_yes_bid_cents + 1
        
        # Validate: must not cross (make_price < best_yes_ask if asks exist)
        if yes_asks and make_price_cents >= yes_asks[0][0]:
            # Would cross, use best bid instead
            make_price_cents = best_yes_bid_cents
        
        # Apply maker fee adjustment
        make_price_adjusted = adjust_maker_price_for_fees(make_price_cents + maker_fee_cents(make_price_cents, 1))
        if make_price_adjusted is not None:
            make_price_cents = make_price_adjusted
        
        # Convert to American odds
        over_make_odds = cents_to_american(make_price_cents)
    
    # Compute UNDER (NO) values
    under_take_odds = None
    under_take_liquidity = None
    under_make_odds = None
    
    if no_asks:
        # TAKE: best NO ask (lowest ask)
        best_no_ask_cents, best_no_ask_qty = no_asks[0]
        
        # Apply taker fee
        taker_fee_cents = fee_dollars(1, best_no_ask_cents) * 100
        effective_take_price_cents = best_no_ask_cents + int(taker_fee_cents)
        
        # Convert to American odds
        under_take_odds = cents_to_american(effective_take_price_cents)
        
        # Compute liquidity in dollars
        P = best_no_ask_cents / 100.0
        liquidity_dollars = best_no_ask_qty * P * (1.0 - P)
        under_take_liquidity = liquidity_dollars
    
    if no_bids:
        # MAKE: best NO bid + 1 tick
        best_no_bid_cents = no_bids[-1][0]
        make_price_cents = best_no_bid_cents + 1
        
        # Validate: must not cross
        if no_asks and make_price_cents >= no_asks[0][0]:
            make_price_cents = best_no_bid_cents
        
        # Apply maker fee adjustment
        make_price_adjusted = adjust_maker_price_for_fees(make_price_cents + maker_fee_cents(make_price_cents, 1))
        if make_price_adjusted is not None:
            make_price_cents = make_price_adjusted
        
        # Convert to American odds
        under_make_odds = cents_to_american(make_price_cents)
    
    # Format output (compact, bettor-native)
    over_take_str = f"TAKE {over_take_odds}" if over_take_odds is not None else "TAKE N/A"
    if over_take_liquidity is not None:
        liquidity_k = int(round(over_take_liquidity / 1000.0))
        over_take_str += f" (${liquidity_k}k)"
    
    over_make_str = f"MAKE {over_make_odds}" if over_make_odds is not None else "MAKE N/A"
    
    under_take_str = f"TAKE {under_take_odds}" if under_take_odds is not None else "TAKE N/A"
    if under_take_liquidity is not None:
        liquidity_k = int(round(under_take_liquidity / 1000.0))
        under_take_str += f" (${liquidity_k}k)"
    
    under_make_str = f"MAKE {under_make_odds}" if under_make_odds is not None else "MAKE N/A"
    
    output = (
        f"O{line}  {over_take_str} | {over_make_str}\n"
        f"U{line}  {under_take_str} | {under_make_str}"
    )
    
    send_response(output, chat_id, cli_mode)


# ============================================================================
# Main Execution Flow
# ============================================================================

def execute_turnin(turnin: Dict[str, Any], chat_id: Optional[str] = None, cli_mode: bool = False) -> None:
    """
    Execute a single turn-in message (moneyline or totals).
    
    Args:
        turnin: Parsed turn-in dictionary
        chat_id: Telegram chat ID (required for Telegram mode)
        cli_mode: If True, print responses to stdout instead of Telegram
    """
    rotation_number = turnin["rotation_number"]
    market_type = turnin.get("market_type", "moneyline")
    execution_mode = turnin.get("execution_mode", "taker")  # Default to taker
    
    # Defensive assertion: make_both must not have juice
    if execution_mode == "make_both":
        assert "juice" not in turnin, "make_both must not include juice field"
        assert turnin.get("execution_mode") == "make_both", "make_both execution mode mismatch"
    
    # Extract juice only for non-make_both paths
    juice = None
    if execution_mode != "make_both":
        juice = turnin.get("juice")
        if juice is None:
            send_response("Missing juice (American odds) in turn-in", chat_id, cli_mode)
            return
    
    amount = turnin["amount"]
    
    # Totals-specific fields
    side = None
    line = None
    if market_type == "totals" and execution_mode != "make_both":
        # For make_both, skip rotation parity validation (no side specified)
        side = turnin.get("side", "").lower()
        line = turnin.get("line")
        
        # Validate rotation parity (not needed for make_both)
        is_valid, error_msg = validate_rotation_parity(rotation_number, side)
        if not is_valid:
            send_response(error_msg, chat_id, cli_mode)
            return
    elif market_type == "totals" and execution_mode == "make_both":
        # For make_both, only need the line
        line = turnin.get("line")
        if line is None:
            send_response("Invalid make both turn-in: missing line", chat_id, cli_mode)
            return
    
    # Calculate budget
    budget_dollars = min(amount * 1000.0, config.MAX_BUDGET_DOLLARS)
    remaining_budget = budget_dollars
    
    # Convert juice to max price limit (only for non-make_both paths)
    max_price_cents = None
    if execution_mode != "make_both":
        max_price_cents = american_to_cents(juice)
    
    if execution_mode == "make_both":
        offset_ticks = turnin.get("offset_ticks")
        print(f"DEBUG: Turn-in parsed - rotation={rotation_number}, market=totals, execution=make_both, line={line}, offset_ticks={offset_ticks}, budget=${budget_dollars}")
    elif market_type == "totals":
        print(f"DEBUG: Turn-in parsed - rotation={rotation_number}, market=totals, side={side}, line={line}, juice={juice}, max_price_cents={max_price_cents}, budget=${budget_dollars}")
    else:
        print(f"DEBUG: Turn-in parsed - rotation={rotation_number}, market=moneyline, juice={juice}, max_price_cents={max_price_cents}, budget=${budget_dollars}")
    
    # Lazy refresh Unabated cache if stale
    maybe_refresh_unabated_cache()
    
    # Extract the global teams lookup map (authoritative source for team names)
    teams_lookup = UNABATED_CACHE["teams"]
    print(f"DEBUG: Loaded {len(teams_lookup)} teams from Unabated teams dictionary")
    
    # Find game by rotation number (uses cached roto_to_game mapping)
    game = find_unabated_game_by_roto(rotation_number)
    if not game:
        send_response(
            f"Rotation number {rotation_number} did not resolve to any game",
            chat_id,
            cli_mode
        )
        return
    print(f"DEBUG: Found Unabated game - eventStart={game.get('eventStart')}")
    
    # Extract game info
    event_start = game.get("eventStart")
    event_teams_raw = game.get("eventTeams", {})
    
    # Normalize eventTeams to a list of dicts
    if isinstance(event_teams_raw, dict):
        event_teams = list(event_teams_raw.values())
    elif isinstance(event_teams_raw, list):
        event_teams = event_teams_raw
    else:
        send_response("Invalid Unabated eventTeams format", chat_id, cli_mode)
        return
    
    if not event_start or len(event_teams) != 2:
        send_response("Could not reliably identify both teams from Unabated data", chat_id, cli_mode)
        return
    
    # ========================================================================
    # CHECKPOINT 1: Resolve both teams from Unabated data
    # ========================================================================
    bet_team_name = None
    opponent_team_name = None
    
    print(f"DEBUG: Processing {len(event_teams)} teams from eventTeams")
    for t in event_teams:
        if not isinstance(t, dict):
            continue
        
        team_id = t.get("id")
        team_rotation = t.get("rotationNumber")
        
        print(f"DEBUG: Team entry - id={team_id}, rotationNumber={team_rotation}")
        
        if team_id is None:
            continue
        
        # Resolve team name via teams lookup dictionary
        team_info = teams_lookup.get(str(team_id), {})
        team_name = team_info.get("name", "").strip()
        
        if not team_name:
            continue
        
        # Check if this team matches the rotation number
        if team_rotation == rotation_number:
            if bet_team_name is not None:
                # Rotation number matches multiple teams (should never happen)
                send_response(
                    f"Rotation number {rotation_number} matched multiple teams (ambiguous)",
                    chat_id,
                    cli_mode
                )
                return
            bet_team_name = team_name
            print(f"DEBUG: Bet team identified - rotation={rotation_number}, name='{team_name}'")
        else:
            # This is the opponent
            if opponent_team_name is not None:
                # More than 2 teams (should never happen)
                send_response(
                    "Could not reliably identify both teams from Unabated data",
                    chat_id,
                    cli_mode
                )
                return
            opponent_team_name = team_name
            print(f"DEBUG: Opponent identified - rotation={team_rotation}, name='{team_name}'")
    
    # Safety check: rotation number must match exactly one team
    if bet_team_name is None:
        send_response(
            f"Rotation number {rotation_number} does not belong to either team",
            chat_id,
            cli_mode
        )
        return
    
    if opponent_team_name is None:
        send_response(
            "Could not reliably identify both teams from Unabated data",
            chat_id,
            cli_mode
        )
        return
    
    # ========================================================================
    # CHECKPOINT 2: Validate xref mappings for both teams
    # ========================================================================
    # Check bet team xref mapping
    bet_team_code = team_to_kalshi_code(LEAGUE, bet_team_name)
    if bet_team_code is None:
        # Team not found in xref
        warning = (
            f"XREF_MISSING: Unabated team \"{bet_team_name}\" not found in {TEAM_XREF_FILE}. "
            f"Rotation {rotation_number}. No bet placed. "
            f"Fix team_xref_cbb.csv mapping for {bet_team_name} -> KalshiCode."
        )
        send_response(warning, chat_id, cli_mode)
        return
    
    if not bet_team_code.strip():
        # Team found but kalshi_code is blank/empty
        warning = (
            f"XREF_BLANK: Unabated team \"{bet_team_name}\" has empty kalshi_code in {TEAM_XREF_FILE}. "
            f"Rotation {rotation_number}. No bet placed. "
            f"Fix team_xref_cbb.csv mapping for {bet_team_name} -> KalshiCode."
        )
        send_response(warning, chat_id, cli_mode)
        return
    
    # Check opponent team xref mapping
    opponent_team_code = team_to_kalshi_code(LEAGUE, opponent_team_name)
    if opponent_team_code is None:
        # Opponent not found in xref
        warning = (
            f"XREF_MISSING: Unabated team \"{opponent_team_name}\" (opponent) not found in {TEAM_XREF_FILE}. "
            f"Rotation {rotation_number}, bet team: {bet_team_name} ({bet_team_code}). No bet placed. "
            f"Fix team_xref_cbb.csv mapping for {opponent_team_name} -> KalshiCode."
        )
        send_response(warning, chat_id, cli_mode)
        return
    
    if not opponent_team_code.strip():
        # Opponent found but kalshi_code is blank/empty
        warning = (
            f"XREF_BLANK: Unabated team \"{opponent_team_name}\" (opponent) has empty kalshi_code in {TEAM_XREF_FILE}. "
            f"Rotation {rotation_number}, bet team: {bet_team_name} ({bet_team_code}). No bet placed. "
            f"Fix team_xref_cbb.csv mapping for {opponent_team_name} -> KalshiCode."
        )
        send_response(warning, chat_id, cli_mode)
        return
    
    print(f"DEBUG: Teams resolved - bet={bet_team_code} ({bet_team_name}), opponent={opponent_team_code} ({opponent_team_name})")
    
    # ========================================================================
    # CHECKPOINT 3: Build canonical key (must succeed after xref validation)
    # ========================================================================
    canonical_key = build_canonical_key(LEAGUE, event_start, bet_team_code, opponent_team_code)
    print(f"DEBUG: Built canonical key: {canonical_key}")
    
    # ========================================================================
    # CHECKPOINT 4: Load Kalshi credentials (required for event/market lookup)
    # ========================================================================
    try:
        api_key_id, private_key_pem = load_creds()
    except FileNotFoundError as e:
        send_response(f"Missing Kalshi credentials: {e}", chat_id, cli_mode)
        return
    
    # Lazy refresh Kalshi events cache if stale (GAME or TOTALS based on market type)
    if market_type == "totals":
        maybe_refresh_kalshi_totals_events_cache(api_key_id, private_key_pem)
    else:
        maybe_refresh_kalshi_events_cache(api_key_id, private_key_pem)
    
    # ========================================================================
    # CHECKPOINT 5: Match Kalshi event using canonical key
    # ========================================================================
    # For totals, use totals events cache; for moneyline, use GAME events cache
    use_totals_cache = (market_type == "totals")
    matched_event = match_kalshi_event(canonical_key, (bet_team_code, opponent_team_code), use_totals_cache=use_totals_cache)
    
    if not matched_event:
        # Build detailed warning with date context
        cache = KALSHI_TOTALS_EVENTS_CACHE if use_totals_cache else KALSHI_EVENTS_CACHE
        candidate_tickers = list(cache['by_key'].keys())
        candidate_count = len(candidate_tickers)
        candidate_samples = candidate_tickers[:5] if candidate_count > 0 else []
        
        # Extract date info for clearer error message
        if USE_PYTZ:
            import pytz
            utc = pytz.UTC
            eastern = pytz.timezone("US/Eastern")
        else:
            utc = ZoneInfo("UTC")
            eastern = ZoneInfo("US/Eastern")
        
        dt_utc = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=utc)
        else:
            dt_utc = dt_utc.astimezone(utc)
        dt_eastern = dt_utc.astimezone(eastern)
        unabated_utc_date = dt_utc.strftime("%Y-%m-%d")
        kalshi_local_date = dt_eastern.strftime("%Y-%m-%d")
        
        event_type = "totals" if use_totals_cache else "game"
        warning = (
            f"NO_KALSHI_EVENT_MATCH: Canonical key mismatch for {event_type} event. "
            f"Rotation {rotation_number}. "
            f"Unabated UTC date: {unabated_utc_date}. "
            f"Kalshi local date expected: {kalshi_local_date}. "
            f"Canonical key: {canonical_key}. "
            f"Teams: {bet_team_name} ({bet_team_code}) vs {opponent_team_name} ({opponent_team_code}). "
            f"Checked {candidate_count} Kalshi {event_type} events. "
            f"Examples: {', '.join(candidate_samples) if candidate_samples else 'none'}. "
            f"No bet placed."
        )
        send_response(warning, chat_id, cli_mode)
        return
    print(f"DEBUG: Matched Kalshi event: {matched_event.get('event_ticker')}")
    
    # ========================================================================
    # CHECKPOINT 6: Find market ticker (moneyline or totals)
    # ========================================================================
    markets = matched_event.get("markets", [])
    print(f"DEBUG: Event has {len(markets)} markets")
    print(f"DEBUG: Market tickers: {[m.get('ticker') for m in markets]}")
    
    selected_market = None
    market_ticker = None
    order_side = "yes"  # Default for moneyline
    
    if market_type == "moneyline":
        # Find market that ends with the bet team code (e.g., -CHSL)
        target_suffix = f"-{bet_team_code}"
        print(f"DEBUG: Looking for moneyline market ending with: {target_suffix}")
        
        for market in markets:
            ticker = market.get("ticker", "")
            print(f"DEBUG: Checking market ticker: {ticker}")
            if ticker.endswith(target_suffix):
                selected_market = market
                print(f"DEBUG: Found moneyline market: {ticker}")
                break
        
        if not selected_market:
            # Build detailed warning with available markets
            available_tickers = [m.get("ticker", "") for m in markets]
            warning = (
                f"XREF_NO_MARKET: Unabated team \"{bet_team_name}\" mapped to \"{bet_team_code}\" "
                f"but no Kalshi market ticker ending in -{bet_team_code} found. "
                f"Rotation {rotation_number}. "
                f"Canonical key: {canonical_key}. "
                f"Teams: {bet_team_name} ({bet_team_code}), {opponent_team_name} ({opponent_team_code}). "
                f"Event: {matched_event.get('event_ticker', 'unknown')}. "
                f"Available markets: {', '.join(available_tickers) if available_tickers else 'none'}. "
                f"No bet placed. Fix team_xref_cbb.csv mapping for {bet_team_name} -> KalshiCode."
            )
            send_response(warning, chat_id, cli_mode)
            return
        
        order_side = "yes"  # Moneyline always buys YES
    
    elif market_type == "totals":
        # Totals markets are in separate events (TOTAL series), not GAME events
        # Fetch all markets for the totals event
        event_ticker = matched_event.get("event_ticker")
        if not event_ticker:
            send_response("Event ticker not found", chat_id, cli_mode)
            return
        
        print(f"DEBUG: Fetching all markets for totals event: {event_ticker}")
        all_markets = fetch_kalshi_markets_for_event(api_key_id, private_key_pem, event_ticker)
        print(f"DEBUG: Found {len(all_markets)} total markets for totals event")
        
        # Find totals market matching line (using int(line) for suffix matching)
        # Kalshi encodes X.5 totals as integers: 141.5 → suffix "-141"
        kalshi_line_suffix = int(line)
        
        if execution_mode == "make_both":
            # For make_both, find market by line only (no side needed)
            target_suffix = f"-{kalshi_line_suffix}"
            selected_market = None
            for market in all_markets:
                ticker = market.get("ticker", "")
                if ticker.endswith(target_suffix):
                    selected_market = market
                    break
            
            if not selected_market:
                available_tickers = [m.get("ticker", "") for m in all_markets[:10]]
                warning = (
                    f"TOTALS_MARKET_NOT_FOUND: No totals market found for "
                    f"rotation {rotation_number}, line={line} (expected suffix: {target_suffix}). "
                    f"Canonical key: {canonical_key}. "
                    f"Teams: {bet_team_name} ({bet_team_code}) vs {opponent_team_name} ({opponent_team_code}). "
                    f"Event: {event_ticker}. "
                    f"Found {len(all_markets)} markets in totals event. "
                    f"Examples: {', '.join(available_tickers) if available_tickers else 'none'}. "
                    f"No orders placed."
                )
                send_response(warning, chat_id, cli_mode)
                return
            
            print(f"DEBUG: Found totals market for make_both: {selected_market.get('ticker')}, line={line}")
        else:
            # Regular totals: find market matching side and line
            print(f"DEBUG: Looking for totals market - side={side}, line={line} (Kalshi suffix: -{kalshi_line_suffix})")
            selected_market = find_totals_market(all_markets, side, line)
            
            if not selected_market:
                # Build detailed warning
                available_tickers = [m.get("ticker", "") for m in all_markets[:10]]  # Show first 10
                warning = (
                    f"TOTALS_MARKET_NOT_FOUND: No totals market found for "
                    f"rotation {rotation_number}, side={side}, line={line} (expected suffix: -{kalshi_line_suffix}). "
                    f"Canonical key: {canonical_key}. "
                    f"Teams: {bet_team_name} ({bet_team_code}) vs {opponent_team_name} ({opponent_team_code}). "
                    f"Event: {event_ticker}. "
                    f"Found {len(all_markets)} markets in totals event. "
                    f"Examples: {', '.join(available_tickers) if available_tickers else 'none'}. "
                    f"No bet placed."
                )
                send_response(warning, chat_id, cli_mode)
                return
            
            # Determine order side: over = buy YES, under = buy NO
            order_side = "yes" if side.lower() == "over" else "no"
            print(f"DEBUG: Found totals market: {selected_market.get('ticker')}, order_side={order_side}")
    
    market_ticker = selected_market.get("ticker")
    if not market_ticker:
        print(f"DEBUG: Market ticker is None")
        send_response("Market ticker not found", chat_id, cli_mode)
        return
    print(f"DEBUG: Using market ticker: {market_ticker}")
    
    # ========================================================================
    # ALL VALIDATIONS PASSED - PROCEED TO ORDER EXECUTION
    # ========================================================================
    # At this point:
    # - Rotation number resolved to exactly one game
    # - Both team IDs resolved to Unabated team names
    # - Both team names exist in xref and map to non-empty kalshi codes
    # - Canonical key built successfully
    # - Kalshi event found for canonical key
    # - Market ticker exists (moneyline or totals)
    # - For totals: rotation parity validated, side/line matched
    # - Maker mode validated (not used with totals)
    # Safe to proceed with order placement.
    
    # Hard-separate execution paths: make_both, maker, taker
    if execution_mode == "make_both":
        # Make both: post passive orders on both sides of totals line
        assert execution_mode == "make_both", "Make both execution path reached with wrong mode"
        assert market_type == "totals", "Make both only supported for totals"
        assert "juice" not in turnin, "make_both must not include juice field"
        
        # Get make_both specific fields
        line = turnin.get("line")
        offset_ticks = turnin.get("offset_ticks")
        
        if line is None or offset_ticks is None:
            send_response("Invalid make both turn-in: missing line or offset_ticks", chat_id, cli_mode)
            return
        
        # Execute make both totals order
        execution_result = execute_make_both_totals(
            api_key_id=api_key_id,
            private_key_pem=private_key_pem,
            market_ticker=market_ticker,
            line=line,
            offset_ticks=offset_ticks,
            budget_dollars=budget_dollars,
            rotation_number=rotation_number,
            chat_id=chat_id,
            cli_mode=cli_mode
        )
        
        if execution_result["status"] == "error":
            return  # Error message already sent
        
        # Build make both confirmation message
        yes_result = execution_result.get("yes", {})
        no_result = execution_result.get("no", {})
        status = execution_result.get("status", "none_posted")
        
        if status == "both_posted":
            budget_per_side = budget_dollars / 2.0
            confirmation = (
                f"POSTED BOTH TOTALS MAKER\n"
                f"OVER {line} @ {yes_result.get('price_cents', 0)}¢ (YES)\n"
                f"UNDER {line} @ {no_result.get('price_cents', 0)}¢ (NO)\n"
                f"Budget per side: ${budget_per_side:.3f}"
            )
        elif status == "one_posted":
            parts = []
            if yes_result.get("posted"):
                parts.append(f"YES @ {yes_result.get('price_cents', 0)}¢ posted")
            else:
                parts.append(f"YES skipped ({yes_result.get('reason', 'unknown')})")
            
            if no_result.get("posted"):
                parts.append(f"NO @ {no_result.get('price_cents', 0)}¢ posted")
            else:
                parts.append(f"NO skipped ({no_result.get('reason', 'unknown')})")
            
            confirmation = f"POSTED MAKE BOTH (PARTIAL)\n" + "\n".join(parts)
        else:
            confirmation = "POSTED MAKE BOTH (FAILED)\nBoth sides rejected"
        
        send_response(confirmation, chat_id, cli_mode)
    
    elif execution_mode == "maker":
        assert execution_mode == "maker", "Maker execution path reached with wrong mode"
        
        if market_type == "moneyline":
            # Execute maker moneyline order
            execution_result = execute_maker_moneyline(
                api_key_id=api_key_id,
                private_key_pem=private_key_pem,
                market_ticker=market_ticker,
                order_side=order_side,
                limit_price_cents=max_price_cents,
                budget_dollars=budget_dollars,
                bet_team_name=bet_team_name,
                bet_team_code=bet_team_code,
                rotation_number=rotation_number,
                chat_id=chat_id,
                cli_mode=cli_mode
            )
            
            if execution_result["status"] == "error":
                return  # Error message already sent
            
            # Build maker confirmation message
            order_id = execution_result.get("order_id", "unknown")
            contracts = execution_result.get("requested_contracts", 0)
            confirmation = (
                f"Posted MAKER order: {bet_team_name or bet_team_code} (R{rotation_number})\n"
                f"Price: {max_price_cents}¢\n"
                f"Contracts: {contracts}\n"
                f"Order ID: {order_id}"
            )
            send_response(confirmation, chat_id, cli_mode)
        
        elif market_type == "totals":
            # Execute maker totals order
            execution_result = execute_maker_totals(
                api_key_id=api_key_id,
                private_key_pem=private_key_pem,
                market_ticker=market_ticker,
                order_side=order_side,
                limit_price_cents=max_price_cents,
                budget_dollars=budget_dollars,
                side=side,
                line=line,
                rotation_number=rotation_number,
                chat_id=chat_id,
                cli_mode=cli_mode
            )
            
            if execution_result["status"] == "error":
                return  # Error message already sent
            
            # Build maker totals confirmation message
            order_id = execution_result.get("order_id", "unknown")
            contracts = execution_result.get("requested_contracts", 0)
            post_price = execution_result.get("post_price_cents", max_price_cents)
            effective_price = execution_result.get("effective_price_cents", max_price_cents)
            confirmation = (
                f"Posted TOTALS MAKER:\n"
                f"{side.upper()} {line} @ {post_price}¢\n"
                f"Contracts: {contracts}\n"
                f"Effective price after fees: {effective_price}¢\n"
                f"Rotation: {rotation_number}\n"
                f"Ticker: {market_ticker}\n"
                f"Order ID: {order_id}"
            )
            send_response(confirmation, chat_id, cli_mode)
        
    else:  # taker mode
        assert execution_mode == "taker", "Taker execution path reached with wrong mode"
        
        # Taker mode: cross the book immediately
        # ========================================================================
        # CHECKPOINT 7: Fetch orderbook and check liquidity (TAKER ONLY)
        # ========================================================================
        orderbook = fetch_orderbook(api_key_id, private_key_pem, market_ticker)
        if not orderbook:
            print(f"DEBUG: Failed to fetch orderbook for {market_ticker}")
            send_response("Failed to fetch orderbook", chat_id, cli_mode)
            return
        
        no_bids = orderbook.get("no") or []  # Normalize None to []
        yes_bids = orderbook.get("yes") or []  # Normalize None to []
        print(f"DEBUG: Orderbook - NO bids: {len(no_bids)}, YES bids: {len(yes_bids)}")
        
        # Check liquidity based on order side
        if order_side == "yes" and not no_bids:
            print(f"DEBUG: No NO bids available (needed for YES asks)")
            send_response("No liquidity at or below max price", chat_id, cli_mode)
            return
        elif order_side == "no" and not yes_bids:
            print(f"DEBUG: No YES bids available (needed for NO asks)")
            send_response("No liquidity at or below max price", chat_id, cli_mode)
            return
        
        # Determine execution prices (taker-style: cross the book)
        execution_prices = determine_execution_prices(orderbook, order_side, max_price_cents)
        
        if not execution_prices:
            send_response("No liquidity at or below max price", chat_id, cli_mode)
            return
        
        # Execute: buy up orderbook (taker-style)
        execution_result = execute_taker_orders(
            api_key_id=api_key_id,
            private_key_pem=private_key_pem,
            market_ticker=market_ticker,
            execution_prices=execution_prices,
            order_side=order_side,
            remaining_budget=remaining_budget,
            chat_id=chat_id,
            cli_mode=cli_mode
        )
        
        if execution_result["status"] == "error":
            return  # Error message already sent
        
        total_contracts = execution_result["filled_contracts"]
        total_spend = execution_result["total_spend"]
        total_fees = execution_result["fees"]
        avg_price_cents = execution_result.get("avg_price_cents", 0)
        
        # Send confirmation
        if total_contracts == 0:
            if execution_result.get("status") == "none":
                send_response("No liquidity at or below max price", chat_id, cli_mode)
            else:
                send_response("No contracts filled", chat_id, cli_mode)
            return
        
        # Build confirmation message based on market type
        if market_type == "moneyline":
            confirmation = (
                f"Filled {bet_team_name or bet_team_code} (R{rotation_number})\n"
                f"Contracts: {total_contracts}\n"
                f"Avg Price: {avg_price_cents:.1f}¢\n"
                f"Total Spend: ${total_spend:.2f}\n"
                f"Fees: ${total_fees:.2f}"
            )
        else:  # totals
            confirmation = (
                f"Filled {side.upper()} {line} (R{rotation_number})\n"
                f"Contracts: {total_contracts}\n"
                f"Avg Price: {avg_price_cents:.1f}¢\n"
                f"Total Spend: ${total_spend:.2f}\n"
                f"Fees: ${total_fees:.2f}"
            )
        
        send_response(confirmation, chat_id, cli_mode)


# ============================================================================
# Initialization
# ============================================================================

def initialize() -> tuple:
    """
    Shared initialization for both Telegram and CLI modes.
    Returns (api_key_id, private_key_pem) for use in execution.
    """
    print("Initializing caches...")
    
    # Load Kalshi credentials for cache initialization
    try:
        api_key_id, private_key_pem = load_creds()
    except FileNotFoundError as e:
        print(f"❌ Missing Kalshi credentials: {e}")
        raise
    
    # Force initial refresh (not lazy, since cache is empty)
    try:
        print("Fetching Unabated snapshot...")
        refresh_unabated_cache()
        print(f"✓ Unabated cache initialized ({len(UNABATED_CACHE['roto_to_game'])} games)")
    except Exception as e:
        print(f"❌ Failed to initialize Unabated cache: {e}")
        raise
    
    try:
        print("Fetching Kalshi GAME events...")
        refresh_kalshi_events_cache(api_key_id, private_key_pem)
        print(f"✓ Kalshi GAME events cache initialized ({len(KALSHI_EVENTS_CACHE['by_key'])} events)")
    except Exception as e:
        print(f"❌ Failed to initialize Kalshi GAME events cache: {e}")
        raise
    
    try:
        print("Fetching Kalshi TOTALS events...")
        refresh_kalshi_totals_events_cache(api_key_id, private_key_pem)
        print(f"✓ Kalshi TOTALS events cache initialized ({len(KALSHI_TOTALS_EVENTS_CACHE['by_key'])} events)")
    except Exception as e:
        print(f"❌ Failed to initialize Kalshi TOTALS events cache: {e}")
        raise
    
    return api_key_id, private_key_pem


# ============================================================================
# Execution Modes
# ============================================================================

def run_telegram():
    """
    Run Telegram polling loop.
    """
    if not config.TELEGRAM_BOT_TOKEN:
        print("❌ Telegram bot token not configured")
        return
    
    print("Starting Telegram polling loop...")
    
    # Send ready message to Telegram
    ready_message = "Bot is ready to receive turn-ins. Caches initialized."
    send_telegram_message(ready_message)
    print("✓ Ready message sent to Telegram")
    
    offset = None
    
    while True:
        try:
            data = poll_updates(config.TELEGRAM_BOT_TOKEN, offset)
            
            if not data.get("ok"):
                print("❌ Telegram API error")
                time.sleep(5)
                continue
            
            for update in data.get("result", []):
                update_id = update.get("update_id")
                offset = update_id + 1
                
                message = update.get("message", {})
                text = message.get("text", "")
                chat_id = str(message.get("chat", {}).get("id"))
                
                if not text or not chat_id:
                    continue
                
                # Check for totals view command first (read-only)
                totals_command = parse_totals_command(text)
                if totals_command:
                    execute_totals_view(totals_command, chat_id, cli_mode=False)
                    continue
                
                # Check for kill command
                kill_command = parse_kill_command(text)
                if kill_command:
                    # Load credentials for kill command
                    try:
                        api_key_id, private_key_pem = load_creds()
                    except FileNotFoundError as e:
                        send_telegram_message(f"Missing Kalshi credentials: {e}", chat_id)
                        continue
                    
                    print(f"Processing kill command: {text}")
                    execute_kill(kill_command, api_key_id, private_key_pem, chat_id, cli_mode=False)
                    continue
                
                # Parse turn-in
                turnin = parse_turnin(text)
                if not turnin:
                    # Check if it looks like a maker turn-in that failed
                    if text.strip().upper().startswith("MAKE "):
                        send_telegram_message(
                            'Invalid maker turn-in.\nExamples:\nmake 892 -900, 0.001\nmake 891 over 141.5 -110, 0.001',
                            chat_id
                        )
                    else:
                        send_telegram_message(
                            'Invalid turn-in. Examples: "892 -900, 0.001" (moneyline taker) or "891 over 141.5 -110, 0.001" (totals taker) or "make 892 -900, 0.001" (maker) or "kill" or "kill 892"',
                            chat_id
                        )
                    continue
                
                # Execute with timing
                print(f"Processing turn-in: {text}")
                start_time = time.time()
                execute_turnin(turnin, chat_id, cli_mode=False)
                elapsed_ms = (time.time() - start_time) * 1000
                print(f"⏱️  Execution time: {elapsed_ms:.0f}ms (from message receipt to order completion)")
        
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            time.sleep(5)


def run_cli():
    """
    Run persistent CLI input loop.
    """
    print("READY – Enter turn-ins or type 'exit'")
    
    while True:
        try:
            line = input("> ").strip()
            
            if line.lower() in ("exit", "quit"):
                print("Shutting down...")
                break
            
            if not line:
                continue
            
            # Check for totals view command first (read-only)
            totals_command = parse_totals_command(line)
            if totals_command:
                execute_totals_view(totals_command, chat_id=None, cli_mode=True)
                print()  # Blank line for readability
                continue
            
            # Check for kill command
            kill_command = parse_kill_command(line)
            if kill_command:
                # Load credentials for kill command
                try:
                    api_key_id, private_key_pem = load_creds()
                except FileNotFoundError as e:
                    print(f"❌ Missing Kalshi credentials: {e}")
                    continue
                
                print(f"Processing kill command: {line}")
                execute_kill(kill_command, api_key_id, private_key_pem, chat_id=None, cli_mode=True)
                print()  # Blank line for readability
                continue
            
            # Parse turn-in
            turnin = parse_turnin(line)
            if not turnin:
                # Check if it looks like a maker turn-in that failed
                if line.strip().upper().startswith("MAKE "):
                    print('❌ Invalid maker turn-in.\nExamples:\nmake 892 -900, 0.001\nmake 891 over 141.5 -110, 0.001')
                else:
                    print('❌ Invalid turn-in. Examples: "892 -900, 0.001" (moneyline taker) or "891 over 141.5 -110, 0.001" (totals taker) or "make 892 -900, 0.001" (maker) or "kill" or "kill 892"')
                continue
            
            # Execute with timing
            print(f"Processing turn-in: {line}")
            start_time = time.time()
            execute_turnin(turnin, chat_id=None, cli_mode=True)
            elapsed_ms = (time.time() - start_time) * 1000
            print(f"⏱️  Execution time: {elapsed_ms:.0f}ms (from input to order completion)")
            print()  # Blank line for readability
        
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# ============================================================================
# Main Loop
# ============================================================================

def main():
    """
    Main entry point. Supports both Telegram and CLI modes.
    """
    parser = argparse.ArgumentParser(description="Kalshi Telegram Turn-In Executor")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode (persistent terminal input instead of Telegram polling)"
    )
    args = parser.parse_args()
    
    # Shared initialization
    try:
        initialize()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Run in selected mode
    if args.cli:
        run_cli()
    else:
        run_telegram()


if __name__ == "__main__":
    main()


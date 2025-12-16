"""
Validation logic: rotation parity, market selection, side validation.
"""

from typing import Optional, Tuple, Dict, Any, List


def validate_rotation_parity(rotation_number: int, side: str) -> Tuple[bool, Optional[str]]:
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

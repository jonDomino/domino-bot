"""
Turn-in message parsing: moneyline, totals, maker, make both formats.
"""

import re
from typing import Optional, Dict, Any


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

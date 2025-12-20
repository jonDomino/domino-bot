"""
Command parsing: kill command, totals view command.
"""

import re
from typing import Optional, Dict, Any


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


def parse_fair_command(msg: str) -> Optional[Dict[str, Any]]:
    """
    Parse a fair command message.
    
    Format: fair {rotation} {side} {fair_line} [{fair_juice}], {budget}
    Examples:
    - "fair 653 over 142.5 100, 0.01"
    - "fair 653 under 142.5, 0.50"
    - "fair 653 over 141.5 -110, 2"
    
    Returns dict with:
    - cmd: "fair"
    - rotation: int
    - side: "over" | "under"
    - fair_line: float
    - fair_juice: int (default = -110 if omitted)
    - budget: float
    Returns None if not a fair command.
    """
    msg = msg.strip()
    msg_lower = msg.lower()
    
    # Check for "fair" command
    if not msg_lower.startswith("fair "):
        return None
    
    remaining = msg[5:].strip()  # Remove "fair " prefix
    
    # Pattern 1: fair {rotation} {side} {fair_line} {fair_juice}, {budget}
    pattern1 = r'^(\d+)\s+(over|under)\s+(\d+\.?\d*)\s+(-?\d+),\s*(\d+\.?\d*)$'
    match1 = re.match(pattern1, remaining, re.IGNORECASE)
    
    if match1:
        rotation = int(match1.group(1))
        side = match1.group(2).lower()
        fair_line = float(match1.group(3))
        fair_juice = int(match1.group(4))
        budget = float(match1.group(5))
        
        return {
            "cmd": "fair",
            "rotation": rotation,
            "side": side,
            "fair_line": fair_line,
            "fair_juice": fair_juice,
            "budget": budget
        }
    
    # Pattern 2: fair {rotation} {side} {fair_line}, {budget} (default fair_juice = -110)
    pattern2 = r'^(\d+)\s+(over|under)\s+(\d+\.?\d*),\s*(\d+\.?\d*)$'
    match2 = re.match(pattern2, remaining, re.IGNORECASE)
    
    if match2:
        rotation = int(match2.group(1))
        side = match2.group(2).lower()
        fair_line = float(match2.group(3))
        budget = float(match2.group(4))
        
        return {
            "cmd": "fair",
            "rotation": rotation,
            "side": side,
            "fair_line": fair_line,
            "fair_juice": -110,  # Default
            "budget": budget
        }
    
    return None


def parse_writeup_command(msg: str) -> Optional[Dict[str, Any]]:
    """
    Parse a writeup command message.
    
    Format: writeup (single word, no arguments)
    
    Returns dict with:
    - command_type: "writeup"
    Returns None if not a writeup command.
    """
    msg = msg.strip()
    msg_lower = msg.lower()
    
    # Check for "writeup" command (exact match, single word)
    if msg_lower == "writeup":
        return {
            "command_type": "writeup"
        }
    
    return None

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

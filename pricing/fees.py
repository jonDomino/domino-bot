"""
Fee calculation utilities: taker fees, maker fees, and price adjustments.
"""

import math
from typing import Optional

from utils import config


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

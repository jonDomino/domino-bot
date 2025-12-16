"""
Price conversion utilities: cents <-> American odds.
"""


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

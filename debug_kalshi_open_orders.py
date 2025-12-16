"""
Temporary debug script to validate Kalshi open orders API.

Calls GET /portfolio/orders?status=resting&limit=200 (paginating if needed)
Prints count and details of each order.
"""

import sys
from utils.kalshi_api import load_creds, make_request


def fetch_resting_orders(api_key_id: str, private_key_pem: str):
    """
    Fetch all resting orders with pagination support.
    """
    path = "/portfolio/orders"
    all_orders = []
    cursor = None
    limit = 200
    
    print(f"Fetching resting orders (status=resting, limit={limit})...")
    
    try:
        while True:
            params = {
                "status": "resting",
                "limit": limit
            }
            if cursor:
                params["cursor"] = cursor
                print(f"  Fetching next page (cursor={cursor[:20]}...)")
            
            resp = make_request(api_key_id, private_key_pem, "GET", path, params)
            
            # Primary response structure: resp["orders"]
            orders = resp.get("orders", [])
            if orders:
                all_orders.extend(orders)
                print(f"  Retrieved {len(orders)} orders (total so far: {len(all_orders)})")
            
            # Check for pagination cursor
            cursor = resp.get("cursor") or resp.get("next_cursor")
            if not cursor:
                break
        
        # Debug: print response keys if structure differs
        if not all_orders and resp:
            print(f"\n⚠️  No orders found. Response keys: {list(resp.keys())}")
            if "orders" not in resp:
                print(f"Full response structure: {resp}")
        
        return all_orders
    except Exception as e:
        print(f"❌ Failed to fetch resting orders: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main entry point."""
    try:
        api_key_id, private_key_pem = load_creds()
    except FileNotFoundError as e:
        print(f"❌ Missing Kalshi credentials: {e}")
        sys.exit(1)
    
    orders = fetch_resting_orders(api_key_id, private_key_pem)
    
    print(f"\n{'='*80}")
    print(f"Total resting orders found: {len(orders)}")
    print(f"{'='*80}\n")
    
    if not orders:
        print("No resting orders found.")
        return
    
    # Print details for each order
    for i, order in enumerate(orders, 1):
        order_id = order.get("order_id") or order.get("id", "unknown")
        ticker = order.get("ticker") or order.get("market_ticker", "unknown")
        side = order.get("side", "unknown")
        yes_price = order.get("yes_price")
        no_price = order.get("no_price")
        price = yes_price or no_price or order.get("price", 0)
        remaining_count = order.get("remaining_count") or order.get("remaining_quantity") or order.get("count", 0)
        status = order.get("status", "unknown")
        
        print(f"[{i}] Order ID: {order_id}")
        print(f"    Ticker: {ticker}")
        print(f"    Side: {side.upper()}")
        print(f"    Price: {price}¢ ({'YES' if yes_price else 'NO' if no_price else 'unknown'})")
        print(f"    Remaining: {remaining_count}")
        print(f"    Status: {status}")
        print()


if __name__ == "__main__":
    main()


## Project Overview

Kalshi turn-in bot for men’s college basketball (CBB). Users send structured “turn-ins” (via Telegram or CLI), the bot resolves the game using Unabated, maps to the correct Kalshi market, fetches live orderbooks, and executes orders (taker or maker) with fee-aware pricing. Totals and moneylines are supported, plus kill commands and a read-only totals view.

### Core Flow
- **Input:** Turn-ins or commands received from Telegram polling loop or interactive CLI.
- **Parse:** Dedicated parsing module recognizes taker, maker, make both (totals), totals view, and kill commands.
- **Resolve game:** Unabated snapshot cache → rotation → teams → canonical key → Kalshi events cache (GAME or TOTALS).
- **Select market:** For moneyline, pick market ending in team code. For totals, find market by line (e.g., 141.5 → suffix -141). Make-both uses the same line lookup.
- **Orderbook:** Always fetch live orderbook before pricing/execution (no orderbook cache).
- **Price & fees:** Helpers convert odds↔cents and compute taker/maker fees. Maker prices can be adjusted down to respect user’s net-of-fees max.
- **Execute:** Branch by execution mode:
  - Taker: buy up book within price/budget.
  - Maker (moneyline/totals): post-only, non-crossing, fee-aware.
  - Make both (totals): post YES/NO with tick offsets from best bids.
- **Kill:** List resting orders (`status=resting`) and cancel by order_id; supports kill-all and kill-by-rotation.
- **Confirm:** Send Telegram or stdout messages with fills/errors.

### Command Reference (user-facing)
- **Moneyline taker:** `892 -900, 0.001`
- **Totals taker:** `891 over 141.5 -110, 0.001`
- **Moneyline maker:** `make 892 -900, 0.001`
- **Totals maker:** `make 891 over 141.5 -110, 0.001`
- **Make both (totals):** `make both 891 141.5, 1 0.01`
- **Totals view (read-only):** `totals 891 141.5`
- **Kill:** `kill` or `kill 891`

Budget is in thousands of dollars (0.001 = $1, 0.5 = $500). Rotation parity: odd→Over, even→Under (except make both).

### Caching Model (singletons)
- `UNABATED_CACHE`: snapshot, teams, roto→game; lazy refresh with TTL.
- `KALSHI_EVENTS_CACHE`: GAME series events by canonical key; lazy refresh with TTL.
- `KALSHI_TOTALS_EVENTS_CACHE`: TOTALS series events by canonical key; lazy refresh with TTL.
- Caches live in one place; imported as `import core.cache as cache`.

### Key Modules (post-Phase 1)
- `pricing/conversion.py`: `cents_to_american`, `american_to_cents`.
- `pricing/fees.py`: taker/maker fee math, maker fee-aware price adjustment, affordability helpers.
- `parsing/turnin.py`: turn-in parsing (taker/maker/make-both).
- `parsing/commands.py`: kill and totals view command parsing.
- `parsing/validators.py`: rotation parity, totals market finder.
- `main.py`: orchestration only (I/O, cache refresh, dispatch, execution).
- `utils/kalshi_api.py`: authenticated Kalshi requests (supports DELETE).
- `utils/telegram_api.py`: send/poll Telegram messages.

### Execution Semantics (high level)
- **Taker:** Crosses the book level-by-level up to user max price and budget; taker fees included in budget.
- **Maker (moneyline):** Post-only; non-crossing vs implied ask from opposite bids; fee-aware price adjustment to honor user max after fees.
- **Maker (totals):** Post-only; non-crossing vs implied ask from opposite bids (not same-side); fee-aware; market invention allowed.
- **Make both (totals):** Tick-offset from best YES/NO bids; independent sides; no odds/juice; fees apply only if filled.
- **Kill:** Fetch `status=resting` orders (paginated), cancel via `DELETE /portfolio/orders/{id}`.

### Safety & Fail-Safe Highlights
- Hard-stop if xref cannot map teams or no matching Kalshi event/market.
- Date alignment: Unabated UTC `eventStart` → US/Eastern for canonical key.
- Non-crossing maker rules use implied asks from opposite-side bids.
- Orderbook always fetched fresh; no orderbook cache.
- Make-both isolated: tick-based, no juice/odds, no shared helpers that assume juice.

### Quick Pointers for a New Engineer
- Start in `main.py` for orchestration and command dispatch.
- Pricing & fees: `pricing/`.
- Parsing & validation: `parsing/`.
- Kalshi auth/signing: `utils/kalshi_api.py`.
- Telegram primitives: `utils/telegram_api.py`.
- Caches: see `core.cache` (singleton objects; import the module, not attributes).


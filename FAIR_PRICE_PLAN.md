## Plan: `fair` Command (Fair-Price Execution)

Goal: Add a `fair` command that consumes profitable resting liquidity (after fees) and, if budget remains, posts resting maker orders using a fair line/juice input and the 1pt = 3% rule to translate line differences into fair probabilities. **Direction is explicit and mandatory; the command only trades the specified side, never the opposite side.**

### Command Contract
- Syntax: `fair {rotation} {side} {fair_line} {fair_juice}, {budget}` OR `fair {rotation} {side} {fair_line}, {budget}`
  - Example: `fair 653 over 142.5 100, 0.01` (explicit juice)
  - Example: `fair 653 over 142.5, 0.01` (default juice = 100, i.e., 50% / 50%)
  - `side`: **mandatory**, must be `over` or `under` (case-insensitive).
  - `fair_juice` in American odds; `100` = 50% fair (default if omitted).
  - `budget` in thousands of dollars (existing convention).
- Scope: Totals markets only (CBB totals series).
- **Directional Constraint (Hard Requirement):**
  - The bot must **ONLY** take taker liquidity and post maker liquidity that results in net exposure in the specified `{side}`.
  - The bot must **NEVER** take or make orders that give exposure to the opposite side.
  - The bot must **NEVER** hedge or "balance" inventory by trading the opposite side.
  - This applies even if opposite-side resting orders appear +EV or both sides appear profitable in isolation.
  - **Direction comes from the command, not from strike location or delta_pts.**
- Behavior: 
  1) Find the nearest Kalshi totals market ticker to `fair_line` (Kalshi strikes are spaced in 3-point increments).
  2) Take any resting orders on the **specified side only** that are +EV after fees (taker economics).
  3) With remaining budget, post resting maker orders on the **specified side only** with timing-aware EV tolerance (maker economics).
  4) Track inventory on the specified side; if exposure cap exceeded, stop quoting (do not quote opposite side).

### Inputs & Normalization
- Parse rotation, **side** (mandatory: "over" or "under"), fair_line (float), fair_juice (optional, default=100), budget (float).
- Convert fair_juice → fair probability `p_fair` via existing `american_to_cents`.
- Locate totals event via existing caches (rotation → Unabated → canonical key → Kalshi totals event).
- **Strike Selection (Multi-Strike Ready):**
  - Structure as list of strikes (initially one, but extensible).
  - Find nearest available Kalshi totals strike:
    - Kalshi totals strikes are integer-coded suffixes (line 141.5 → suffix `-141`).
    - Allowed strikes spaced by 3 points; pick strike minimizing |kalshi_line - fair_line|.
    - Record `line_used` (ticker strike) and `delta_pts = line_used - fair_line` **per strike**.
    - **Important:** The strike is just a proxy instrument. Direction comes from the command, not from delta_pts.
  - Encapsulate "evaluate one strike (take + make)" into reusable function for future multi-strike extension.
  - **Future extension:** Evaluate multiple strikes, but all in the same direction (no balancing across sides).

### Fair Probability Adjustment (1pt = 3% Rule, Direction-Aware)
- Adjust fair probability to the ticker's strike (per-strike, not global):
  - **Only compute probability for the specified side:**
    - If `side == "over"`: `p_over = clamp(p_fair + (-delta_pts) * 0.03, 0.01, 0.99)`
    - If `side == "under"`: `p_under = clamp(p_fair + (delta_pts) * 0.03, 0.01, 0.99)` (note: sign flipped)
  - Intuition: If ticker line is lower than fair, Over becomes more likely; if ticker line is higher than fair, Under becomes more likely.
  - **Do not compute or use the opposite side's probability for decision-making.**
  - The opposite side probability is not needed since we never trade it.

### Economics & Fee Models (Corrected)
- **Critical Correction: Fees are paid only on winning outcomes, not all outcomes.**
- Taker fee: existing `fee_dollars` (0.07 * C * P * (1-P), rounded up).
- Maker fee: existing `maker_fee_cents` (0.0175 * C * P * (1-P), rounded up).
- **EV Calculation (Corrected):**
  - For a contract at price `P` (dollars) with win probability `p_win`:
    - Win payoff: `(1 - P)` dollars
    - Loss: `P` dollars
    - Fee on win: `fee_rate * (1 - P) * P` (taker) or `maker_fee_rate * (1 - P) * P` (maker)
    - **Correct EV formula:**
      - `EV = p_win * ((1 - P) - fee_on_win) - (1 - p_win) * P`
      - Simplified: `EV = p_win * (1 - P) * (1 - fee_rate * P) - (1 - p_win) * P`
  - This must be used for both taker and maker EV calculations internally (not exposed in messages).

### Step 1: Evaluate Resting Liquidity (Taker Path, One-Sided Only)
- Fetch live orderbook for the selected totals market.
- **Encapsulate in reusable function:** `evaluate_strike_taker(market_ticker, side, p_win, budget)`
- **Hard Constraint: Only evaluate orders on the specified side.**
- If `side == "over"` (YES):
  - Orderbook bids: `yes_bids`; implied asks from `no_bids` (ask = 100 - no_bid).
  - For each ask level (price `a`, qty `q`): compute taker EV using corrected formula with `p_over`.
    - Accept level if `EV >= 0` (strict, since we're taking liquidity).
  - Consume in price ascending order until budget exhausted or EV<0.
  - **Ignore all NO-side resting liquidity (even if it appears +EV).**
- If `side == "under"` (NO):
  - Orderbook bids: `no_bids`; implied asks from `yes_bids` (ask = 100 - yes_bid).
  - For each ask level (price `a`, qty `q`): compute taker EV using corrected formula with `p_under`.
    - Accept level if `EV >= 0` (strict, since we're taking liquidity).
  - Consume in price ascending order until budget exhausted or EV<0.
  - **Ignore all YES-side resting liquidity (even if it appears +EV).**
- Track spend+fees vs budget (budget in thousands → dollars).
- **Track inventory:** Record contracts on the specified side only (not net exposure across sides).
- Stop when budget exhausted or no +EV resting levels on the specified side.

### Step 2: Post Maker Orders with Remaining Budget (Timing-Aware, One-Sided Only)
- **Critical Correction: Do NOT require maker EV ≥ 0. Use timing-aware tolerance.**
- **Hard Constraint: Only post maker orders on the specified side.**
- **Timing-Aware EV Tolerance Model:**
  - Allow maker orders with `EV >= -ε`, where `ε` is justified by expected consensus movement.
  - Tie `ε` explicitly to the 3% per point rule:
    - Expected move ≥ 1.0 pt → allow maker EV down to −1.5%
    - Expected move ≥ 1.5 pt → allow maker EV down to −2.5%
    - Expected move ≥ 2.0 pt → allow maker EV down to −3.5%
    - Otherwise → do not quote (too far from fair)
  - **For v1:** Use conservative threshold: `ε = 0.015` (1.5%) if `|delta_pts| >= 1.0`, else require `EV >= 0`.
- **Price Strategy (Shifted Emphasis):**
  - **Do NOT aggressively shade price.** Keep maker prices near the fee-neutral band (typically around -110 after fees).
  - Counterparties will not accept meaningfully worse prices; aggressive shading increases adverse selection.
  - **Express conviction via:**
    - **Order size** (all remaining budget goes to the specified side)
    - **Persistence** (willingness to keep the order posted)
    - **NOT price shading**
- **Maker Price Selection (One Side Only):**
  - If `side == "over"` (YES):
    - Compute max postable maker price where `EV >= -ε` (using corrected EV formula with `p_over`).
    - Apply non-crossing rule vs implied YES ask (from NO bids): `post < implied_yes_ask`.
    - Use existing `adjust_maker_price_for_fees` helper to respect net-of-fees requirement.
  - If `side == "under"` (NO):
    - Compute max postable maker price where `EV >= -ε` (using corrected EV formula with `p_under`).
    - Apply non-crossing rule vs implied NO ask (from YES bids): `post < implied_no_ask`.
    - Use existing `adjust_maker_price_for_fees` helper to respect net-of-fees requirement.
  - **Do not compute or post on the opposite side, even if it appears profitable.**
- **Inventory Gating (Directional Discipline):**
  - Track exposure on the specified side only: `exposure = yes_contracts` (if side=over) or `exposure = no_contracts` (if side=under).
  - If `exposure > threshold` (e.g., 50 contracts for v1):
    - **Stop quoting entirely.**
    - **Do NOT quote the opposite side to reduce risk.**
    - The command is allowed to end with fully directional exposure.
  - This prevents building offsetting positions and preserves directional edge.
- **Size Allocation:**
  - **All remaining budget goes to the specified side** (no splitting across sides).
  - There is no "bias" logic based on delta_pts; direction is explicit from the command.
- **Encapsulate in reusable function:** `evaluate_strike_maker(market_ticker, side, p_win, budget, exposure, delta_pts)`
- Post one maker order on the specified side at the fee-aware `post_price`.
- Quantity: derive from remaining budget using effective cost at post price (maker fees apply on fill).
- If no budget remains or no +EV/-ε levels on the specified side, skip maker posting.

### Inventory Tracking (Directional, Basic v1)
- **Per-Market State (One-Sided):**
  - Track: `{market_ticker: {"side": "over"|"under", "contracts": int, "exposure": int}}`
  - Initialize from existing resting orders (query Kalshi portfolio for this market, filter by side).
  - Update after each taker fill and maker post on the specified side only.
- **Inventory Gating Logic:**
  - Before posting maker orders, check `exposure` on the specified side.
  - If threshold exceeded, stop quoting (do not quote opposite side).
- **Future Extension:** This structure allows for multi-market inventory aggregation and risk limits, but all in the same direction.

### Edge Cases & Safety
- If no totals market found for nearest strike → return error with available strikes sample.
- If p_adj hits clamp (0.01/0.99) → log that a clamp occurred.
- If no +EV resting levels on the specified side and maker price can't meet timing tolerance → return a clear "no profitable trades on {side}" message.
- Respect existing parity validation only for taker/maker single-side commands; `fair` should not rely on parity (it's line-based with explicit direction).
- Do not require opposite-side bids (market invention allowed for maker on the specified side).
- If inventory threshold exceeded → log and stop quoting, do not fail silently or quote opposite side.
- **If opposite-side resting orders appear +EV → ignore them (this is intentional; edge is directional).**

### Outputs / Messaging
- Summarize:
  - Selected strike (`line_used`) and delta vs fair line.
  - **Specified side** and fair probability (adjusted for delta_pts): `p_over` or `p_under` (whichever applies).
  - Resting fills taken: **side only**, price, qty, spend, fees, EV≥0 assertion.
  - Maker posts: **side only**, post price, qty, EV (may be slightly negative if within tolerance), implied ask reference.
  - Inventory: exposure on specified side before/after, gating applied (if any).
  - Budget used vs remaining; final status.
- Errors: missing event, missing market, no profitable levels on specified side, or budget too small for 1 contract.

### Testing Plan (manual)
- Scenario A: Book has profitable resting asks on specified side. Expect taker fills, no maker.
- Scenario B: Book has profitable resting asks on opposite side only. Expect no fills (ignore opposite side).
- Scenario C: No profitable asks on specified side, but bids present → maker posts on specified side only within budget.
- Scenario D: One-sided liquidity only (opposite side) → no maker posts (ignore opposite side).
- Scenario E: Fair line equals strike (delta_pts = 0) → verify direction still comes from command, not inference.
- Scenario F: Fair line far from available strike → verify 1pt=3% adjustment shifts probabilities correctly, timing tolerance applied.
- Scenario G: Inventory threshold exceeded → verify quoting stops (no opposite-side hedging).
- Scenario H: Default fair_juice → verify `fair 653 over 142.5, 0.01` assumes 50% fair.

### Implementation Notes (sequencing)
1) Add parser for `fair` command → returns rotation, **side** (mandatory), fair_line, fair_juice (default=100 if omitted), budget.
2) Add market strike selection helper (nearest 3-point spaced strike) → returns list of strikes (initially one).
3) Add probability adjuster (1pt=3% rule, clamped) → per-strike, returns `p_win` for specified side only, delta_pts.
4) Add **corrected** EV calculator for taker levels with fees (fees only on wins).
5) Add **corrected** EV-capable maker price solver with timing-aware tolerance (EV >= -ε) and implied ask non-crossing.
6) Add inventory tracking structure (one-sided) and gating logic (stop quoting if threshold exceeded).
7) Encapsulate strike evaluation: `evaluate_strike_taker(side, ...)` and `evaluate_strike_maker(side, ...)` (reusable for multi-strike, same direction).
8) Wire into execution dispatcher; reuse existing caches, orderbook fetch, fee helpers.
9) Add user-facing messages; no code changes to unrelated commands.

### Future Extensions (Multi-Strike Ready, Same Direction)
- Structure already supports evaluating multiple strikes.
- Future: Evaluate two nearest strikes, allocate budget across both **in the same direction**.
- Future: Aggregate inventory across related strikes for risk management, but all in the same direction.
- Future: Dynamic ε adjustment based on market volatility or consensus movement models.
- **Do not introduce any logic that balances across strikes or sides.**

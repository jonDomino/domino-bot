# Codebase Modularization Plan

## Executive Summary

The `main.py` file has grown to 3,355+ lines, containing all business logic, API integrations, parsing, execution, and orchestration. This document proposes a modular architecture that separates concerns, improves maintainability, and preserves existing functionality.

**⚠️ This plan has been reviewed and approved with amendments. See "Approved Amendments" section below.**

**Key Principles:**
- Zero breaking changes to external interfaces
- Preserve existing caching architecture
- Maintain performance characteristics
- Clear separation of concerns
- Testable, isolated modules
- **Preserve cache identity across all imports**
- **Hard guardrails for paradigm separation (especially make_both)**
- **Behavioral snapshot tests to catch subtle drift**

---

## Approved Amendments

This plan has been reviewed and the following amendments are **required**:

1. **Defer `core/types.py`** - Do NOT create in Phase 1 or 2. Introduce in Phase 4+ after behavior is stable.
2. **Decompose `pricing/orderbook.py`** - Must be internally decomposed with clear function grouping, even if in one file initially.
3. **Hard assertions in `make_both.py`** - Add explicit guardrails at function entry to prevent paradigm leakage.
4. **Realistic `main.py` target** - <800 lines (not <500). Focus on boundaries, not arbitrary compression.
5. **Cache identity preservation** - Use module import pattern (`import core.cache as cache`), never direct import.
6. **Behavioral snapshot tests** - Add golden output tests to catch subtle behavior drift.

---

## Current State Analysis

### File Structure
```
main.py (3,355 lines)
├── Constants & Configuration (lines 37-44)
├── Caching Infrastructure (lines 46-72)
├── Parsing (lines 75-268)
├── Team Xref (lines 273-310)
├── Unabated Integration (lines 315-396)
├── Date/Time Utilities (lines 401-441)
├── Kalshi Event Management (lines 447-622)
├── Price Conversion (lines 625-690)
├── Fee Calculation (lines 691-775)
├── Orderbook Operations (lines 780-890)
├── Order Placement (lines 973-1027)
├── Order Management (lines 1033-1367)
├── Validation & Market Selection (lines 1373-1441)
├── Execution Functions (lines 1460-2244)
├── Main Orchestration (lines 2491-3119)
└── Entry Points (lines 3124-3355)
```

### Key Dependencies
- `utils/config.py` - Configuration constants
- `utils/kalshi_api.py` - Kalshi API authentication & requests
- `utils/telegram_api.py` - Telegram Bot API
- `team_xref_cbb.csv` - Team name mapping

### Critical Global State
- `UNABATED_CACHE` - Unabated snapshot cache
- `KALSHI_EVENTS_CACHE` - Kalshi GAME events cache
- `KALSHI_TOTALS_EVENTS_CACHE` - Kalshi TOTALS events cache
- `TEAM_XREF` - Team name cross-reference (loaded at startup)

---

## Proposed Architecture

### Directory Structure
```
kalshi/
├── main.py                    # Entry point only (~50 lines)
├── core/
│   ├── __init__.py
│   ├── cache.py              # Cache management & TTL logic
│   ├── config.py             # League constants & TTLs
│   └── types.py              # Type definitions & enums
├── parsing/
│   ├── __init__.py
│   ├── turnin.py             # Turn-in message parsing
│   ├── commands.py            # Kill, totals view commands
│   └── validators.py          # Rotation parity, market validation
├── data/
│   ├── __init__.py
│   ├── team_xref.py          # Team name cross-reference loading
│   ├── unabated.py           # Unabated API integration
│   └── kalshi_events.py       # Kalshi events & markets
├── pricing/
│   ├── __init__.py
│   ├── conversion.py          # Cents <-> American odds
│   ├── fees.py               # Fee calculation (taker & maker)
│   └── orderbook.py          # Orderbook fetching & analysis
├── execution/
│   ├── __init__.py
│   ├── taker.py              # Taker order execution
│   ├── maker_moneyline.py    # Maker moneyline execution
│   ├── maker_totals.py       # Maker totals execution
│   ├── make_both.py          # Make both totals execution
│   └── totals_view.py        # Totals read-only view
├── orders/
│   ├── __init__.py
│   ├── placement.py          # Order placement logic
│   └── management.py         # Kill command, order fetching
└── utils/                     # Existing utils (unchanged)
    ├── config.py
    ├── kalshi_api.py
    └── telegram_api.py
```

---

## Module Breakdown

### 1. `core/` - Core Infrastructure

#### `core/cache.py` (~200 lines)
**Purpose:** Centralized cache management with TTL logic

**⚠️ CRITICAL: Cache Identity Must Be Preserved**

**Problem:**
When moving caches into `core/cache.py`, you must ensure there is exactly one cache instance. All modules must import the same object. You cannot accidentally shadow globals.

**❌ Bad Pattern (Creates Multiple Instances):**
```python
# In data/unabated.py
from core.cache import UNABATED_CACHE
# This creates a local reference that can diverge
```

**✅ Correct Pattern (Single Instance):**
```python
# In data/unabated.py
import core.cache as cache
# Access via cache.UNABATED_CACHE
# All modules share the same object
```

**Exports:**
```python
# Cache structures (module-level, shared identity)
UNABATED_CACHE: Dict
KALSHI_EVENTS_CACHE: Dict
KALSHI_TOTALS_EVENTS_CACHE: Dict

# Cache operations
def refresh_unabated_cache() -> None
def maybe_refresh_unabated_cache() -> None
def refresh_kalshi_events_cache(api_key_id, private_key_pem) -> None
def maybe_refresh_kalshi_events_cache(api_key_id, private_key_pem) -> None
def refresh_kalshi_totals_events_cache(api_key_id, private_key_pem) -> None
def maybe_refresh_kalshi_totals_events_cache(api_key_id, private_key_pem) -> None
```

**Source:** Lines 46-72, 332-396, 505-571

**Rationale:** 
- Single source of truth for cache state
- Encapsulates TTL logic
- Prevents cache state from leaking across modules
- **Preserves cache identity across all imports**

**Testing Requirement:**
After Phase 2, verify:
```python
import core.cache as cache1
import core.cache as cache2
assert cache1.UNABATED_CACHE is cache2.UNABATED_CACHE  # Same object
```

---

#### `core/config.py` (~50 lines)
**Purpose:** League-scoped constants and TTL configuration

**Exports:**
```python
LEAGUE: str
UNABATED_LEAGUE_PREFIX: str
KALSHI_SERIES_TICKER: str
TEAM_XREF_FILE: str
UNABATED_TTL: int
KALSHI_EVENTS_TTL: int
MONTHS: Dict[str, str]  # Date parsing
```

**Source:** Lines 30-44

**Rationale:**
- Centralized configuration
- Easy to change league scope
- Reduces magic strings/numbers

---

#### `core/types.py` (~100 lines) - **DEFERRED TO PHASE 4**
**Purpose:** Type definitions, enums, and shared data structures

**⚠️ IMPORTANT: This module should NOT be created in Phase 1 or 2.**

**Rationale for Deferral:**
- System currently relies on implicit contracts with conditional fields
- Early strict typing risks forcing incorrect normalization
- Many execution paths have conditional fields (e.g., `juice` only for non-make_both)
- Type errors could mask subtle behavior drift
- Better to stabilize behavior first, then add types

**When to Introduce:**
- After Phase 3 (execution functions extracted)
- After behavior is frozen and tested
- Start with docstrings and comments, then add types incrementally

**Future Exports (Phase 4+):**
```python
from typing import TypedDict, Literal

ExecutionMode = Literal["taker", "maker", "make_both"]
MarketType = Literal["moneyline", "totals"]
OrderStatus = Literal["filled", "partial", "none", "posted", "error"]

# Conditional TypedDict variants for different execution modes
# (to be designed after behavior is stable)
```

**Rationale:**
- Type safety (once behavior is stable)
- Documentation via types
- IDE autocomplete support

---

### 2. `parsing/` - Input Parsing & Validation

#### `parsing/turnin.py` (~150 lines)
**Purpose:** Parse turn-in messages (moneyline, totals, maker, make both)

**Exports:**
```python
def parse_turnin(msg: str) -> Optional[TurninParsed]
```

**Source:** Lines 158-268

**Rationale:**
- Single responsibility: input parsing
- Easy to extend with new formats
- Testable in isolation

---

#### `parsing/commands.py` (~100 lines)
**Purpose:** Parse special commands (kill, totals view)

**Exports:**
```python
def parse_kill_command(msg: str) -> Optional[Dict[str, Any]]
def parse_totals_command(msg: str) -> Optional[Dict[str, Any]]
```

**Source:** Lines 79-153

**Rationale:**
- Separates command parsing from turn-in parsing
- Clear extension point for new commands

---

#### `parsing/validators.py` (~100 lines)
**Purpose:** Validation logic (rotation parity, market selection)

**Exports:**
```python
def validate_rotation_parity(rotation_number: int, side: str) -> Tuple[bool, Optional[str]]
def find_totals_market(all_markets: List[Dict], side: str, line: float) -> Optional[Dict]
```

**Source:** Lines 1373-1441

**Rationale:**
- Reusable validation logic
- Testable independently
- Clear error messages

---

### 3. `data/` - Data Access Layer

#### `data/team_xref.py` (~80 lines)
**Purpose:** Team name cross-reference loading and lookup

**Exports:**
```python
def load_team_xref(path: Optional[str] = None) -> Dict[Tuple[str, str], str]
def team_to_kalshi_code(league: str, team_raw: str) -> Optional[str]
```

**Source:** Lines 273-310

**Rationale:**
- Isolates CSV file I/O
- Single source for team mapping logic
- Easy to swap data source (CSV → DB → API)

---

#### `data/unabated.py` (~200 lines)
**Purpose:** Unabated API integration and game resolution

**Exports:**
```python
def fetch_unabated_snapshot() -> Dict[str, Any]
def find_unabated_game_by_roto(roto: int) -> Optional[Dict[str, Any]]
def unabated_event_to_kalshi_date(event_start: str) -> str
def build_canonical_key(league: str, event_start: str, team_a: str, team_b: str) -> str
```

**Source:** Lines 315-441

**Rationale:**
- Encapsulates Unabated API contract
- Handles date conversion logic
- Canonical key generation centralized

**Dependencies:**
- `core.cache` - Uses `UNABATED_CACHE`
- `data.team_xref` - For team name resolution

---

#### `data/kalshi_events.py` (~250 lines)
**Purpose:** Kalshi events and markets fetching/matching

**Exports:**
```python
def parse_kalshi_event_ticker(event_ticker: str) -> Optional[Tuple[str, str]]
def fetch_kalshi_events(api_key_id: str, private_key_pem: str, series_ticker: Optional[str] = None) -> List[Dict[str, Any]]
def fetch_kalshi_markets_for_event(api_key_id: str, private_key_pem: str, event_ticker: str) -> List[Dict[str, Any]]
def match_kalshi_event(canonical_key: str, team_codes: Tuple[str, str], use_totals_cache: bool = False) -> Optional[Dict[str, Any]]
```

**Source:** Lines 447-622, 793-821

**Rationale:**
- Encapsulates Kalshi event/market API
- Handles ticker parsing complexity
- Centralizes event matching logic

**Dependencies:**
- `core.cache` - Uses `KALSHI_EVENTS_CACHE`, `KALSHI_TOTALS_EVENTS_CACHE`
- `utils.kalshi_api` - For authenticated requests

---

### 4. `pricing/` - Pricing & Fee Logic

#### `pricing/conversion.py` (~80 lines)
**Purpose:** Price format conversions

**Exports:**
```python
def cents_to_american(price_cents: int) -> int
def american_to_cents(odds: int) -> int
```

**Source:** Lines 629-690

**Rationale:**
- Pure functions, no side effects
- Highly testable
- Reusable across execution modes

---

#### `pricing/fees.py` (~150 lines)
**Purpose:** Fee calculation (taker and maker)

**Exports:**
```python
def fee_dollars(contracts: int, price_cents: int) -> float
def maker_fee_cents(price_cents: int, contracts: int = 1) -> int
def adjust_maker_price_for_fees(limit_price_cents: int) -> Optional[int]
def level_all_in_cost(contracts: int, price_cents: int) -> float
def max_affordable_contracts(remaining: float, price_cents: int, available: int) -> int
```

**Source:** Lines 691-775

**Rationale:**
- Centralized fee logic
- Critical for correctness
- Easy to update fee formulas

---

#### `pricing/orderbook.py` (~200 lines) - **REQUIRES CAREFUL DECOMPOSITION**
**Purpose:** Orderbook fetching and analysis

**⚠️ CRITICAL: This module must be internally decomposed to prevent "mini-main.py" syndrome.**

**Why This Matters:**
- Orderbook logic is where most bugs live
- Most market-specific assumptions live here
- Most paradigm leakage historically occurs here
- You've already experienced bugs with:
  - Asks being inferred incorrectly
  - Maker assuming opposite liquidity
  - Totals vs moneyline differences

**Required Internal Structure:**
Even if functions stay in one file initially, they must be clearly grouped:

```python
# ============================================================================
# Orderbook Fetching
# ============================================================================
def fetch_orderbook(...) -> Optional[Dict[str, Any]]

# ============================================================================
# Implied Ask Derivation (Pure Functions)
# ============================================================================
def derive_implied_yes_asks(no_bids: List[List[int]]) -> List[Tuple[int, int]]
def derive_implied_no_asks(yes_bids: List[List[int]]) -> List[Tuple[int, int]]

# ============================================================================
# Taker Price Selection
# ============================================================================
def determine_execution_prices(orderbook: Dict, max_price_cents: int, budget_dollars: float) -> List[Tuple[int, int]]

# ============================================================================
# Maker Price Ceiling (Moneyline)
# ============================================================================
def determine_execution_price_maker(orderbook: Dict, side: str, limit_price_cents: int) -> Tuple[Optional[int], Optional[str]]

# ============================================================================
# Maker Price Ceiling (Totals - Same-Side Logic)
# ============================================================================
def determine_execution_price_maker_totals(orderbook: Dict, side: str, limit_price_cents: int) -> Tuple[Optional[int], Optional[str]]
```

**Future Consideration:**
After Phase 3, consider splitting into:
- `pricing/orderbook_fetch.py` - API calls only
- `pricing/orderbook_analysis.py` - Implied asks, price selection
- `pricing/maker_pricing.py` - Maker-specific price validation

**Source:** Lines 780-890, 1751-1825

**Dependencies:**
- `pricing.fees` - For fee-aware pricing
- `utils.kalshi_api` - For authenticated requests

---

### 5. `execution/` - Order Execution Logic

#### `execution/taker.py` (~200 lines)
**Purpose:** Taker order execution (moneyline and totals)

**Exports:**
```python
def execute_taker_orders(
    api_key_id: str,
    private_key_pem: str,
    market_ticker: str,
    order_side: str,
    execution_prices: List[Tuple[int, int]],
    budget_dollars: float,
    chat_id: Optional[str] = None,
    cli_mode: bool = False
) -> ExecutionResult
```

**Source:** Lines 1460-1577

**Rationale:**
- Single responsibility: taker execution
- Reusable for moneyline and totals
- Clear contract via ExecutionResult

**Dependencies:**
- `orders.placement` - For order placement
- `pricing.fees` - For fee calculation

---

#### `execution/maker_moneyline.py` (~200 lines)
**Purpose:** Maker moneyline order execution

**Exports:**
```python
def execute_maker_moneyline(
    api_key_id: str,
    private_key_pem: str,
    market_ticker: str,
    order_side: str,
    limit_price_cents: int,
    budget_dollars: float,
    rotation_number: int,
    chat_id: Optional[str] = None,
    cli_mode: bool = False
) -> ExecutionResult
```

**Source:** Lines 1578-1750

**Rationale:**
- Isolated maker moneyline logic
- Fee-aware pricing
- Price improvement validation

**Dependencies:**
- `pricing.orderbook` - For orderbook analysis
- `pricing.fees` - For fee adjustment
- `orders.placement` - For order placement

---

#### `execution/maker_totals.py` (~200 lines)
**Purpose:** Maker totals order execution

**Exports:**
```python
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
) -> ExecutionResult
```

**Source:** Lines 1827-2009

**Rationale:**
- Isolated maker totals logic
- Same-side bid validation
- Fee-aware pricing

**Dependencies:**
- `pricing.orderbook` - For orderbook analysis
- `pricing.fees` - For fee adjustment
- `orders.placement` - For order placement

---

#### `execution/make_both.py` (~250 lines) - **SPECIAL CASE: REQUIRES HARD GUARDRAILS**
**Purpose:** Make both totals execution (tick-based)

**⚠️ CRITICAL: This execution mode has unique invariants and must be treated as a special case.**

**Unique Invariants:**
- No juice (tick-based, not odds-based)
- No odds conversion
- No reliance on asks (same-side bids only)
- Independent side processing (no netting)
- Allowed to invent markets (no bids = post at default)

**Required Hard Assertions:**
Add explicit guardrails at function entry:

```python
def execute_make_both_totals(...) -> Dict[str, Any]:
    """
    Execute make both totals: post passive maker orders on both sides.
    
    This function is fully decoupled from juice-based logic.
    Pricing is pure cents, derived from orderbook best bids and offset_ticks.
    """
    # Hard guardrails - fail fast if wrong paradigm leaks in
    assert "juice" not in locals(), "make_both must not receive juice parameter"
    assert offset_ticks >= 0, "offset_ticks must be non-negative"
    assert budget_dollars > 0, "budget must be positive"
    
    # Explicit budget split (no implicit assumptions)
    budget_per_side = budget_dollars / 2.0
    assert budget_per_side > 0, "budget per side must be positive"
    
    # ... rest of function
```

**Reuse Policy:**
- Do NOT reuse helper functions written for maker totals unless provably neutral
- Do NOT call functions that expect juice/odds
- Do NOT infer asks from opposite side
- Each side processed independently

**Exports:**
```python
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
) -> Dict[str, Any]
```

**Source:** Lines 2010-2244

**Rationale:**
- Fully decoupled from juice-based logic
- Tick-based pricing paradigm
- Independent side processing
- Hard guardrails prevent paradigm leakage

**Dependencies:**
- `pricing.orderbook` - For orderbook fetching (same-side bids only)
- `orders.placement` - For order placement

---

#### `execution/totals_view.py` (~250 lines)
**Purpose:** Read-only totals orderbook view

**Exports:**
```python
def execute_totals_view(
    totals_command: Dict[str, Any],
    chat_id: Optional[str] = None,
    cli_mode: bool = False
) -> None
```

**Source:** Lines 2245-2486

**Rationale:**
- Read-only diagnostic command
- No order placement
- Formatting logic isolated

**Dependencies:**
- `data.kalshi_events` - For market resolution
- `pricing.orderbook` - For orderbook fetching
- `pricing.conversion` - For odds conversion
- `pricing.fees` - For fee calculation

---

### 6. `orders/` - Order Management

#### `orders/placement.py` (~100 lines)
**Purpose:** Low-level order placement

**Exports:**
```python
def place_kalshi_order(
    api_key_id: str,
    private_key_pem: str,
    market_ticker: str,
    count: int,
    price_cents: int,
    side: str,
    execution_mode: str
) -> Optional[Dict[str, Any]]
```

**Source:** Lines 973-1027

**Rationale:**
- Single point for order placement
- Handles API contract details
- Reusable across execution modes

**Dependencies:**
- `utils.kalshi_api` - For authenticated requests

---

#### `orders/management.py` (~350 lines)
**Purpose:** Order fetching, cancellation, and kill command

**Exports:**
```python
def fetch_open_orders(api_key_id: str, private_key_pem: str) -> List[Dict[str, Any]]
def cancel_order(api_key_id: str, private_key_pem: str, order_id: str) -> bool
def resolve_rotation_from_market_ticker(market_ticker: str, api_key_id: str, private_key_pem: str) -> Optional[int]
def execute_kill(
    kill_command: Dict[str, Any],
    api_key_id: str,
    private_key_pem: str,
    chat_id: Optional[str] = None,
    cli_mode: bool = False
) -> None
```

**Source:** Lines 1033-1367

**Rationale:**
- Centralized order management
- Kill command logic isolated
- Rotation resolution reusable

**Dependencies:**
- `data.kalshi_events` - For rotation resolution
- `utils.kalshi_api` - For authenticated requests

---

### 7. `main.py` - Orchestration & Entry Points

**Purpose:** Thin orchestration layer and entry points

**Structure:**
```python
# Imports from new modules
import core.cache as cache  # ⚠️ Use module import, not direct import
from core import config
from parsing import turnin, commands, validators
from data import team_xref, unabated, kalshi_events
from pricing import conversion, fees, orderbook
from execution import taker, maker_moneyline, maker_totals, make_both, totals_view
from orders import placement, management
from utils import telegram_api

# Helper function for response routing
def send_response(msg: str, chat_id: Optional[str] = None, cli_mode: bool = False) -> None:
    """Route responses to Telegram or CLI"""

# Main orchestration function
def execute_turnin(turnin: Dict[str, Any], chat_id: Optional[str] = None, cli_mode: bool = False) -> None:
    """
    Main execution dispatcher.
    
    Flow:
    1. Validate turn-in
    2. Resolve game via Unabated
    3. Resolve teams via xref
    4. Build canonical key
    5. Match Kalshi event
    6. Select market
    7. Dispatch to execution function
    
    Note: No business logic, no pricing logic, no API calls except orchestration.
    """
    # ... orchestration logic (~300-400 lines)

# Initialization
def initialize() -> Tuple[str, str, Dict]:
    """Load credentials, xref, refresh caches"""

# Entry points
def run_telegram() -> None:
    """Telegram polling loop"""

def run_cli() -> None:
    """CLI interactive loop"""

def main() -> None:
    """Entry point with argparse"""
```

**Source:** Lines 1446-1455, 2491-3119, 3124-3355

**Rationale:**
- Thin orchestration layer
- Clear entry points
- Minimal business logic
- No business logic, pricing logic, or direct API calls

**⚠️ Realistic Line Count:**
- Target: <800 lines (not <500)
- CLI loop: ~100 lines
- Telegram loop: ~100 lines
- Cache refresh orchestration: ~50 lines
- Dispatch logic: ~300-400 lines
- Error routing: ~50 lines
- Helper functions: ~100 lines

**Why <800, not <500:**
- Chasing arbitrary line count risks pushing logic into wrong places
- Orchestration is inherently non-trivial
- Better to have clear boundaries than artificial compression

---

## Migration Strategy

### Phase 1: Extract Pure Functions (Low Risk)
**Goal:** Extract functions with no dependencies on global state

**Modules:**
1. `pricing/conversion.py` - Pure price conversions
2. `pricing/fees.py` - Pure fee calculations
3. `parsing/turnin.py` - Pure parsing logic
4. `parsing/commands.py` - Pure command parsing
5. `parsing/validators.py` - Pure validation

**Steps:**
1. Create new modules
2. Copy functions (no changes)
3. Update imports in `main.py`
4. Test thoroughly
5. Remove old code

**Risk:** Low - Pure functions, easy to test

**⚠️ Note:** Do NOT create `core/types.py` in this phase. Types will be introduced in Phase 4 after behavior is stable.

---

### Phase 2: Extract Data Access (Medium Risk)
**Goal:** Extract data fetching and caching logic

**Modules:**
1. `data/team_xref.py` - Team mapping
2. `data/unabated.py` - Unabated integration
3. `data/kalshi_events.py` - Kalshi events
4. `core/cache.py` - Cache management

**Steps:**
1. Create `core/cache.py` first
2. Move cache structures and TTL logic
3. **CRITICAL: Use module import pattern (`import core.cache as cache`)**
4. Update all cache references to use `cache.UNABATED_CACHE` pattern
5. Extract data modules one at a time
6. Test after each extraction
7. **Verify cache identity: all imports reference same object**

**Risk:** Medium - Global state migration, requires careful testing

**Testing Strategy:**
- Unit tests for cache TTL logic
- Integration tests for cache refresh
- Verify cache state preserved across calls
- **Verify cache identity: `cache1.UNABATED_CACHE is cache2.UNABATED_CACHE`**

---

### Phase 3: Extract Execution Functions (Medium Risk)
**Goal:** Extract execution logic into separate modules

**Modules:**
1. `execution/taker.py`
2. `execution/maker_moneyline.py`
3. `execution/maker_totals.py`
4. `execution/make_both.py` - **SPECIAL: Add hard assertions**
5. `execution/totals_view.py`

**Steps:**
1. Extract one execution function at a time
2. **For `make_both.py`: Add hard assertions at function entry**
3. Update imports in `main.py`
4. Test that execution path end-to-end
5. Remove old code

**Risk:** Medium - Complex dependencies, requires end-to-end testing

**Testing Strategy:**
- Mock dependencies (orderbook, API calls)
- Integration tests with real API (staging)
- Verify execution results match current behavior
- **Behavioral snapshot tests (see Testing Strategy section)**

**⚠️ Special Handling for `make_both.py`:**
- Add assertions: no juice, no odds, offset_ticks >= 0
- Explicit budget split
- Do not reuse maker totals helpers unless provably neutral

---

### Phase 4: Extract Order Management (Low Risk)
**Goal:** Extract order placement and management

**Modules:**
1. `orders/placement.py`
2. `orders/management.py`

**Steps:**
1. Extract order placement first
2. Extract order management
3. Update imports
4. Test kill command end-to-end

**Risk:** Low - Well-isolated functionality

---

### Phase 4: Extract Order Management (Low Risk)
**Goal:** Extract order placement and management

**Modules:**
1. `orders/placement.py`
2. `orders/management.py`

**Steps:**
1. Extract order placement first
2. Extract order management
3. Update imports
4. Test kill command end-to-end

**Risk:** Low - Well-isolated functionality

---

### Phase 5: Refactor Orchestration (Low Risk)
**Goal:** Clean up `main.py` to be thin orchestration layer

**Steps:**
1. Move helper functions to appropriate modules
2. Simplify `execute_turnin()` to pure orchestration
3. Clean up entry points
4. **Add docstrings and comments (types deferred)**

**Risk:** Low - Mostly reorganization

**⚠️ Note:** Do NOT add strict typing in this phase. Types will be introduced incrementally after behavior is frozen.

---

## Testing Strategy

### Unit Tests
**Location:** `tests/unit/`

**Coverage:**
- All parsing functions
- Price conversions
- Fee calculations
- Validation logic
- Cache TTL logic

**Framework:** `pytest`

**Example:**
```python
# tests/unit/test_pricing_conversion.py
def test_cents_to_american_favorite():
    assert cents_to_american(60) == -150

def test_cents_to_american_underdog():
    assert cents_to_american(40) == +150
```

---

### Integration Tests
**Location:** `tests/integration/`

**Coverage:**
- End-to-end execution paths
- Cache refresh behavior
- Order placement (with mocks)
- Market resolution

**Framework:** `pytest` with fixtures

**Example:**
```python
# tests/integration/test_execution_taker.py
def test_taker_moneyline_execution(mock_orderbook, mock_place_order):
    result = execute_taker_orders(...)
    assert result["status"] == "filled"
```

---

### Behavioral Snapshot Tests (CRITICAL - Previously Missing)
**Location:** `tests/snapshots/`

**Purpose:** Catch subtle behavior drift that doesn't throw errors but produces different outputs

**Why This Matters:**
- Modularization bugs often don't throw errors
- They just place slightly different orders
- Or produce slightly different output
- These are the most dangerous bugs in trading systems

**Coverage:**
- Known turn-in scenarios with "golden" expected output
- Order price selection verification
- Number of orders sent verification
- Printed output format verification

**Framework:** `pytest` with snapshot comparison

**Examples:**
```python
# tests/snapshots/test_totals_view.py
def test_totals_view_output_snapshot(mock_orderbook):
    """Verify totals view produces exact expected output"""
    result = execute_totals_view({"rotation_number": 891, "line": 141.5})
    assert result == """
O141.5  TAKE -105 ($12k) | MAKE +100
U141.5  TAKE -110 ($21k) | MAKE -105
"""

# tests/snapshots/test_make_both.py
def test_make_both_order_selection(mock_orderbook):
    """Verify make both selects correct prices and places exactly 2 orders"""
    result = execute_make_both_totals(..., offset_ticks=1, budget_dollars=0.01)
    assert result["status"] == "both_posted"
    assert result["yes"]["price_cents"] == 50  # Exact price
    assert result["no"]["price_cents"] == 47    # Exact price
    # Verify exactly 2 orders placed (not 1, not 3)

# tests/snapshots/test_taker_execution.py
def test_taker_price_levels(mock_orderbook):
    """Verify taker selects correct price levels in correct order"""
    result = execute_taker_orders(..., max_price_cents=55)
    # Verify prices are ascending
    # Verify all prices <= 55
    # Verify correct number of orders
```

**Golden Output Files:**
Store expected outputs in `tests/snapshots/golden/`:
- `totals_891_141.5.txt`
- `make_both_891_141.5_1_0.01.json`
- `taker_moneyline_892_-900_0.001.json`

**Update Process:**
- When behavior intentionally changes, update golden files
- Review all snapshot changes in PR
- Never auto-update snapshots

---

### Manual Testing Checklist
After each phase:
- [ ] Telegram turn-in (moneyline taker)
- [ ] Telegram turn-in (totals taker)
- [ ] Telegram turn-in (maker moneyline)
- [ ] Telegram turn-in (maker totals)
- [ ] Telegram turn-in (make both)
- [ ] CLI turn-in (all modes)
- [ ] Kill command (all, roto-specific)
- [ ] Totals view command
- [ ] Cache refresh behavior
- [ ] Error handling (invalid turn-ins)

---

## Benefits

### Maintainability
- **Single Responsibility:** Each module has one clear purpose
- **Easier Navigation:** Find code by domain (parsing, execution, pricing)
- **Reduced Cognitive Load:** Smaller files, focused concerns

### Testability
- **Isolated Units:** Pure functions easy to test
- **Mockable Dependencies:** Clear interfaces for mocking
- **Test Coverage:** Can test modules independently

### Extensibility
- **New Execution Modes:** Add new file in `execution/`
- **New Commands:** Add parser in `parsing/commands.py`
- **New Data Sources:** Swap implementations in `data/`

### Performance
- **No Runtime Overhead:** Pure reorganization
- **Same Caching:** Cache architecture preserved
- **Same Execution:** Business logic unchanged

---

## Risks & Mitigations

### Risk 1: Breaking Changes
**Mitigation:**
- Zero changes to function signatures
- Preserve all existing behavior
- Comprehensive testing before removal

### Risk 2: Import Cycles
**Mitigation:**
- Clear dependency hierarchy
- `core/` has no dependencies
- `parsing/` depends only on `core/`
- `data/` depends on `core/` and `utils/`
- `pricing/` depends on `core/` and `utils/`
- `execution/` depends on all above
- `orders/` depends on `data/` and `utils/`

### Risk 3: Cache State Issues
**Mitigation:**
- Centralize cache in `core/cache.py`
- **Use module import pattern: `import core.cache as cache`**
- **Never use direct import: `from core.cache import UNABATED_CACHE`**
- Explicit cache access functions
- No direct cache mutation outside module
- **Verify cache identity in tests: `cache1.UNABATED_CACHE is cache2.UNABATED_CACHE`**

### Risk 4: Behavioral Drift (Subtle Bugs)
**Mitigation:**
- Behavioral snapshot tests for all execution paths
- Golden output files for known scenarios
- Never auto-update snapshots
- Review all snapshot changes in PR
- Manual testing checklist after each phase

### Risk 5: Migration Complexity
**Mitigation:**
- Phased approach (5 phases)
- Test after each phase
- Keep old code until new code verified
- Rollback plan for each phase

### Risk 6: Paradigm Leakage (make_both)
**Mitigation:**
- Hard assertions in `make_both.py` function entry
- Explicit budget split
- Do not reuse maker totals helpers unless provably neutral
- Separate internal logic from other execution modes

---

## Timeline Estimate

**Phase 1:** 2-3 days (pure functions)
**Phase 2:** 3-4 days (data access + cache)
**Phase 3:** 4-5 days (execution functions)
**Phase 4:** 2-3 days (order management)
**Phase 5:** 2-3 days (orchestration cleanup)

**Total:** 13-18 days

**Buffer:** +5 days for testing and bug fixes

**Recommended:** 3-week sprint with daily checkpoints

---

## Success Criteria

1. ✅ `main.py` reduced to <800 lines (realistic target, not <500)
2. ✅ `main.py` contains no business logic, pricing logic, or direct API calls (except orchestration)
3. ✅ All existing functionality preserved
4. ✅ No performance regression
5. ✅ All tests passing (unit, integration, behavioral snapshots)
6. ✅ Code review approved
7. ✅ Manual testing checklist complete
8. ✅ Cache identity verified (all imports reference same objects)
9. ✅ `make_both.py` has hard assertions preventing paradigm leakage
10. ✅ `pricing/orderbook.py` is internally decomposed with clear boundaries

---

## Post-Migration Improvements

Once modularized, consider:

1. **Type Hints:** Add comprehensive type hints throughout
2. **Documentation:** Add module-level docstrings
3. **Logging:** Structured logging with module names
4. **Error Handling:** Custom exception classes
5. **Configuration:** Environment-based config (dev/staging/prod)
6. **Monitoring:** Add metrics/telemetry hooks

---

## Conclusion

This modularization plan preserves all existing functionality while dramatically improving code organization. The phased approach minimizes risk and allows for incremental validation. The resulting architecture will be more maintainable, testable, and extensible.

**Next Steps:**
1. Review and approve this plan
2. Create module directory structure
3. Begin Phase 1 (pure functions)
4. Set up testing infrastructure
5. Execute migration phases sequentially


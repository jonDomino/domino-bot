# Kalshi Turn-In Executor

A Python trading bot that executes Kalshi trades based on structured "turn-in" messages sent via Telegram or CLI.

## Features

- **Multiple Execution Modes:**
  - Taker orders (moneyline & totals)
  - Maker orders (moneyline & totals)
  - Make both (tick-based totals market making)
  - Totals view (read-only orderbook summary)

- **Commands:**
  - Turn-in execution (various formats)
  - Kill command (cancel resting orders)
  - Totals view command (read-only)

- **Performance Optimizations:**
  - Lazy-refresh caching for Unabated snapshots
  - Cached Kalshi events (GAME and TOTALS series)
  - Precomputed rotation-to-game mappings

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure credentials:**
   - Copy `utils/config.py.example` to `utils/config.py`
   - Fill in your Telegram bot token and chat ID
   - Fill in your Unabated API key
   - Create `kalshi_api_key_id.txt` with your Kalshi API key ID
   - Create `kalshi_private_key.pem` with your Kalshi private key

3. **Configure team mapping:**
   - Ensure `team_xref_cbb.csv` is populated with team name mappings

## Usage

### Telegram Mode
```bash
python main.py
```

### CLI Mode
```bash
python main.py --cli
```

## Turn-In Formats

### Taker Orders
- Moneyline: `892 -900, 0.001`
- Totals: `891 over 141.5 -110, 0.001`

### Maker Orders
- Moneyline: `make 892 -900, 0.001`
- Totals: `make 891 over 141.5 -110, 0.001`

### Make Both
- Totals: `make both 891 141.5, 1 0.01`

### Commands
- Kill all: `kill`
- Kill by rotation: `kill 892`
- Totals view: `totals 891 141.5`

## Architecture

See [MODULARIZATION_PLAN.md](MODULARIZATION_PLAN.md) for detailed architecture documentation and planned refactoring.

## License

Private repository - All rights reserved.


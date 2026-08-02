"""One-shot dust purge for the Alpaca paper account."""
import os
import cvar_trader as ct
from alpaca.trading.client import TradingClient

c = TradingClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], paper=True)

print("Before:")
for p in sorted(c.get_all_positions(), key=lambda x: -abs(float(x.market_value))):
    print(f"  {p.symbol:6} qty={float(p.qty):.12f} mv=${float(p.market_value):.4f}")

# Keep only meaningful positions (mv >= $1); purge everything else as dust.
keep = [p.symbol for p in c.get_all_positions() if abs(float(p.market_value)) >= 1.0]
print("Keeping:", keep)
ct.purge_unexpected_or_dust_positions(keep)

print("\nAfter:")
acct = c.get_account()
print(f"Equity={acct.equity} Cash={acct.cash}")
for p in sorted(c.get_all_positions(), key=lambda x: -abs(float(x.market_value))):
    print(f"  {p.symbol:6} qty={float(p.qty):.12f} mv=${float(p.market_value):.4f}")

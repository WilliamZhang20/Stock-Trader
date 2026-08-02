import argparse
import os

from alpaca.trading.client import TradingClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Close tiny dust positions in an Alpaca paper account.")
    parser.add_argument("--threshold", type=float, default=1.0, help="Close positions with market value below this amount.")
    args = parser.parse_args()

    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    client = TradingClient(key, secret, paper=True)

    positions = list(client.get_all_positions())
    dust_positions = [p for p in positions if 0.0 < float(p.market_value) < args.threshold]

    if not dust_positions:
        print(f"No dust positions below ${args.threshold:.2f} were found.")
        return

    print(f"Closing {len(dust_positions)} dust position(s) below ${args.threshold:.2f}...")
    for position in dust_positions:
        symbol = position.symbol
        market_value = float(position.market_value)
        qty = float(position.qty)
        client.close_position(symbol)
        print(f"  closed {symbol:6} qty={qty:.9f} market_value=${market_value:.6f}")

    remaining = list(client.get_all_positions())
    print("\nRemaining positions:")
    if not remaining:
        print("  None")
        return

    for position in remaining:
        print(f"  {position.symbol:6} qty={float(position.qty):.9f} market_value=${float(position.market_value):.2f}")


if __name__ == "__main__":
    main()
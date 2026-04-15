from pathlib import Path
from datetime import datetime, time
from zoneinfo import ZoneInfo
import os

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

load_dotenv()

api_key = os.getenv("APCA_API_KEY_ID")
api_secret = os.getenv("APCA_API_SECRET_KEY")

if not api_key or not api_secret:
    raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY in .env")

client = StockHistoricalDataClient(api_key, api_secret)

symbol = "AAPL"
ny_tz = ZoneInfo("America/New_York")

start = datetime(2026, 3, 16, 9, 30, tzinfo=ny_tz)
end = datetime(2026, 4, 14, 16, 0, tzinfo=ny_tz)

request = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=TimeFrame.Minute,
    start=start,
    end=end,
    adjustment=Adjustment.RAW,
    feed=DataFeed.SIP,
)

bars = client.get_stock_bars(request).df.reset_index()

bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(ny_tz)

bars = bars[
    (bars["timestamp"].dt.time >= time(9, 30)) &
    (bars["timestamp"].dt.time < time(16, 0))
].copy()

bars = bars.sort_values("timestamp")
bars = bars.drop_duplicates(subset=["timestamp"])

bars = bars[
    ["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
]

raw_dir = Path("data/raw/alpaca/minute_bars")
raw_dir.mkdir(parents=True, exist_ok=True)

out_path = raw_dir / f"{symbol}.parquet"
bars.to_parquet(out_path, index=False)

bars["date"] = bars["timestamp"].dt.date
daily_counts = bars.groupby("date").size().reset_index(name="bar_count")

print(bars.head())
print()
print(f"saved to: {out_path}")
print(f"total rows: {len(bars)}")
print(f"days: {bars['date'].nunique()}")
print()
print(daily_counts.head(10))
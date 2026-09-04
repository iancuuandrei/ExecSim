import os
from datetime import UTC, datetime, timedelta

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["APCA_API_KEY_ID"]
api_secret = os.environ["APCA_API_SECRET_KEY"]

client = StockHistoricalDataClient(api_key, api_secret)

end_utc = datetime.now(UTC) - timedelta(minutes=20)
start_utc = end_utc - timedelta(days=3)

request = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame.Minute,
    start=start_utc,
    end=end_utc,
    adjustment=Adjustment.RAW,
    feed=DataFeed.SIP,
)

bars = client.get_stock_bars(request).df

print(bars.head())
print()
print(f"rows: {len(bars)}")
print(f"columns: {list(bars.columns)}")

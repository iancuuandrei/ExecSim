from datetime import datetime, timedelta, timezone
import os

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

load_dotenv()

api_key = os.environ["APCA_API_KEY_ID"]
api_secret = os.environ["APCA_API_SECRET_KEY"]

client = StockHistoricalDataClient(api_key, api_secret)

end_utc = datetime.now(timezone.utc) - timedelta(minutes=20)
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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

app = FastAPI()

class CurrencyRequest(BaseModel):
    currency: str  # Currency pair to analyze, e.g. 'USD/EUR'

def fetch_forex_data(currency):
    """
    Fetches historical forex data for the given currency pair from Yahoo Finance.
    Converts currency format to Yahoo Finance style (e.g. USD/EUR -> USDEUR=X).
    """
    try:
        # Convert the currency pair to the Yahoo Finance symbol format (e.g. USD/EUR -> USDEUR=X)
        formatted_currency = currency.replace("/", "") + "=X"
        
        # Use Yahoo Finance's API to download historical forex data
        data = yf.download(formatted_currency, period="1mo", interval="1d")  # 1 month of daily data
        if data.empty:
            return {"error": f"No data found for currency pair {currency}"}
        
        # Extract the 'Close' prices as the prices for ARIMA
        prices = data["Close"].dropna().tolist()
        return prices
    except Exception as e:
        return {"error": str(e)}

def analyze_prices_with_arima(prices, currency):
    """
    Analyzes price data using ARIMA and returns forecasted data and sentiment.
    """
    data = pd.Series(prices)
    model = ARIMA(data, order=(1, 1, 1))  # You can adjust the ARIMA order parameters (p, d, q)
    model_fit = model.fit()
    forecast_steps = 10
    forecast = model_fit.forecast(steps=forecast_steps)
    predicted_price = forecast[0]
    daily_high = max(data)
    daily_low = min(data)

    sentiment = "Neutral"
    if predicted_price > daily_high:
        sentiment = "Bullish"
    elif predicted_price < daily_low:
        sentiment = "Bearish"

    return {
        "currency": currency,
        "predicted_price": predicted_price,
        "daily_high": daily_high,
        "daily_low": daily_low,
        "sentiment": sentiment
    }

@app.post("/analyze")
async def analyze_currency(request: CurrencyRequest):
    """
    Endpoint to analyze a single currency pair.
    """
    currency = request.currency
    if not currency:
        raise HTTPException(status_code=400, detail="Currency pair cannot be empty.")
    
    # Fetch forex data from Yahoo Finance
    prices = fetch_forex_data(currency)
    if isinstance(prices, list):  # Check if valid prices were returned
        analysis = analyze_prices_with_arima(prices, currency)
        return analysis
    else:
        return prices  # Return error message if prices retrieval failed

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Optional
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from forex_python.converter import CurrencyCodes

currency = CurrencyCodes()
app = FastAPI()

class CurrencyRequest(BaseModel):
    currency: str  # Currency pair to analyze, e.g. 'USD/EUR'




def fetch_forex_data(currency: str):
    """
    Fetches historical forex data for the given currency pair from Yahoo Finance.
    Converts currency format to Yahoo Finance style (e.g. USD/EUR -> USDEUR=X).
    """

    try:
            # Convert the currency pair to the Yahoo Finance symbol format
        formatted_currency = currency.replace("/", "") + "=X"
        
        # Use Yahoo Finance's API to download historical forex data
        data = yf.download(formatted_currency, period="1mo", interval="1d")
        if data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for currency pair {currency}"
            )
        
        # Extract the 'Close' prices
        prices = data['Close'].dropna().astype(float).to_dict()

        price_list = pd.DataFrame(prices).iloc[:,0].to_list()
        
        if not prices:
            raise HTTPException(
                status_code=404,
                detail="No valid price data available"
            )
        
        return price_list

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )



def analyze_prices_with_arima(prices: list, currency: str):
    """
    Analyzes price data using ARIMA and returns forecasted data and sentiment.
    """
    try:
        data = pd.Series(prices)
        model = ARIMA(data, order=(1, 1, 1))
        model_fit = model.fit()
        forecast_steps = 10
        forecast = model_fit.forecast(steps=forecast_steps).reset_index()
        predicted_price = float(forecast.iloc[0,1])  # Convert numpy float to Python float
        daily_high = float(max(data))
        daily_low = float(min(data))

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
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/analyze")
async def analyze_currency(request: CurrencyRequest):
    """
    Analyzes a currency pair and returns price predictions and sentiment.
    """
    prices = fetch_forex_data(request.currency)
    return analyze_prices_with_arima(prices, request.currency)
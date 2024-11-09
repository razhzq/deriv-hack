from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import websockets
import json
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()

class SymbolsRequest(BaseModel):
    symbols: list[str]  # List of symbols to analyze

async def request_ticks_history(symbol):
    """
    Builds the ticks_history request message for the given symbol.
    """
    
    return json.dumps({
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": 10,  # Number of ticks to retrieve
        "end": "latest",
        "start": 1,
        "style": "ticks"
    })

async def connect_to_websocket(symbols):
    """
    Connects to the WebSocket and processes tick history for each symbol.
    """
    
    app_id = "16929"  # Replace with your actual app_id if available
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"  # WebSocket URI with the app_id
    results = []

    try:
        # Establish a connection to the WebSocket server
        async with websockets.connect(uri) as websocket:
            print("\n[open] Connection established")
            print("\nRequesting historical tick data")

            for symbol in symbols:
                request_message = await request_ticks_history(symbol)
                await websocket.send(request_message)
                response = await websocket.recv()
                data = json.loads(response)


            if 'history' in data and 'prices' in data['history']:
                prices = [float(price) for price in data['history']['prices']]
                analysis = analyze_prices_with_arima(prices, symbol)
                results.append(analysis)
            else:
                results.append({"symbol": symbol, "error": "Unexpected data structure"})

    except websockets.ConnectionClosedError:
        results.append({"error": "Connection closed unexpectedly"})
    except Exception as e:
        results.append({"error": str(e)})

    return results


def analyze_prices_with_arima(prices, symbol):
    """
    Analyzes price data using ARIMA and returns forecasted data and sentiment.
    """

    # Step 1: Prepare the data
    data = pd.Series(prices)

    # Step 2: Fit the ARIMA model
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()

    # Step 3: Forecast future price
    forecast_steps = 10
    forecast = model_fit.forecast(steps=forecast_steps)

    # Print all forecast values
    print(f"Forecasted Prices: {forecast}")

    # Access the first forecasted value
    predicted_price = forecast.iloc[0]

    # Step 4: Calculate the daily high and low prices
    daily_high = max(data)
    daily_low = min(data)

    # # Step 5: Determine market sentiment
    sentiment = "Neutral"
    if predicted_price > daily_high:
        sentiment = "Bullish"
    elif predicted_price < daily_low:
        sentiment = "Bearish"

    # Step 6: Output results
    return {
        "symbol": symbol,
        "predicted_price": predicted_price,
        "daily_high": daily_high,
        "daily_low": daily_low,
        "sentiment": sentiment
    }


@app.post("/analyze")
async def analyze_symbols(request: SymbolsRequest):
    """
    Endpoint to analyze multiple symbols.
    """
    symbols = request.symbols
    if not symbols:
        raise HTTPException(status_code=400, detail="Symbol list cannot be empty.")
    
    results = await connect_to_websocket(symbols)
    return {"results": results}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import re
from enum import Enum
import statistics
from transformers import pipeline
import websockets
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from forex_python.converter import CurrencyCodes

currency = CurrencyCodes()

app = FastAPI()

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class NewsItem(BaseModel):
    text: str
    sentiment: SentimentType
    sentiment_score: float

class NewsResponse(BaseModel):
    company_name: str
    news_items: List[NewsItem]
    total_news: int
    overall_sentiment: SentimentType
    average_sentiment_score: float
    risk_assessment: str

class CurrencyRequest(BaseModel):
    currency: str  # Currency pair to analyze, e.g., 'USD/EUR'
    duration: str  # Duration for prediction: '1d', '1wk', '1mo', '1yr'
     
# Initialize sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

def analyze_sentiment(text: str) -> tuple[SentimentType, float]:
    """
    Analyze the sentiment of a given text using the sentiment analyzer.
    Returns sentiment type and score.
    """
    try:
        result = sentiment_analyzer(text)[0]
        label = result['label'].lower()
        score = result['score']

        if label == 'pos':
            return SentimentType.POSITIVE, score
        elif label == 'neg':
            return SentimentType.NEGATIVE, score
        else:
            return SentimentType.NEUTRAL, score
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return SentimentType.NEUTRAL, 0.5

def assess_risk(sentiment_scores: List[float], overall_sentiment: SentimentType) -> str:
    """
    Assess investment risk based on sentiment analysis.
    """
    avg_score = statistics.mean(sentiment_scores)
    volatility = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0

    if overall_sentiment == SentimentType.POSITIVE and avg_score > 0.7:
        risk = "Low risk - Strong positive sentiment indicates favorable market conditions"
    elif overall_sentiment == SentimentType.POSITIVE and avg_score > 0.5:
        risk = "Moderate-low risk - Positive but mixed sentiment suggests cautious optimism"
    elif overall_sentiment == SentimentType.NEGATIVE and avg_score < 0.3:
        risk = "High risk - Strong negative sentiment indicates unfavorable conditions"
    elif overall_sentiment == SentimentType.NEGATIVE:
        risk = "Moderate-high risk - Generally negative sentiment suggests caution"
    else:
        risk = "Moderate risk - Mixed or neutral sentiment indicates unclear market direction"

    if volatility > 0.2:
        risk += ". High sentiment volatility suggests increased uncertainty"

    return risk

def google_query(search_term: str) -> str:
    search_term = search_term + ' currency'
    url = f"https://news.google.com/search?q={search_term}&hl=en-US&gl=US&ceid=US%3Aen"
    return re.sub(r"\s", "+", url)
   
# Encode special characters in a text string
def encode_special_characters(text):
    encoded_text = ''
    special_characters = {'&': '%26', '=': '%3D', '+': '%2B', ' ': '%20'}  # Add more special characters as needed
    for char in text.lower():
        encoded_text += special_characters.get(char, char)
    return encoded_text


async def fetch_data_from_websocket(custom_request: dict):
    app_id = 16929  # Replace with your app_id
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("[open] Connection established")

            # Send the custom JSON request to the WebSocket server
            await websocket.send(json.dumps(custom_request))
            print(f"[send] Sent: {custom_request}")

            # Receive the response from the WebSocket server
            response = await websocket.recv()
            print(f"[recv] Received: {response}")
            return json.loads(response)  # Convert response to JSON format

    except websockets.ConnectionClosedError as e:
        print(f"[close] Connection closed, code={e.code}, reason={e.reason}")
        raise HTTPException(status_code=503, detail="WebSocket connection closed unexpectedly")
    except Exception as e:
        print(f"[error] {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching data from WebSocket")

def get_stock_news(currency: str, limit: int = 4):
    """
    Fetch and analyze recent news about stocks for a given company
    
    Parameters:
    - company_name: Name of the company to fetch news for
    - limit: Maximum number of news items to return (default: 4)
    
    Returns:
    - JSON object containing company name, news items with sentiment analysis, and risk assessment
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
        }

        currency_full = get_currency_pair_full_name(currency)

        query_url = google_query(currency_full)
        response = requests.get(query_url, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to fetch news from source"
            )

        soup = BeautifulSoup(response.text, "html.parser")
        raw_news = []

        articles = soup.find_all('article')
        links = [article.find('a')['href'] for article in articles]
        links = [link.replace("./articles/", "https://news.google.com/articles/") for link in links]

        news_text = [article.get_text(separator='\n') for article in articles]
        news_text_split = [text.split('\n') for text in news_text]

        titles = [text[2] for text in news_text_split]

        raw_news = titles[:limit] if len(titles) > limit else titles
        
        # Analyze sentiment for each news item
        analyzed_news = []
        sentiment_scores = []
        
        for news_text in raw_news:
            sentiment_type, score = analyze_sentiment(news_text)
            analyzed_news.append(NewsItem(
                text=news_text,
                sentiment=sentiment_type,
                sentiment_score=score
            ))
            sentiment_scores.append(score)

        # Calculate overall sentiment
        avg_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0.5
        if avg_sentiment > 0.6:
            overall_sentiment = SentimentType.POSITIVE
        elif avg_sentiment < 0.4:
            overall_sentiment = SentimentType.NEGATIVE
        else:
            overall_sentiment = SentimentType.NEUTRAL

        # Generate risk assessment
        risk_assessment = assess_risk(sentiment_scores, overall_sentiment)

        return NewsResponse(
            company_name=currency,
            news_items=analyzed_news,
            total_news=len(analyzed_news),
            overall_sentiment=overall_sentiment,
            average_sentiment_score=avg_sentiment,
            risk_assessment=risk_assessment
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error fetching news: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    

def fetch_forex_data(currency, period="1mo", interval="1d"):
    """
    Fetches historical forex data for the given currency pair from Yahoo Finance.
    Converts currency format to Yahoo Finance style (e.g. USD/EUR -> USDEUR=X).
    """

    try:
        # Convert the currency pair to the Yahoo Finance symbol format
        formatted_currency = currency.replace("/", "") + "=X"
        
        # Use Yahoo Finance's API to download historical forex data
        data = yf.download(formatted_currency, period=period, interval=interval)
        if data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for currency pair {currency}"
            )
        
        # Extract the 'Close' prices
        prices = data['Close'].dropna().astype(float).to_dict()

        price_list = pd.DataFrame(prices).iloc[:,0].to_list()

        print(price_list)
        
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


def analyze_prices_with_arima(prices, currency, forecast_steps):
    """
    Analyzes price data using ARIMA and returns forecasted data and sentiment.
    """
    try:
        data = pd.Series(prices)
        model = ARIMA(data, order=(1, 1, 1))
        model_fit = model.fit()
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
    
# Function to convert a currency pair to its full names
def get_currency_pair_full_name(currency_pair: str) -> str:
    """
    Convert a currency pair code (e.g., 'AUD/JPY') to full names (e.g., 'AUD (Australian Dollar) / JPY (Japanese Yen)').

    Parameters:
    - currency_pair: str, the currency pair code in the format 'CURRENCY1/CURRENCY2'

    Returns:
    - str, formatted string with full names of both currencies
    """
    try:
        # Split the currency pair into two codes
        currency1, currency2 = currency_pair.split('/')

        # Get the full names for each currency code
        currency1_name = currency.get_currency_name(currency1) or "Unknown currency"
        currency2_name = currency.get_currency_name(currency2) or "Unknown currency"

        # Format and return the result
        return f"{currency1} ({currency1_name}) / {currency2} ({currency2_name})"
    
    except ValueError:
        return "Invalid currency pair format. Please use 'CURRENCY1/CURRENCY2' format."
    

@app.post("/analyze")
async def analyze_currency(request: CurrencyRequest):
    """
    Endpoint to analyze a currency pair with specified prediction duration.
    """
    currency = request.currency
    duration = request.duration.lower()

    # Set forecast steps based on duration
    forecast_steps = {
        "1d": 1,      # Predict the next day
        "1wk": 7,     # Predict the next 7 days
        "1mo": 30,    # Predict the next 30 days
        "1yr": 365    # Predict the next 365 days
    }.get(duration)

    if not forecast_steps:
        raise HTTPException(status_code=400, detail="Invalid duration. Choose from '1d', '1wk', '1mo', or '1yr'.")

    # Set period for historical data based on duration
    period = {
        "1d": "1mo",  # For 1-day prediction, use 1 month of data
        "1wk": "3mo", # For 1-week prediction, use 3 months of data
        "1mo": "6mo", # For 1-month prediction, use 6 months of data
        "1yr": "2y"   # For 1-year prediction, use 2 years of data
    }[duration]

    # Fetch historical forex data
    prices = fetch_forex_data(currency, period=period)
    if isinstance(prices, list):  # Check if valid prices were returned
        analysis = analyze_prices_with_arima(prices, currency, forecast_steps)

        sentiment_news = get_stock_news(request.currency)

        analysis.update(sentiment_news)

        return analysis
    else:
        return prices  # Return error message if prices retrieval failed


@app.get("/")
async def root():
    return {"message": "Welcome to Deriv Hack API"}
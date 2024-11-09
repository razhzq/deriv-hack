from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import re
from enum import Enum
import statistics
from transformers import pipeline
import websockets
from fastapi import Request
import json
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
    if "news" not in search_term:
        search_term = search_term + " stock news"
    url = f"https://www.google.com/search?q={search_term}&cr=countryIN"
    return re.sub(r"\s", "+", url)

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

@app.get("/stock-news/{company_name}", response_model=NewsResponse)
async def get_stock_news(company_name: str, limit: int = 4):
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

        query_url = google_query(company_name)
        response = requests.get(query_url, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to fetch news from source"
            )

        soup = BeautifulSoup(response.text, "html.parser")
        raw_news = []

        for div_class in ["n0jPhd ynAwRc tNxQIb nDgy9d", "IJl0Z"]:
            raw_news.extend([n.text for n in soup.find_all("div", div_class)])

        raw_news = raw_news[:limit] if len(raw_news) > limit else raw_news
        
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
            company_name=company_name,
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
    

@app.post("/fetch-deriv-data")
async def fetch_ticks_history(request: Request):
    # Define your custom JSON request here (or pass as request body for flexibility)
    custom_request = await request.json()

    # Call the WebSocket connection function
    response = await fetch_data_from_websocket(custom_request)
    
    return response

@app.get("/")
async def root():
    return {"message": "Welcome to Deriv Hack API"}
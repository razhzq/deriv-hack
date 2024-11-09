from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from typing import List
import time
from urllib.parse import quote_plus
import re

app = FastAPI()

class NewsResponse(BaseModel):
    company_name: str
    news_items: List[str]
    total_news: int


def google_query(search_term):
    if "news" not in search_term:
        search_term=search_term+" stock news"
    url=f"https://www.google.com/search?q={search_term}&cr=countryIN"
    url=re.sub(r"\s","+",url)
    return url


@app.get("/stock-news/{company_name}", response_model=NewsResponse)
async def get_stock_news(company_name: str, limit: int = 4):
    """
    Fetch recent news about stocks for a given company
    
    Parameters:
    - company_name: Name of the company to fetch news for
    - limit: Maximum number of news items to return (default: 4)
    
    Returns:
    - JSON object containing company name, news items, and total count
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
        }

        # Generate and send request
        query_url = google_query(company_name)
        response = requests.get(query_url, headers=headers)
        
        # Check if request was successful
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to fetch news from source"
            )

        # Parse HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        news = []

        # Extract news from different div classes
        for div_class in ["n0jPhd ynAwRc tNxQIb nDgy9d", "IJl0Z"]:
            news.extend([n.text for n in soup.find_all("div", div_class)])

        # Limit the number of news items
        news = news[:limit] if len(news) > limit else news

        # Prepare response
        return NewsResponse(
            company_name=company_name,
            news_items=news,
            total_news=len(news)
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


@app.get("/")
async def root():
    return {"message": "Hello World"}
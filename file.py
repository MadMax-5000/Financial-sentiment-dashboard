from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import tweepy
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging
import json
import time
import threading
import sqlite3
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finsentiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Custom insertion method for SQLite to prevent duplicates
def insert_or_ignore_sqlite(table, conn, keys, data_iter):
    sql = f"INSERT OR IGNORE INTO {table} ({','.join(keys)}) VALUES ({','.join(['?']*len(keys))})"
    conn.executemany(sql, data_iter)

# Database setup
def setup_database():
    conn = sqlite3.connect('finsentiment.db')
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS news_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        source TEXT,
        url TEXT UNIQUE,
        content TEXT,
        date_published DATETIME,
        sentiment_score REAL,
        sentiment_label TEXT,
        ticker TEXT,
        date_collected DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS social_media_posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        platform TEXT,
        content TEXT,
        post_id TEXT UNIQUE,
        author TEXT,
        date_posted DATETIME,
        sentiment_score REAL,
        sentiment_label TEXT,
        ticker TEXT,
        date_collected DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        date DATETIME,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        UNIQUE(ticker, date)
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database setup complete")

# Sentiment Analysis Class
class SentimentAnalyzer:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text):
        if not text or pd.isna(text):
            return {"compound": 0, "pos": 0, "neu": 0, "neg": 0, "label": "neutral"}
        
        # VADER sentiment
        sentiment = self.sid.polarity_scores(text)
        
        # Determine label
        if sentiment['compound'] >= 0.05:
            sentiment['label'] = 'positive'
        elif sentiment['compound'] <= -0.05:
            sentiment['label'] = 'negative'
        else:
            sentiment['label'] = 'neutral'
            
        return sentiment
    
    def analyze_dataframe(self, df, text_column):
        results = []
        for text in df[text_column]:
            results.append(self.analyze_text(text))
        
        # Add sentiment columns to dataframe
        df['sentiment_score'] = [r['compound'] for r in results]
        df['sentiment_pos'] = [r['pos'] for r in results]
        df['sentiment_neu'] = [r['neu'] for r in results]
        df['sentiment_neg'] = [r['neg'] for r in results]
        df['sentiment_label'] = [r['label'] for r in results]
        
        return df

# News Scraper Class
class NewsDataCollector:
    def __init__(self, api_key=None):
        self.api_key = api_key
        
    def get_financial_news(self, ticker, days_back=7):
        """Collect financial news from various sources"""
        df_combined = pd.DataFrame()
        
        # Try multiple sources and combine results
        sources = [self._get_news_finviz, self._get_news_yahoo, self._get_news_marketwatch]
        
        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            futures = [executor.submit(source, ticker, days_back) for source in sources]
            for future in futures:
                try:
                    df = future.result()
                    if not df.empty:
                        df_combined = pd.concat([df_combined, df], ignore_index=True)
                except Exception as e:
                    logger.error(f"Error collecting news: {str(e)}")
        
        # Remove duplicates based on title or content similarity
        if not df_combined.empty:
            df_combined = df_combined.drop_duplicates(subset=['title'], keep='first')
            
        return df_combined
    
    def _get_news_finviz(self, ticker, days_back=7):
        """Scrape financial news from Finviz"""
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_table = soup.find(id='news-table')
            if news_table is None:
                return pd.DataFrame()
                
            df_list = []
            
            for row in news_table.find_all('tr'):
                title = row.a.text
                source = row.span.text
                
                # Extract datetime
                date_data = row.td.text.strip().split(' ')
                if len(date_data) == 1:
                    time = date_data[0]
                    date = None  # Will use the last available date
                else:
                    date = date_data[0]
                    time = date_data[1]
                
                url = row.a['href']
                
                df_list.append({
                    'title': title,
                    'source': source.strip(),
                    'date': date,
                    'time': time,
                    'url': url,
                    'ticker': ticker
                })
            
            df = pd.DataFrame(df_list)
            
            # Fill missing dates with the last available date
            last_date = None
            for i, row in df.iterrows():
                if row['date'] is not None:
                    last_date = row['date']
                else:
                    df.at[i, 'date'] = last_date
            
            # Convert date and time to datetime
            df['date_published'] = pd.to_datetime(
                df['date'] + ' ' + df['time'],
                format='%b-%d-%y %I:%M%p',
                errors='coerce'
)
            
            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df = df[df['date_published'] >= cutoff_date]
            
            # Keep only relevant columns
            df = df[['title', 'source', 'url', 'date_published', 'ticker']]
            df['content'] = ''  # Initialize content column
            
            return df
            
        except Exception as e:
            logger.error(f"Error scraping Finviz news: {str(e)}")
            return pd.DataFrame()
    
    def _get_news_yahoo(self, ticker, days_back=7):
        """Get financial news from Yahoo Finance"""
        try:
            ticker_data = yf.Ticker(ticker)
            news = ticker_data.news
            
            if not news:
                return pd.DataFrame()
                
            df_list = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for article in news:
                # Convert timestamp to datetime
                date_published = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                
                if date_published >= cutoff_date:
                    df_list.append({
                        'title': article.get('title', ''),
                        'source': article.get('publisher', ''),
                        'url': article.get('link', ''),
                        'date_published': date_published,
                        'content': article.get('summary', ''),
                        'ticker': ticker
                    })
            
            df = pd.DataFrame(df_list)
            return df
            
        except Exception as e:
            logger.error(f"Error getting Yahoo Finance news: {str(e)}")
            return pd.DataFrame()
    
    def _get_news_marketwatch(self, ticker, days_back=7):
        """Scrape financial news from MarketWatch"""
        try:
            url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_items = soup.select('.article__content')
            if not news_items:
                return pd.DataFrame()
                
            df_list = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for item in news_items:
                title_elem = item.select_one('.article__headline')
                if not title_elem or not title_elem.a:
                    continue
                    
                title = title_elem.a.text.strip()
                url = title_elem.a['href']
                
                # Try to extract date
                date_elem = item.select_one('.article__timestamp')
                if date_elem:
                    date_str = date_elem.text.strip()
                    try:
                        # Handle different date formats
                        if 'ago' in date_str:
                            # Approximate date for relative timestamps
                            if 'hour' in date_str or 'min' in date_str:
                                date_published = datetime.now()
                            elif 'day' in date_str:
                                days = int(re.search(r'(\d+)', date_str).group(1))
                                date_published = datetime.now() - timedelta(days=days)
                            else:
                                date_published = datetime.now() - timedelta(days=1)
                        else:
                            date_published = pd.to_datetime(date_str, errors='coerce')
                    except:
                        date_published = datetime.now()
                else:
                    date_published = datetime.now()
                
                if date_published >= cutoff_date:
                    df_list.append({
                        'title': title,
                        'source': 'MarketWatch',
                        'url': url,
                        'date_published': date_published,
                        'content': '',
                        'ticker': ticker
                    })
            
            df = pd.DataFrame(df_list)
            return df
            
        except Exception as e:
            logger.error(f"Error scraping MarketWatch news: {str(e)}")
            return pd.DataFrame()
    
    def fetch_article_content(self, df):
        """Fetch content for articles where content is empty"""
        def fetch_content(url):
            try:
                #Add https:// if missing and handle relative URLs
                if not url.startswith(('http://', 'https://')):
                    if url.startswith('/'):
                        url = 'https://finviz.com' + url  
                    else:
                        url = 'https://' + url
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract main content - this is a simplistic approach
                content = ""
                
                # Look for common content containers
                content_containers = soup.select('.article-content, .article__body, .article-body, .story-body, .post-content, .entry-content')
                
                if content_containers:
                    for container in content_containers:
                        paragraphs = container.select('p')
                        content += ' '.join([p.text.strip() for p in paragraphs])
                
                # If no content found, try to get at least some text
                if not content:
                    # Get all paragraphs
                    paragraphs = soup.select('p')
                    content = ' '.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 100])
                
                return content
            except Exception as e:
                logger.error(f"Error fetching article content: {str(e)}")
                return ""
        
        # Only fetch content for rows where content is empty and we have a URL
        for i, row in df.iterrows():
            if not row['content'] and row['url']:
                df.at[i, 'content'] = fetch_content(row['url'])
                # Be nice to the servers
                time.sleep(1)
        
        return df
    
    def save_to_database(self, df):
        if df.empty:
            return
        
        conn = sqlite3.connect('finsentiment.db')
    
        required_cols = ['title', 'source', 'url', 'content', 'date_published', 
                    'sentiment_score', 'sentiment_label', 'ticker']
    
        for col in required_cols:
            if col not in df.columns:
                if col in ['sentiment_score', 'sentiment_label']:
                    df[col] = None
                else:
                    df[col] = ''
    
        df['date_published'] = df['date_published'].astype(str)
    
        try:
            df[required_cols].to_sql('news_articles', conn, if_exists='append', index=False, 
                              method='multi', chunksize=500)
            conn.commit()
            logger.info(f"Saved {len(df)} news articles to database")
        except sqlite3.IntegrityError as e:
            logger.warning(f"Duplicate entries found: {e}")
        except Exception as e:
            logger.error(f"Error saving news articles: {e}")
        finally:
            conn.close()

# Social Media Collector
class SocialMediaCollector:
    def __init__(self, twitter_api_key=None, twitter_api_secret=None, 
                twitter_access_token=None, twitter_access_secret=None):
        self.twitter_credentials = {
            'api_key': os.getenv('TWITTER_API_KEY'),
            'api_secret': os.getenv('TWITTER_API_SECRET'),
            'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
            'access_secret': os.getenv('TWITTER_ACCESS_SECRET')
        }
        self.twitter_api = None
        
        # Initialize Twitter API if credentials are provided
        if all(self.twitter_credentials.values()):
            self._init_twitter_api()
    
    def _init_twitter_api(self):
        """Initialize Twitter API with credentials"""
        try:
            auth = tweepy.OAuthHandler(
                self.twitter_credentials['api_key'], 
                self.twitter_credentials['api_secret']
            )
            auth.set_access_token(
                self.twitter_credentials['access_token'], 
                self.twitter_credentials['access_secret']
            )
            self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
            logger.info("Twitter API initialized")
        except Exception as e:
            logger.error(f"Error initializing Twitter API: {str(e)}")
            self.twitter_api = None
    
    def get_twitter_data(self, ticker, days_back=7, max_tweets=100):
        """Get Twitter data for a given ticker"""
        if not self.twitter_api:
            logger.warning("Twitter API not initialized, using simulated data")
            return self._get_simulated_twitter_data(ticker, days_back, max_tweets)
            
        try:
            # Search terms - both ticker and company name if available
            search_query = f"${ticker} OR #{ticker}"
            
            # Get tweets
            tweets = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for tweet in tweepy.Cursor(self.twitter_api.search_tweets, 
                                       q=search_query, 
                                       lang="en", 
                                       tweet_mode="extended").items(max_tweets):
                
                if tweet.created_at >= cutoff_date:
                    tweets.append({
                        'platform': 'Twitter',
                        'content': tweet.full_text,
                        'post_id': tweet.id_str,
                        'author': tweet.user.screen_name,
                        'date_posted': tweet.created_at,
                        'ticker': ticker
                    })
            
            df = pd.DataFrame(tweets)
            return df
            
        except Exception as e:
            logger.error(f"Error getting Twitter data: {str(e)}")
            return self._get_simulated_twitter_data(ticker, days_back, max_tweets)
    
    def _get_simulated_twitter_data(self, ticker, days_back=7, max_tweets=100):
        """Generate simulated Twitter data when API is not available"""
        sentiments = ['positive', 'negative', 'neutral']
        sentiment_weights = [0.4, 0.3, 0.3]  # 40% positive, 30% negative, 30% neutral
        
        positive_templates = [
            f"Bullish on ${ticker}! Their latest earnings report shows strong growth potential.",
            f"Just bought more ${ticker} shares. The company's fundamentals look solid.",
            f"${ticker} is outperforming expectations. Great long-term investment.",
            f"The new product line from ${ticker} is impressive. Stock will go up!",
            f"Analyst upgrades for ${ticker} today! 🚀📈",
            f"${ticker} crushing the competition. Strong buy recommendation."
        ]
        
        negative_templates = [
            f"${ticker} disappointing results this quarter. Might be time to sell.",
            f"Concerned about ${ticker}'s debt levels. Proceed with caution.",
            f"${ticker} facing increased competition. Not looking good.",
            f"Regulatory issues might impact ${ticker} in the coming months. 📉",
            f"Shorting ${ticker} after their latest announcement.",
            f"${ticker} management seems lost. No clear strategy."
        ]
        
        neutral_templates = [
            f"Waiting for more data before making a decision on ${ticker}.",
            f"${ticker} seems fairly valued at current price levels.",
            f"Mixed signals from ${ticker}'s latest report.",
            f"Holding my ${ticker} position for now. Let's see what happens.",
            f"Anyone have insights on ${ticker}'s expansion plans?",
            f"Interesting developments at ${ticker}, but not enough to change my position."
        ]
        
        templates = {
            'positive': positive_templates,
            'negative': negative_templates,
            'neutral': neutral_templates
        }
        
        # Generate simulated tweets
        tweets = []
        for _ in range(max_tweets):
            # Select sentiment
            sentiment = np.random.choice(sentiments, p=sentiment_weights)
            
            # Select template and add some randomness
            content = np.random.choice(templates[sentiment])
            
            # Add some variability with hashtags
            hashtags = ['#investing', '#stocks', '#finance', '#trading', 
                       '#market', '#WallStreet', '#investing', '#StockMarket']
            content += f" {np.random.choice(hashtags)}"
            
            # Random date within days_back
            days_ago = np.random.uniform(0, days_back)
            date_posted = datetime.now() - timedelta(days=days_ago)
            
            tweets.append({
                'platform': 'Twitter',
                'content': content,
                'post_id': f"sim_{int(time.time())}_{np.random.randint(10000, 99999)}",
                'author': f"trader_{np.random.randint(100, 999)}",
                'date_posted': date_posted,
                'ticker': ticker
            })
        
        df = pd.DataFrame(tweets)
        return df
    
    def get_reddit_data(self, ticker, days_back=7, max_posts=100):
        """Get Reddit data for a given ticker"""
        # Since we're not using Reddit API, we'll use simulated data
        return self._get_simulated_reddit_data(ticker, days_back, max_posts)
    
    def _get_simulated_reddit_data(self, ticker, days_back=7, max_posts=100):
        """Generate simulated Reddit data"""
        subreddits = ['wallstreetbets', 'investing', 'stocks', f'{ticker}', 'finance']
        
        positive_templates = [
            f"DD: Why {ticker} is poised for a breakout in the next quarter",
            f"Just YOLOed my life savings into {ticker} calls",
            f"The bull case for {ticker} that no one is talking about",
            f"{ticker} fundamentals analysis - Strong BUY signal",
            f"How {ticker} is disrupting the industry - Long-term hold",
            f"{ticker} technical analysis shows golden cross pattern"
        ]
        
        negative_templates = [
            f"Why I'm shorting {ticker} - Bear case analysis",
            f"Red flags in {ticker}'s latest 10-K that everyone missed",
            f"{ticker} puts are printing today! 🐻",
            f"Concerning trends in {ticker}'s cash flow statement",
            f"Is {ticker} overvalued? My detailed analysis says YES",
            f"{ticker} might not survive the upcoming recession"
        ]
        
        neutral_templates = [
            f"{ticker} Earnings Thread - Q2 2023",
            f"Can someone explain what's happening with {ticker}?",
            f"Should I buy {ticker} at current price? Looking for opinions",
            f"Historical performance of {ticker} during market downturns",
            f"Comparing {ticker} with its competitors - Comprehensive analysis",
            f"What's a fair valuation for {ticker}? Let's discuss"
        ]
        
        templates = {
            'positive': positive_templates,
            'negative': negative_templates,
            'neutral': neutral_templates
        }
        
        sentiments = ['positive', 'negative', 'neutral']
        sentiment_weights = [0.35, 0.35, 0.3]  # 35% positive, 35% negative, 30% neutral
        
        # Generate simulated Reddit posts
        posts = []
        for _ in range(max_posts):
            # Select sentiment and subreddit
            sentiment = np.random.choice(sentiments, p=sentiment_weights)
            subreddit = np.random.choice(subreddits)
            
            # Select template
            title = np.random.choice(templates[sentiment])
            
            # Generate content based on title
            content = f"Title: {title}\n\n"
            
            if "DD" in title or "analysis" in title:
                content += f"""I've been researching {ticker} for the past few weeks and wanted to share my findings.

Key points:
1. Revenue growth: {'impressive' if sentiment == 'positive' else 'concerning' if sentiment == 'negative' else 'steady'}
2. Profit margins: {'expanding' if sentiment == 'positive' else 'shrinking' if sentiment == 'negative' else 'stable'}
3. Debt levels: {'manageable' if sentiment == 'positive' else 'dangerous' if sentiment == 'negative' else 'typical for the industry'}
4. Market share: {'growing' if sentiment == 'positive' else 'eroding' if sentiment == 'negative' else 'holding steady'}

My price target: ${np.random.randint(50, 500)}
Timeframe: {np.random.choice(['3 months', '6 months', '1 year', '2 years'])}

What are your thoughts?"""
            else:
                paragraphs = np.random.randint(2, 5)
                content += "\n\n".join([f"This is paragraph {i+1} about {ticker}. " * np.random.randint(1, 3) for i in range(paragraphs)])
            
            # Random date within days_back
            days_ago = np.random.uniform(0, days_back)
            date_posted = datetime.now() - timedelta(days=days_ago)
            
            posts.append({
                'platform': f'Reddit/r/{subreddit}',
                'content': content,
                'post_id': f"sim_reddit_{int(time.time())}_{np.random.randint(10000, 99999)}",
                'author': f"u/{np.random.choice(['Deep', 'Value', 'Stock', 'Market', 'Bull', 'Bear', 'Diamond', 'Hands'])}{np.random.randint(10, 9999)}",
                'date_posted': date_posted,
                'ticker': ticker
            })
        
        df = pd.DataFrame(posts)
        return df
    
    def save_to_database(self, df):
        """Save social media posts to the database"""
        if df.empty:
            return
            
        conn = sqlite3.connect('finsentiment.db')
        
        # Ensure all required columns exist
        required_cols = ['platform', 'content', 'post_id', 'author', 'date_posted', 
                        'sentiment_score', 'sentiment_label', 'ticker']
        
        # Filter and prepare dataframe
        for col in required_cols:
            if col not in df.columns:
                if col in ['sentiment_score', 'sentiment_label']:
                    df[col] = None
                else:
                    df[col] = ''
        
        # Convert date_posted to string format for SQLite
        df['date_posted'] = df['date_posted'].astype(str)
        
        # Save to database, ignoring duplicates
        try:
            df[required_cols].to_sql('social_media_posts', conn, if_exists='append', index=False, 
                                  index_label='id', chunksize=500)
            logger.info(f"Saved {len(df)} social media posts to database")
        except Exception as e:
            logger.error(f"Error saving social media posts to database: {str(e)}")
        finally:
            conn.close()

# Stock Data Collector
class StockDataCollector:
    def __init__(self):
        pass
    
    def get_stock_data(self, ticker, period="1y"):
        """Get historical stock data for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Add ticker column
            df['ticker'] = ticker
            
            return df
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def save_to_database(self, df):
        """Save stock data to the database"""
        if df.empty:
            return
            
        conn = sqlite3.connect('finsentiment.db')
        
        # Prepare dataframe
        df_save = df.copy()
        
        # Rename columns to match database schema
        df_save = df_save.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Convert date to string format for SQLite
        df_save['date'] = df_save['date'].astype(str)
        
        # Select only the columns we need
        columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        df_save = df_save[columns]
        
        # Save to database, replacing duplicates
        try:
            # First try to append
            df_save.to_sql('stock_prices', conn, if_exists='append', index=False)
            logger.info(f"Saved {len(df_save)} stock price records to database")
        except sqlite3.IntegrityError:
            # If duplicates exist, update them one by one
            cursor = conn.cursor()
            for _, row in df_save.iterrows():
                cursor.execute('''
                INSERT OR REPLACE INTO stock_prices 
                (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', tuple(row[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]))
            conn.commit()
            logger.info(f"Updated {len(df_save)} stock price records in database")
        except Exception as e:
            logger.error(f"Error saving stock prices to database: {str(e)}")
        finally:
            conn.close()

# Data Integration Class
class DataIntegrator:
    def __init__(self):
        self.conn = sqlite3.connect('finsentiment.db')
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def get_sentiment_by_date(self, ticker, days=30):
        """Get aggregated sentiment by date"""
        query = f"""
        SELECT 
            strftime('%Y-%m-%d', date_published) as date,
            AVG(sentiment_score) as avg_sentiment,
            COUNT(*) as article_count
        FROM news_articles
        WHERE ticker = '{ticker}'
        AND date_published >= date('now', '-{days} days')
        GROUP BY strftime('%Y-%m-%d', date_published)
        ORDER BY date
        """
        
        news_sentiment = pd.read_sql_query(query, self.conn)
        
        query = f"""
        SELECT 
            strftime('%Y-%m-%d', date_posted) as date,
            AVG(sentiment_score) as avg_sentiment,
            COUNT(*) as post_count
        FROM social_media_posts
        WHERE ticker = '{ticker}'
        AND date_posted >= date('now', '-{days} days')
        GROUP BY strftime('%Y-%m-%d', date_posted)
        ORDER BY date
        """
        
        social_sentiment = pd.read_sql_query(query, self.conn)
        
        # Get stock prices
        query = f"""
        SELECT 
            strftime('%Y-%m-%d', date) as date,
            close,
            volume
        FROM stock_prices
        WHERE ticker = '{ticker}'
        AND date >= date('now', '-{days} days')
        ORDER BY date
        """
        
        stock_data = pd.read_sql_query(query, self.conn)
        
        # Merge all data
        result = pd.merge(stock_data, news_sentiment, on='date', how='left')
        result = pd.merge(result, social_sentiment, on='date', how='left')
        
        # Fill NaN values
        result['avg_sentiment_x'] = result['avg_sentiment_x'].fillna(0.0)
        result['article_count'] = result['article_count'].fillna(0).astype(int)
        result['avg_sentiment_y'] = result['avg_sentiment_y'].fillna(0.0)
        result['post_count'] = result['post_count'].fillna(0).astype(int)
        
        # Rename columns
        result = result.rename(columns={
            'avg_sentiment_x': 'news_sentiment',
            'avg_sentiment_y': 'social_sentiment',
            'article_count': 'news_count',
            'post_count': 'social_count'
        })
        
        return result
    
    def get_sentiment_distribution(self, ticker, days=30):
        """Get sentiment distribution for news and social media"""
        # Get news sentiment distribution
        query = f"""
        SELECT 
            sentiment_label,
            COUNT(*) as count
        FROM news_articles
        WHERE ticker = '{ticker}'
        AND date_published >= date('now', '-{days} days')
        GROUP BY sentiment_label
        """
        
        news_sentiment = pd.read_sql_query(query, self.conn)
        if not news_sentiment.empty:
            news_sentiment['source'] = 'News'
        
        # Get social media sentiment distribution
        query = f"""
        SELECT 
            sentiment_label,
            COUNT(*) as count
        FROM social_media_posts
        WHERE ticker = '{ticker}'
        AND date_posted >= date('now', '-{days} days')
        GROUP BY sentiment_label
        """
        
        social_sentiment = pd.read_sql_query(query, self.conn)
        if not social_sentiment.empty:
            social_sentiment['source'] = 'Social Media'
        
        # Combine results
        combined = pd.concat([news_sentiment, social_sentiment], ignore_index=True)
        
        # Ensure all sentiment labels are represented
        for label in ['positive', 'negative', 'neutral']:
            for source in ['News', 'Social Media']:
                if len(combined[(combined['sentiment_label'] == label) & (combined['source'] == source)]) == 0:
                    combined = pd.concat([combined, pd.DataFrame({
                        'sentiment_label': [label],
                        'count': [0],
                        'source': [source]
                    })], ignore_index=True)
        
        return combined
    
    def get_top_entities(self, ticker, days=30, entity_type='news', limit=10):
        """Get top entities mentioned in news or social media"""
        if entity_type == 'news':
            table = 'news_articles'
            date_col = 'date_published'
            content_col = 'content'
        else:
            table = 'social_media_posts'
            date_col = 'date_posted'
            content_col = 'content'
        
        query = f"""
        SELECT {content_col}
        FROM {table}
        WHERE ticker = '{ticker}'
        AND {date_col} >= date('now', '-{days} days')
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            return pd.DataFrame(columns=['entity', 'count'])
        
        # Combine all text
        text = ' '.join(df[content_col].fillna('').astype(str))
        
        # Use simple word counting for now
        # In a real application, you might want to use NER (Named Entity Recognition)
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9]{1,15}\b', text)
        word_counts = {}
        
        # Common stopwords to filter out
        stopwords = set([
            'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'it', 'was', 'for',
            'on', 'with', 'as', 'by', 'at', 'from', 'be', 'this', 'have', 'are',
            'but', 'not', 'or', 'an', 'they', 'their', 'he', 'she', 'we', 'you', 'i',
            'has', 'been', 'would', 'could', 'should', 'will', 'can', 'may', 'more',
            'some', 'such', 'than', 'then', 'about', 'when', 'there', 'these', 'them'
        ])
        
        for word in words:
            word = word.lower()
            if word not in stopwords and len(word) > 2:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        
        # Create dataframe and sort
        entity_df = pd.DataFrame({
            'entity': list(word_counts.keys()),
            'count': list(word_counts.values())
        })
        
        entity_df = entity_df.sort_values('count', ascending=False).head(limit).reset_index(drop=True)
        
        return entity_df
    
    def get_recent_news(self, ticker, days=7, limit=10):
        """Get recent news articles"""
        query = f"""
        SELECT 
            title,
            source,
            url,
            date_published,
            sentiment_score,
            sentiment_label
        FROM news_articles
        WHERE ticker = '{ticker}'
        AND date_published >= date('now', '-{days} days')
        ORDER BY date_published DESC
        LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_recent_social_posts(self, ticker, days=7, limit=10):
        """Get recent social media posts"""
        query = f"""
        SELECT 
            platform,
            content,
            author,
            date_posted,
            sentiment_score,
            sentiment_label
        FROM social_media_posts
        WHERE ticker = '{ticker}'
        AND date_posted >= date('now', '-{days} days')
        ORDER BY date_posted DESC
        LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def perform_topic_modeling(self, ticker, days=30, num_topics=5):
        """Perform topic modeling on news and social media content"""
        # Get news content
        query = f"""
        SELECT content
        FROM news_articles
        WHERE ticker = '{ticker}'
        AND date_published >= date('now', '-{days} days')
        """
        
        news_df = pd.read_sql_query(query, self.conn)
        
        # Get social media content
        query = f"""
        SELECT content
        FROM social_media_posts
        WHERE ticker = '{ticker}'
        AND date_posted >= date('now', '-{days} days')
        """
        
        social_df = pd.read_sql_query(query, self.conn)
        
        # Combine content
        all_content = pd.concat([news_df, social_df], ignore_index=True)
        
        if all_content.empty or all_content['content'].isna().all():
            return None, None
        
        # Filter out empty content
        all_content = all_content[all_content['content'].notna()]
        
        # Prepare text data
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=1000,
            stop_words='english'
        )
        
        try:
            # Fit and transform the data
            dtm = vectorizer.fit_transform(all_content['content'])
            
            # Create LDA model
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=10
            )
            
            # Fit the model
            lda.fit(dtm)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                # Get top words for this topic
                top_words_idx = topic.argsort()[:-11:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weight': topic.sum()
                })
                
            # Convert to dataframe
            topics_df = pd.DataFrame(topics)
            
            # Get document-topic matrix
            doc_topic = lda.transform(dtm)
            
            return topics_df, doc_topic
            
        except Exception as e:
            logger.error(f"Error performing topic modeling: {str(e)}")
            return None, None

# Streamlit Dashboard
def run_dashboard():
    # Set page config
    st.set_page_config(page_title="Financial Sentiment Analysis Dashboard", 
                      page_icon="📊", 
                      layout="wide", 
                      initial_sidebar_state="expanded")
    
    # Add title and description
    st.title("Financial Sentiment Analysis Dashboard")
    st.markdown("""
    Analyze sentiment from news and social media for stocks and correlate with price movements.
    """)
    
    # Initialize database if it doesn't exist
    setup_database()
    
    # Sidebar
    st.sidebar.header("Controls")
    
    # Ticker input
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
    
    # Date range
    days_options = {
        "1 Week": 7,
        "2 Weeks": 14,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365
    }
    
    days_selection = st.sidebar.selectbox("Time Period", list(days_options.keys()))
    days = days_options[days_selection]
    
    # Collect data button
    if st.sidebar.button("Collect Fresh Data"):
        with st.spinner("Collecting stock data..."):
            stock_collector = StockDataCollector()
            stock_data = stock_collector.get_stock_data(ticker)
            stock_collector.save_to_database(stock_data)
        
        with st.spinner("Collecting news data..."):
            news_collector = NewsDataCollector()
            news_data = news_collector.get_financial_news(ticker, days_back=days)
            if not news_data.empty:
                news_data = news_collector.fetch_article_content(news_data)
                # Analyze sentiment
                sentiment_analyzer = SentimentAnalyzer()
                news_data = sentiment_analyzer.analyze_dataframe(news_data, 'content')
                news_collector.save_to_database(news_data)
        
        with st.spinner("Collecting social media data..."):
            social_collector = SocialMediaCollector()
            # Get Twitter data
            twitter_data = social_collector.get_twitter_data(ticker, days_back=days)
            # Get Reddit data
            reddit_data = social_collector.get_reddit_data(ticker, days_back=days)
            # Combine social media data
            social_data = pd.concat([twitter_data, reddit_data], ignore_index=True)
            if not social_data.empty:
                # Analyze sentiment
                social_data = sentiment_analyzer.analyze_dataframe(social_data, 'content')
                social_collector.save_to_database(social_data)
        
        st.success("Data collection complete!")
    
    # Initialize data integrator
    data_integrator = DataIntegrator()
    
    try:
        # Get sentiment data
        sentiment_data = data_integrator.get_sentiment_by_date(ticker, days)
        
        if sentiment_data.empty:
            st.warning(f"No data available for {ticker}. Please collect data first.")
        else:
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "News Analysis", "Social Media Analysis", "Topic Modeling"])
            
            # Tab 1: Overview
            with tab1:
                # Header metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # Current stock price
                try:
                    current_price = sentiment_data['close'].iloc[-1]
                    prev_price = sentiment_data['close'].iloc[-2] if len(sentiment_data) > 1 else current_price
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100 if prev_price > 0 else 0
                    
                    col1.metric(
                        "Current Price", 
                        f"${current_price:.2f}", 
                        f"{price_change:.2f} ({price_change_pct:.2f}%)"
                    )
                except:
                    col1.metric("Current Price", "N/A", "0.00 (0.00%)")
                
                # Average news sentiment
                try:
                    avg_news_sentiment = sentiment_data['news_sentiment'].mean()
                    col2.metric(
                        "Avg News Sentiment", 
                        f"{avg_news_sentiment:.2f}",
                        "positive" if avg_news_sentiment > 0 else "negative" if avg_news_sentiment < 0 else "neutral"
                    )
                except:
                    col2.metric("Avg News Sentiment", "N/A", "neutral")
                
                # Average social media sentiment
                try:
                    avg_social_sentiment = sentiment_data['social_sentiment'].mean()
                    col3.metric(
                        "Avg Social Sentiment", 
                        f"{avg_social_sentiment:.2f}",
                        "positive" if avg_social_sentiment > 0 else "negative" if avg_social_sentiment < 0 else "neutral"
                    )
                except:
                    col3.metric("Avg Social Sentiment", "N/A", "neutral")
                
                # Total volume
                try:
                    total_volume = sentiment_data['volume'].sum()
                    col4.metric("Total Trading Volume", f"{total_volume:,}")
                except:
                    col4.metric("Total Trading Volume", "N/A")
                
                # Main chart - Price and sentiment
                st.subheader("Stock Price and Sentiment Correlation")
                
                fig = go.Figure()
                
                # Add stock price
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['date'],
                        y=sentiment_data['close'],
                        name='Stock Price',
                        line=dict(color='#1f77b4', width=2)
                    )
                )
                
                # Create second y-axis for sentiment
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['date'],
                        y=sentiment_data['news_sentiment'],
                        name='News Sentiment',
                        line=dict(color='#ff7f0e', width=2, dash='dot'),
                        yaxis='y2'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['date'],
                        y=sentiment_data['social_sentiment'],
                        name='Social Sentiment',
                        line=dict(color='#2ca02c', width=2, dash='dot'),
                        yaxis='y2'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{ticker} Price and Sentiment Over Time",
                    xaxis_title="Date",
                    yaxis_title="Stock Price ($)",
                    yaxis2=dict(
                        title="Sentiment Score",
                        overlaying="y",
                        side="right",
                        range=[-1, 1]
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment distribution chart
                st.subheader("Sentiment Distribution")
                
                # Get sentiment distribution
                sentiment_dist = data_integrator.get_sentiment_distribution(ticker, days)
                
                if not sentiment_dist.empty:
                    fig = px.bar(
                        sentiment_dist,
                        x="sentiment_label",
                        y="count",
                        color="source",
                        barmode="group",
                        title=f"Sentiment Distribution for {ticker}",
                        labels={
                            "sentiment_label": "Sentiment",
                            "count": "Count",
                            "source": "Source"
                        },
                        color_discrete_map={
                            "News": "#ff7f0e",
                            "Social Media": "#2ca02c"
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sentiment data available for distribution chart.")
                
                # Volume and post count
                st.subheader("Volume and Content Count")
                
                fig = go.Figure()
                
                # Add volume as bar chart
                fig.add_trace(
                    go.Bar(
                        x=sentiment_data['date'],
                        y=sentiment_data['volume'],
                        name='Trading Volume',
                        marker_color='#1f77b4',
                        opacity=0.7
                    )
                )
                
                # Add news and social media counts on secondary y-axis
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['date'],
                        y=sentiment_data['news_count'],
                        name='News Articles',
                        line=dict(color='#ff7f0e', width=2),
                        yaxis='y2'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['date'],
                        y=sentiment_data['social_count'],
                        name='Social Posts',
                        line=dict(color='#2ca02c', width=2),
                        yaxis='y2'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{ticker} Volume and Content Count",
                    xaxis_title="Date",
                    yaxis_title="Trading Volume",
                    yaxis2=dict(
                        title="Content Count",
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: News Analysis
            with tab2:
                st.subheader(f"Recent News for {ticker}")
                
                # Get recent news
                news = data_integrator.get_recent_news(ticker, days)
                
                if not news.empty:
                    # Word cloud of news titles
                    st.subheader("News Word Cloud")
                    
                    # Combine all titles
                    all_titles = ' '.join(news['title'].fillna(''))
                    
                    if all_titles.strip():
                        # Generate word cloud
                        wordcloud = WordCloud(
                            width=800,
                            height=400,
                            background_color='white',
                            colormap='viridis',
                            max_words=100
                        ).generate(all_titles)
                        
                        # Display word cloud
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(plt)
                    else:
                        st.info("Not enough text data for word cloud.")
                    
                    # Top entities in news
                    st.subheader("Top Entities in News")
                    col1, col2 = st.columns(2)
                    
                    # Get top entities
                    entities = data_integrator.get_top_entities(ticker, days, entity_type='news')
                    
                    if not entities.empty:
                        with col1:
                            # Bar chart of entities
                            fig = px.bar(
                                entities,
                                x="count",
                                y="entity",
                                orientation='h',
                                title="Most Mentioned Terms in News",
                                labels={"count": "Mention Count", "entity": "Term"},
                                color="count",
                                color_continuous_scale=px.colors.sequential.Viridis
                            )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        with col1:
                            st.info("No entity data available.")
                    
                    # News sentiment over time
                    with col2:
                        # Group news by date and calculate average sentiment
                        news_by_date = news.copy()
                        news_by_date['date'] = pd.to_datetime(news_by_date['date_published']).dt.date
                        news_grouped = news_by_date.groupby('date').agg({
                            'sentiment_score': 'mean',
                            'title': 'count'
                        }).reset_index()
                        news_grouped = news_grouped.rename(columns={'title': 'article_count'})
                        
                        # Line chart of sentiment over time
                        fig = px.line(
                            news_grouped,
                            x="date",
                            y="sentiment_score",
                            title="News Sentiment Over Time",
                            labels={
                                "date": "Date",
                                "sentiment_score": "Sentiment Score"
                            },
                            markers=True
                        )
                        
                        # Add article count as bubble size
                        fig.update_traces(
                            mode='lines+markers',
                            marker=dict(
                                size=news_grouped['article_count'] * 3 + 5,
                                opacity=0.7,
                                line=dict(width=1)
                            )
                        )
                        
                        # Add zero line
                        fig.add_hline(
                            y=0,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Neutral",
                            annotation_position="bottom right"
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recent news table
                    st.subheader("Recent News Articles")
                    
                    # Format news table
                    news_table = news.copy()
                    news_table['date_published'] = pd.to_datetime(news_table['date_published']).dt.strftime('%Y-%m-%d')
                    news_table['sentiment_score'] = news_table['sentiment_score'].round(2)

                    # Add sentiment color function
                    def get_sentiment_color(score):
                        if score >= 0.05:
                            return 'background-color: rgba(0, 128, 0, 0.2)'  # Green for positive
                        elif score <= -0.05:
                            return 'background-color: rgba(255, 0, 0, 0.2)'  # Red for negative
                        else:
                            return 'background-color: rgba(255, 255, 0, 0.1)'  # Yellow for neutral

                    # Step 1: Select the desired columns first
                    news_subset = news_table[['title', 'source', 'date_published', 'sentiment_score', 'sentiment_label']]

                    # Step 2: Apply styling to the subset
                    styled_news = news_subset.style.map(
                        lambda x: get_sentiment_color(x), 
                        subset=['sentiment_score']
                    )

                    # Step 3: Display the styled DataFrame
                    st.dataframe(
                        styled_news,
                        height=400,
                        use_container_width=True
                    )
                    
                else:
                    st.info(f"No recent news available for {ticker}. Try collecting data first.")
            
            # Tab 3: Social Media Analysis
            with tab3:
                st.subheader(f"Social Media Analysis for {ticker}")
                
                # Get recent social media posts
                social_posts = data_integrator.get_recent_social_posts(ticker, days)
                
                if not social_posts.empty:
                    # Word cloud of social media posts
                    st.subheader("Social Media Word Cloud")
                    
                    # Combine all posts
                    all_posts = ' '.join(social_posts['content'].fillna(''))
                    
                    if all_posts.strip():
                        # Generate word cloud
                        wordcloud = WordCloud(
                            width=800,
                            height=400,
                            background_color='white',
                            colormap='plasma',
                            max_words=100
                        ).generate(all_posts)
                        
                        # Display word cloud
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(plt)
                    else:
                        st.info("Not enough text data for word cloud.")
                    
                    # Top entities and sentiment by platform
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Top entities in social media
                        st.subheader("Top Mentioned Terms")
                        
                        # Get top entities
                        entities = data_integrator.get_top_entities(ticker, days, entity_type='social')
                        
                        if not entities.empty:
                            # Bar chart of entities
                            fig = px.bar(
                                entities,
                                x="count",
                                y="entity",
                                orientation='h',
                                title="Most Mentioned Terms in Social Media",
                                labels={"count": "Mention Count", "entity": "Term"},
                                color="count",
                                color_continuous_scale=px.colors.sequential.Plasma
                            )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No entity data available.")
                    
                    with col2:
                        # Sentiment by platform
                        st.subheader("Sentiment by Platform")
                        
                        # Group by platform and calculate average sentiment
                        platform_sentiment = social_posts.groupby('platform').agg({
                            'sentiment_score': 'mean',
                            'content': 'count'
                        }).reset_index()
                        platform_sentiment = platform_sentiment.rename(columns={'content': 'post_count'})
                        
                        if not platform_sentiment.empty:
                            # Bar chart of sentiment by platform
                            fig = px.bar(
                                platform_sentiment,
                                x="platform",
                                y="sentiment_score",
                                title="Average Sentiment by Platform",
                                labels={
                                    "platform": "Platform",
                                    "sentiment_score": "Sentiment Score"
                                },
                                color="sentiment_score",
                                text="post_count",
                                color_continuous_scale=px.colors.diverging.RdBu,
                                color_continuous_midpoint=0
                            )
                            
                            # Add zero line
                            fig.add_hline(
                                y=0,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Neutral",
                                annotation_position="bottom right"
                            )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No platform sentiment data available.")
                    
                    # Recent social media posts
                    st.subheader("Recent Social Media Posts")
                    
                    # Format social media table
                    social_table = social_posts.copy()
                    social_table['date_posted'] = pd.to_datetime(social_table['date_posted']).dt.strftime('%Y-%m-%d')
                    social_table['sentiment_score'] = social_table['sentiment_score'].round(2)

                    # Add emoji based on sentiment
                    def get_sentiment_emoji(label):
                        if label == 'positive':
                            return '😃 ' + label
                        elif label == 'negative':
                            return '😞 ' + label
                        else:
                            return '😐 ' + label

                    social_table['sentiment'] = social_table['sentiment_label'].apply(get_sentiment_emoji)

                    # Add sentiment color function
                    def get_sentiment_color(score):
                        if score >= 0.05:
                            return 'background-color: rgba(0, 128, 0, 0.2)'  # Green for positive
                        elif score <= -0.05:
                            return 'background-color: rgba(255, 0, 0, 0.2)'  # Red for negative
                        else:
                            return 'background-color: rgba(255, 255, 0, 0.1)'  # Yellow for neutral

                    # Step 1: Select the desired columns first
                    social_subset = social_table[['platform', 'content', 'author', 'date_posted', 'sentiment_score', 'sentiment']]

                    # Step 2: Apply styling to the subset
                    styled_social = social_subset.style.map(
                        lambda x: get_sentiment_color(x), 
                        subset=['sentiment_score']
                    )

                    # Step 3: Display the styled DataFrame
                    st.dataframe(
                        styled_social,
                        height=400,
                        use_container_width=True
                    )
                    
                else:
                    st.info(f"No social media data available for {ticker}. Try collecting data first.")
            
            # Tab 4: Topic Modeling
            with tab4:
                st.subheader("Topic Modeling Analysis")
                
                # Perform topic modeling
                topics_df, doc_topic = data_integrator.perform_topic_modeling(ticker, days, num_topics=5)
                
                if topics_df is not None:
                    # Display topics
                    st.subheader("Discovered Topics")
                    
                    for _, row in topics_df.iterrows():
                        topic_id = row['topic_id']
                        words = row['words']
                        weight = row['weight']
                        
                        # Format for display
                        st.markdown(f"**Topic {topic_id+1}** (Weight: {weight:.2f})")
                        st.write(", ".join(words))
                        st.markdown("---")
                    
                    # Visualize topic distribution
                    st.subheader("Topic Distribution")
                    
                    # Create dataframe for visualization
                    topic_weights = [row['weight'] for _, row in topics_df.iterrows()]
                    topic_names = [f"Topic {i+1}" for i in range(len(topic_weights))]
                    
                    topic_dist_df = pd.DataFrame({
                        'Topic': topic_names,
                        'Weight': topic_weights
                    })
                    
                    # Pie chart
                    fig = px.pie(
                        topic_dist_df,
                        values='Weight',
                        names='Topic',
                        title="Topic Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Topic over time analysis (if enough data)
                    st.subheader("Topic Evolution Over Time")
                    
                    st.info("This feature would track how topics evolve over time, requiring more historical data.")
                    
                    # In a real implementation, you might analyze how topics change over the selected time period
                    # This would require storing document dates and grouping by date ranges
                    
                else:
                    st.info("Not enough text data available for topic modeling. Try collecting more data.")
    
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")
        logger.error(f"Dashboard error: {str(e)}")
    
    finally:
        # Close database connection
        data_integrator.close()

# Main function to run the application
def main():
    # Setup logging
    logger.info("Starting Financial Sentiment Analysis Dashboard")
    
    # Setup database
    setup_database()
    
    # Run the Streamlit dashboard
    run_dashboard()

# Entry point of the application
if __name__ == "__main__":
    main()
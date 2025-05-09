# 📊 Financial Sentiment Dashboard

A powerful dashboard for analyzing and visualizing sentiment in financial data using Natural Language Processing (NLP), machine learning, and Python. 

This project collects and processes news articles, social media posts, and stock price data to provide insights into market sentiment for specific stocks.

## 🚀 Features

- **Sentiment Analysis** 😊📊: Analyzes sentiment in financial news, tweets, and Reddit posts using VADER and TextBlob.
- **Data Collection** 🔍📰: Scrapes news from Finviz, Yahoo Finance, and MarketWatch; gathers Twitter and Reddit data (simulated if needed).
- **Data Storage** 💾📋: Saves data to a local SQLite DB with tables for news, social posts, and stock prices.
- **Logging** 📝⚠️: Tracks operations and errors in `finsentiment.log`.
- **Visualization** 📈🌐: Streamlit dashboard with charts, word clouds, and sentiment graphs.
- **Advanced Analytics** 🧠🔍: Includes topic modeling (LDA) and entity recognition for trends and keywords.
- **Real-time Updates** ⏱️🔄: Pulls fresh stock data, news, and sentiment on demand.


## 📂 Project Structure

```
├── file.py              # Main script 
├── finsentiment.db      # SQLite database
├── finsentiment.log     # Logs file
├── requirements.txt     # Python dependencies
├── .env.local           # Local environment 
└── README.md            # Project overview
```

## 🛠️ Getting Started

Follow these steps to set up and run the Financial Sentiment Dashboard:

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/MadMax-5000/Financial-sentiment-dashboard.git
   cd Financial-sentiment-dashboard
   ```

2. **Install Dependencies**  
   Install the required Python packages using the following command:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**  
   Create a `.env.local` file in the project root and add your Twitter API credentials (if you want to use real Twitter data). Example:  
   ```
   TWITTER_API_KEY=your_api_key
   TWITTER_API_SECRET=your_api_secret
   TWITTER_ACCESS_TOKEN=your_access_token
   TWITTER_ACCESS_SECRET=your_access_secret
   ```

4. **Run the Dashboard**  
   Launch the Streamlit dashboard using:  
   ```bash
   streamlit run file.py
   ```

   This will open an interactive web interface in your default browser where you can input a stock ticker (e.g., "AAPL", "NVDA") and analyze sentiment data.

## ⚠️ Notes

- Keep your `.env.local` file private and do not commit it to version control.
- The `finsentiment.db` and `finsentiment.log` files are automatically generated and ignored from version control after setup.
- The project uses simulated data for social media if API credentials are not provided or if there are connection issues.

## 📌 To Do

- [ ] Enhance web UI with real-time data updates and user authentication.
- [ ] Integrate more advanced NLP models (e.g., BERT) for improved sentiment analysis.
- [ ] Add support for additional social media platforms (e.g., LinkedIn, StockTwits).
- [ ] Improve error handling and add unit tests for robustness.
- [ ] Implement caching to speed up data retrieval and analysis.

## 📚 Dependencies

The project relies on the following Python libraries (listed in `requirements.txt`):

- `pandas`, `numpy` for data manipulation
- `nltk`, `textblob` for NLP and sentiment analysis
- `yfinance` for stock price data
- `streamlit`, `plotly` for interactive visualization
- `requests`, `beautifulsoup4` for web scraping
- `tweepy` for Twitter API access
- `wordcloud` for generating word clouds
- `sklearn` for topic modeling
- `sqlite3` for database management
- `logging` for tracking operations

## 📈 Usage Example

Once the dashboard is running, you can:

1. Enter a stock ticker (e.g., "AAPL", "NVDA") in the sidebar.
2. Select a time period (e.g., 1 week, 1 month) to analyze.
3. Click "Collect Fresh Data" to gather the latest news, social media posts, and stock prices.
4. Explore tabs like "Overview," "News Analysis," "Social Media Analysis," and "Topic Modeling" to view visualizations, sentiment trends, and key insights.

## 🤝 Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.

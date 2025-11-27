# Financial News Sentiment Analysis & Stock Price Prediction

## Project Overview

This project analyzes a comprehensive financial news dataset to discover correlations between news sentiment and stock market movements. It combines Data Engineering (DE), Financial Analytics (FA), and Machine Learning Engineering (MLE) techniques.

## Business Objective

As part of Nova Financial Solutions, this analysis aims to:

1. **Sentiment Analysis**: Perform NLP-based sentiment analysis on financial news headlines to quantify tone and sentiment
2. **Correlation Analysis**: Establish statistical correlations between news sentiment and stock price movements
3. **Predictive Insights**: Develop investment strategies based on news sentiment patterns

## Dataset

**Financial News and Stock Price Integration Dataset (FNSPID)**

- **headline**: Article release headline
- **url**: Direct link to full news article
- **publisher**: Author/creator of article
- **date**: Publication date and time (UTC-4 timezone)
- **stock**: Stock ticker symbol

Total records: 1,407,329 news articles

## Project Structure

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── Data/
│   └── newsData/
│       └── raw_analyst_ratings.csv
├── notebooks/
│   ├── task_1_eda.ipynb
│   ├── task_2_technical_analysis.ipynb
│   └── task_3_correlation_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── sentiment_analysis.py
│   ├── technical_indicators.py
│   └── correlation_analysis.py
├── scripts/
│   ├── __init__.py
│   └── download_stock_data.py
├── tests/
│   └── __init__.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Tasks

### Task 1: Git and GitHub + EDA
- ✅ Setup Python environment
- ✅ Git version control
- ✅ CI/CD pipeline
- 🔄 Exploratory Data Analysis:
  - Descriptive Statistics
  - Text Analysis & Topic Modeling
  - Time Series Analysis
  - Publisher Analysis

### Task 2: Quantitative Analysis
- Load stock price data
- Calculate technical indicators (MA, RSI, MACD) with TA-Lib
- Apply PyNance for financial metrics
- Visualize indicators and stock movements

### Task 3: Correlation Analysis
- Date alignment between news and stock data
- Sentiment analysis on headlines
- Calculate daily stock returns
- Correlation analysis between sentiment and returns

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Week-1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

### 1. Exploratory Data Analysis
```bash
jupyter notebook notebooks/task_1_eda.ipynb
```

### 2. Technical Analysis
```bash
jupyter notebook notebooks/task_2_technical_analysis.ipynb
```

### 3. Correlation Analysis
```bash
jupyter notebook notebooks/task_3_correlation_analysis.ipynb
```

## Key Performance Indicators

- **Proactivity**: Self-learning and sharing references
- **EDA Techniques**: Understanding data and discovering insights
- **Statistical Understanding**: Using suitable distributions and plots
- **Accuracy**: Correct implementation of indicators
- **Completeness**: Thorough data analysis
- **Correlation Strength**: Statistical significance of findings

## Technologies Used

- **Python 3.x**
- **Pandas & NumPy**: Data manipulation
- **Matplotlib, Seaborn, Plotly**: Visualization
- **NLTK, TextBlob**: Sentiment analysis
- **TA-Lib, pandas-ta**: Technical indicators
- **yfinance**: Stock data retrieval
- **Scikit-learn, SciPy**: Statistical analysis

## Timeline

- **Challenge Introduction**: Nov 19, 2025
- **Interim Submission**: Nov 23, 2025
- **Final Submission**: Nov 25, 2025

## Team & Support

- **Facilitators**: Kerod, Mahbubah, Filimon
- **Slack Channel**: #all-week1
- **Office Hours**: Mon-Fri, 08:00-15:00 UTC

## License

This project is part of the Nova Financial Insights training program.

## Contributors

[Your Name]

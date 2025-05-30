# Week1: Financial News Stock Analysis

## Challenge Overview
This project aims to analyze a substantial collection of financial news data to uncover correlations between news sentiment and stock market movements.

### Objectives
1. Conduct sentiment analysis on financial news headlines.
2. Identify statistical correlations between news sentiment and stock price fluctuations.
3. Offer actionable insights and investment strategies based on the analysis.

## Folder Structure

```plaintext
├── .vscode/
│   └── settings.json      
├── .github/
│   └── workflows/
│       └── unittests.yml      # GitHub Actions configuration
├── .gitignore                 # Specifies files to be ignored by Git
├── requirements.txt           # Lists project dependencies
├── README.md                  
├── src/
│   ├── __init__.py
│   
├── notebooks/
    ├── __init__.py
    ├── EDA_analysis.ipynb  
├── tests/
└── scripts/
    ├── __init__.py
    ├── EDA_analysis.py         # Script for EDA on financial news 
    └── README.md                # Documentation for the scripts directory
```

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/SimachewG/AIM_Week_1.git>
   cd  D:\KAIM\Week0_12\Week_1\AIM_Week_1>
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, since I am using windows`
   ```

3. **Install Required Packages**:
   ```bash
   pip install pandas
   pip install ipykernel
   pip install -r requirements.txt
   ```

4. **Load dataset**
5. **Analyze headline lengths and publication patterns**
6. **Plot trends by day, week, and month**
7. **Perform sentiment analysis using NLTK's `SentimentIntensityAnalyzer`**
8. **Identify major publication spikes**
9. **Perform topic modeling using LDA (Latent Dirichlet Allocation)**
10. **Analyze top publishers and their headline content**

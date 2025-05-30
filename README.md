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
├── app/
│   └── app.py     # Streamlit application at the end of the project       
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
│   ├── __init__.py 
├── tests/
└── scripts/
    ├── __init__.py
    ├── eda_analysis.py         # Script for EDA on financial news 
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


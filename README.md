# 📉 AI-Based Sales Drop Analyzer


## What it does
Upload any CSV file → detects sales drops → predicts future values → gives AI recommendations

## Project Structure
sales-analyzer/
├── app.py                  ← Tkinter Desktop UI (run this)
├── analysis.py             ← Load, Clean, Detect Drops
├── prediction.py           ← Linear Regression Prediction
├── ai_recommendations.py   ← Free Groq AI API
├── Sales_Drop_Analyzer.ipynb ← Jupyter Notebook version
├── requirements.txt        ← pip packages
└── .env.template           ← rename to .env and add API key

## Setup
1. pip install -r requirements.txt
2. Get FREE Groq API key from console.groq.com
3. python app.py

## Tech Stack
Python | Tkinter | Pandas | Scikit-learn | Groq AI (Llama 3.3)

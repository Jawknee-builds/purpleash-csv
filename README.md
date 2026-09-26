# 📊 Purpleash CSV Pipeline & ML Analytics Engine

[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Charts-Plotly-3F4F75.svg?style=flat-square&logo=plotly)](https://plotly.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> Interactive exploratory data analytics and lead conversion predictor for high-volume sales pipelines.

---

## 🎯 What Purpleash Does

Purpleash turns unstructured or messy CRM sales exports into high-signal executive insights:
- **Automated Data Sanitization**: Identifies missing records, type anomalies, and categorical skews.
- **Conversion Prediction Model**: Trains and serves a Scikit-Learn classification model to score open deals.
- **Interactive Deal Funnel Visuals**: Dynamic drill-down charting powered by Plotly.
- **Cohort Analysis**: Segment lead velocity by deal size, acquisition source, and sales rep cycle time.

---

## 🚀 Running Locally

```bash
git clone https://github.com/Jawknee-builds/purpleash-csv.git
cd purpleash-csv

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

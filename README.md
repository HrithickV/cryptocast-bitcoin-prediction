# CryptoCast: Bitcoin Price Prediction Pipeline

An end-to-end Machine Learning project comparing 1D-CNN, LSTM, and Simple RNN architectures for time-series forecasting, optimized for macOS Apple Silicon (M1).

## Project Overview
This project analyzes 14 years of Bitcoin historical data to predict future price trends. I built a full data pipeline that cleans financial data, scales features, and creates 60-day sliding window sequences for deep learning models.

## 🚀 Technical Highlights
- **Hardware Optimization**: Implemented `tensorflow-metal` to utilize the M1 GPU.
- **Data Engineering**: Processed raw CSV data, handling currency formatting and volume conversions.
- **Model Comparison**: 
    - **1D-CNN**: Focused on spatial pattern recognition.
    - **LSTM**: Captured long-term market dependencies.
    - **Simple RNN**: Served as the baseline model.

## Key Findings
The 1D-CNN model proved to be the most robust architecture for this specific task, maintaining numerical stability and accurately identifying local price trends compared to recurrent models.

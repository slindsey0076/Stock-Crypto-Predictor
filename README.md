
# Stock/Crypto Price Prediction App

Welcome to the **Stock/Crypto Price Prediction App**! This project is designed to predict future stock or cryptocurrency prices using Long Short-Term Memory (LSTM) neural networks. The app features a user-friendly graphical interface built with Tkinter and integrates real-time financial data from Yahoo Finance to provide historical price information and future predictions.

## Features

- **Stock and Crypto Prediction**: Predict the future prices of stocks or cryptocurrencies for the next 7 days.
- **Search Tickers**: Search for ticker symbols in real-time using a debounce feature to provide quick and accurate suggestions.
- **Graphical Price Display**: Visualize historical prices and future predictions using a dynamic plot.
- **Simple GUI**: Easy-to-use interface built with Tkinter to facilitate user interaction.
- **Machine Learning with LSTM**: Use of LSTM model for accurate sequential data predictions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### Prerequisites

To run this project, you'll need to have the following installed on your machine:

- Python 3.7+
- Required Python packages (listed in the `requirements.txt` file)

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

Here are the key libraries used in the project:

- `Tkinter`: For creating the graphical user interface.
- `yfinance`: To download historical stock/crypto price data.
- `NumPy`: For handling numerical operations.
- `scikit-learn`: For data preprocessing (scaling).
- `TensorFlow`: For building and training the LSTM model.
- `matplotlib`: For plotting historical prices and future predictions.
- `yahooquery`: For real-time ticker symbol search.

### Steps to Install

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/stock-crypto-prediction-app.git
   ```

2. Navigate to the project directory:

   ```bash
   cd stock-crypto-prediction-app
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Once the installation is complete, you can run the app as follows:

```bash
python app.py
```

### How to Use

1. **Enter Stock/Crypto Ticker**: Type in the stock or cryptocurrency ticker symbol (e.g., `AAPL` for Apple or `BTC-USD` for Bitcoin) in the input field.
2. **Search for Tickers**: You can search for the correct ticker symbol by typing part of the company or crypto name. The search field will suggest valid ticker symbols.
3. **Predict**: Click the "Predict" button to start the prediction process. The app will display the predicted price for the next 7 days and plot both historical and predicted prices on a graph.

---

## How It Works

1. **Data Collection**: The app uses the `yfinance` library to download historical closing prices for the selected stock or cryptocurrency over a given time period (default: 3 months).

2. **Data Preprocessing**: Data is scaled using `MinMaxScaler` to prepare it for the LSTM model. The app uses a lookback period of 30 days to create input sequences for the LSTM.

3. **Model Creation**: The LSTM model consists of two LSTM layers, followed by Dense layers. The model is compiled using the Adam optimizer and mean squared error as the loss function.

4. **Training**: The LSTM model is trained on the prepared data. This model uses the past 30 days of price data to predict the next price.

5. **Prediction**: After training, the model is used to predict prices for the next 7 days. The predicted values are displayed in both text and graphical form.

6. **Plotting**: The app uses `matplotlib` to plot both historical and predicted prices, with date formatting to improve readability.

---

## Technologies Used

- **Programming Language**: Python 3.7+
- **GUI Framework**: Tkinter
- **Data Sources**: Yahoo Finance (via `yfinance`)
- **Machine Learning Framework**: TensorFlow/Keras
- **Numerical Computing**: NumPy
- **Data Preprocessing**: scikit-learn
- **Plotting Library**: Matplotlib
- **Ticker Search**: YahooQuery

---

## Future Improvements

Here are some ideas for future development and improvements:

- **More Models**: Add other machine learning models (e.g., Random Forest, ARIMA) for comparison.
- **User Customization**: Allow users to customize prediction horizons (e.g., 1 week, 1 month).
- **GUI Enhancements**: Improve the aesthetics of the Tkinter interface with more intuitive design elements.
- **Multi-Ticker Support**: Enable predictions for multiple stocks or cryptocurrencies at once.
- **Real-Time Data**: Integrate real-time price updates and predictions.


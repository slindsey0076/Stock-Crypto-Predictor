# Import necessary libraries
import tkinter as tk  # For creating the GUI
from tkinter import ttk, messagebox  # For additional GUI elements and message boxes
import yfinance as yf  # For downloading financial data
import numpy as np  # For numerical operations
from sklearn.preprocessing import MinMaxScaler  # For scaling data
from tensorflow.keras.models import Sequential  # For creating the LSTM model
from tensorflow.keras.layers import LSTM, Dense  # For LSTM and Dense layers in the model
import matplotlib.pyplot as plt  # For plotting graphs
import matplotlib.dates as mdates  # For formatting dates on the plot
import pandas as pd  # For handling data in DataFrame
from yahooquery import search  # For searching ticker symbols

# Function to get the correct ticker symbol based on user input
def get_ticker(symbol_or_name):
    return yf.Ticker(symbol_or_name).ticker  # Returns the ticker symbol for the given name

# Function to download historical price data
def get_data(symbol, period='3mo', interval='1d'):
    # Downloads historical closing prices for the specified symbol
    data = yf.download(symbol, period=period, interval=interval)['Close']
    return data if not data.empty else None  # Returns data or None if no data is found

# Function to prepare data for the LSTM model
def prepare_data(data, lookback=30):
    scaler = MinMaxScaler()  # Scales data to a range of 0 to 1
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))  # Scale the data
    
    # Create sequences of historical data for LSTM input
    X = np.array([scaled_data[i-lookback:i] for i in range(lookback, len(scaled_data))])
    return X, scaled_data[lookback:], scaler  # Returns sequences and scaler for later use

# Function to create and compile the LSTM model
def create_model(input_shape):
    model = Sequential([  # Initializes the model
        LSTM(50, return_sequences=True, input_shape=input_shape),  # First LSTM layer
        LSTM(50),  # Second LSTM layer
        Dense(25),  # Dense layer with 25 neurons
        Dense(1)  # Output layer for price prediction
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile model with Adam optimizer and mean squared error loss
    return model

# Function to predict future prices
def predict_future(model, last_sequence, scaler, num_days=7):
    future_prices = [model.predict(last_sequence.reshape(1, -1, 1), verbose=0)[0][0]]  # Predict the first future price
    for _ in range(1, num_days):
        # Predict the next price using the last predicted price
        future_prices.append(model.predict(np.append(last_sequence[1:], future_prices[-1]).reshape(1, -1, 1), verbose=0)[0][0])
    return scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))  # Inverse scale to get actual prices

# Function to plot historical prices and predictions
def plot_predictions(symbol, actual_prices, future_prices):
    plt.figure(figsize=(12, 6))  # Set figure size for the plot
    plt.plot(actual_prices.index, actual_prices, label='Historical Prices')  # Plot historical prices
    plt.plot(pd.date_range(actual_prices.index[-1], periods=len(future_prices)+1, freq='D')[1:], future_prices, label='Future Predictions', color='red')  # Plot future predictions
    
    plt.title(f"{symbol} Price Prediction")  # Title of the plot
    plt.legend()  # Show legend
    
    # Format the x-axis dates to MM/DD/YYYY
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()  # Display the plot

# Function to handle the prediction process when the "Predict" button is clicked
def on_predict():
    try:
        # Get the ticker symbol and download data
        symbol = get_ticker(symbol_entry.get().strip().upper())  # Get user input and format
        data = get_data(symbol)  # Download historical price data
        
        # Prepare data for the model
        X, y, scaler = prepare_data(data)  # Prepare data
        
        # Create and train the model
        model = create_model((X.shape[1], 1))  # Create model with the shape of input data
        model.fit(X, y, batch_size=32, epochs=100, verbose=0)  # Train model
        
        # Make predictions
        future_prices = predict_future(model, X[-1], scaler)  # Predict future prices
        
        # Display the prediction result
        result_label.config(text=f"Predicted price after 7 days: ${future_prices[-1][0]:.2f}")  # Update result label with predicted price
        
        # Plot the results
        plot_predictions(symbol, data, future_prices)  # Plot actual vs. predicted prices
    except Exception as e:
        messagebox.showerror("Error", str(e))  # Handle exceptions and show error message

# Function to search for tickers based on user input with debouncing
def search_tickers(event):
    global search_timer
    if search_timer is not None:
        root.after_cancel(search_timer)  # Cancel previous timer if it exists

    search_query = search_entry.get().strip()  # Get the user input
    if search_query:
        search_timer = root.after(300, lambda: fetch_tickers(search_query))  # Set a new timer for search with a debounce of 300ms

# Helper function to fetch tickers based on the search query
def fetch_tickers(search_query):
    try:
        results = search(search_query)  # Fetch search results
        ticker_list.delete(0, tk.END)  # Clear current list
        if results['quotes']:
            # Insert the first result into the list
            symbol = results['quotes'][0].get('symbol', 'N/A')
            ticker_list.insert(tk.END, symbol)  # Add the ticker symbol to the list
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch tickers: {e}")  # Handle errors

# Function to populate the symbol entry from the selected ticker
def on_select(event):
    selected_ticker = ticker_list.get(ticker_list.curselection())  # Get selected ticker from the list
    symbol_entry.delete(0, tk.END)  # Clear current entry
    symbol_entry.insert(0, selected_ticker)  # Insert selected ticker into the entry

# Create the main application window
root = tk.Tk()  # Initialize the main window
root.title("Stock/Crypto Price Prediction App")  # Set window title
root.geometry("500x400")  # Set window size
root.configure(bg="#f0f0f0")  # Set background color

search_timer = None  # Initialize timer variable for debounce functionality

# Create a frame for the content
frame = ttk.Frame(root, padding="20")  # Frame for layout
frame.pack(fill=tk.BOTH, expand=True)  # Fill the frame with available space

# Create and pack GUI elements
ttk.Label(frame, text="Enter Stock/Crypto Ticker:", font=("Arial", 14)).pack(pady=10)  # Input label
symbol_entry = ttk.Entry(frame, width=30, font=("Arial", 12))  # Entry for ticker symbol
symbol_entry.pack(pady=5)  # Add space around the entry

# Search entry for ticker
ttk.Label(frame, text="Search for Tickers:", font=("Arial", 12)).pack(pady=10)  # Search label
search_entry = ttk.Entry(frame, width=30, font=("Arial", 12))  # Entry for search
search_entry.pack(pady=5)  # Add space around the search entry
search_entry.bind("<KeyRelease>", search_tickers)  # Bind key release event for the search entry

# Listbox to show search results
ticker_list = tk.Listbox(frame, width=30, height=1)  # Listbox for ticker results
ticker_list.pack(pady=10)  # Add space around the listbox
ticker_list.bind("<<ListboxSelect>>", on_select)  # Bind selection event to populate entry

# Button to trigger the prediction process
ttk.Button(frame, text="Predict", command=on_predict).pack(pady=20)  # Add space around the button

result_label = ttk.Label(frame, text="", foreground="blue", font=("Arial", 12, "bold"))  # Result label for prediction output
result_label.pack(pady=10)  # Add space around the result label

# Start the GUI event loop
root.mainloop()  # Run the application

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import yfinance as yf
import tensorflow as tf
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates

# ----------------------------------------- Step 1: Data Collection------------------------------------------

# Streamlit Title
st.title("Time Series Forecasting of Sector-Wise Stock Performance")

# Sidebar Glossary Section
st.sidebar.title("Glossary")
with st.sidebar.expander("Key Definitions"):
    st.write("**1. Sharpe Ratio:**")
    st.write("A measure of risk-adjusted return, calculated as:")
    st.latex(
        r"Sharpe\ Ratio = \frac{Portfolio\ Return - Risk-Free\ Rate}{Portfolio\ Volatility}"
    )
    st.write("Higher values indicate better risk-adjusted performance.")

    st.write("**2. Volatility:**")
    st.write(
        "The degree of variation in stock prices over time. Higher volatility indicates greater risk and "
        "uncertainty, while lower volatility suggests stability."
    )
    st.write("Volatility is often measured as the standard deviation of returns:")
    st.latex(r"Volatility = \sigma(Portfolio\ Returns)")

    st.write("**3. Correlation:**")
    st.write(
        "A statistical measure that indicates the degree to which two variables move in relation to each other."
    )
    st.write("Formula for **Pearson Correlation Coefficient**:")
    st.write("Values range from -1 to 1:")
    st.write(
        "- **+1:** Perfect positive correlation (both variables move in the same direction)."
    )
    st.write("- **0:** No correlation (variables are independent).")
    st.write(
        "- **-1:** Perfect negative correlation (variables move in opposite directions)."
    )

    st.write("**4. Time Series Forecasting:**")
    st.write(
        "The process of predicting future values based on historical time-ordered data. Commonly used to "
        "forecast stock performance and trends."
    )

    st.write("**5. SARIMAX Model:**")
    st.write("A statistical model for time series forecasting that accounts for:")
    st.write("- **Seasonality** (recurring patterns)")
    st.write("- **Autoregression** (dependence on past values)")
    st.write("- **Moving Averages** (smoothing past errors)")
    st.write("- **Exogenous Variables** (external factors)")

    st.write("**6. LSTM Model:**")
    st.write(
        "A type of neural network designed to capture long-term dependencies in sequential data. "
        "LSTM (Long Short-Term Memory) is widely used for time series forecasting, including stock price prediction."
    )

    st.write("**7. RMSE (Root Mean Square Error):**")
    st.write(
        "A metric for evaluating forecast accuracy. RMSE is calculated as the square root of the average squared "
        "differences between predicted and actual values. Lower RMSE indicates better performance."
    )
    st.write("Formula for RMSE:")
    st.latex(r"RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")

    st.write("**8. MAE (Mean Absolute Error):**")
    st.write(
        "A measure of forecast accuracy that calculates the average of absolute differences between predicted "
        "and actual values. MAE is simpler and less sensitive to outliers compared to RMSE."
    )
    st.write("Formula for MAE:")
    st.latex(r"MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")

    st.write("**9. R² Score (Coefficient of Determination):**")
    st.write(
        "A metric that measures how well the model explains the variability of the target variable. Values range "
        "from 0 (no explanation) to 1 (perfect explanation)."
    )
    st.write("Formula for R² Score:")
    st.latex(r"R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")

    st.write("**10. Monte Carlo Simulation:**")
    st.write(
        "A computational method that uses random sampling to model and predict the behavior of complex systems, "
        "such as stock prices or portfolios. It helps estimate risks and uncertainties."
    )

    st.write("**11. Risk-Free Rate:**")
    st.write(
        "The theoretical return on an investment with zero risk, typically represented by the yield of government bonds. "
        "It serves as a baseline for evaluating investment performance."
    )


# List of tickers
sectors = {
    "Technology": {
        "Apple Inc.": "AAPL",
        "Microsoft Corporation": "MSFT",
        "Alphabet Inc.": "GOOGL",
        "Amazon.com Inc.": "AMZN",
        "NVIDIA Corporation": "NVDA",
        "Taiwan Semiconductor": "TSM",
        "Salesforce Inc.": "CRM",
        "Advanced Micro Devices": "AMD",
        "Intel Corporation": "INTC",
        "Cisco Systems": "CSCO",
    },
    "Healthcare": {
        "Johnson & Johnson": "JNJ",
        "Pfizer Inc.": "PFE",
        "AbbVie Inc.": "ABBV",
        "Merck & Co.": "MRK",
        "Thermo Fisher Scientific": "TMO",
        "UnitedHealth Group": "UNH",
        "Abbott Laboratories": "ABT",
        "Eli Lilly and Company": "LLY",
        "Bristol-Myers Squibb": "BMY",
        "CVS Health Corporation": "CVS",
    },
    "Energy": {
        "Exxon Mobil Corporation": "XOM",
        "Chevron Corporation": "CVX",
        "ConocoPhillips": "COP",
        "Phillips 66": "PSX",
        "Schlumberger Limited": "SLB",
        "Marathon Petroleum": "MPC",
        "Valero Energy Corporation": "VLO",
        "Kinder Morgan": "KMI",
        "Williams Companies": "WMB",
        "Hess Corporation": "HES",
    },
}

st.sidebar.header("Select Stocks for Analysis")
selected_stocks = {}

default_stocks = {
    "Technology": "Apple Inc.",
    "Healthcare": "Johnson & Johnson",
    "Energy": "Exxon Mobil Corporation",
}

for sector, stocks in sectors.items():
    selected_stocks[sector] = st.sidebar.multiselect(
        f"Select {sector} Stocks",
        options=list(stocks.keys()),
        default=[default_stocks.get(sector)],
        key=f"{sector}_stocks",
    )


# Fetching selected stocks data
def fetch_stock_data(selected_stocks, sectors):
    """Fetch data for the selected stocks using Yahoo Finance."""
    sector_data = {}
    for sector, stock_names in selected_stocks.items():
        df_list = []
        for stock_name in stock_names:
            ticker = sectors[sector][stock_name]
            st.write(f"Fetching data for {stock_name} ({ticker})...")
            df_list.append(
                yf.download(ticker, start="2015-01-01", end="2024-12-02")["Close"]
            )
        if df_list:
            sector_data[sector] = pd.concat(df_list, axis=1).mean(
                axis=1
            )  # Average prices per sector
    return sector_data


# Button to download data
if st.sidebar.button("Download Data"):
    if not any(selected_stocks.values()):
        st.error("Please select at least one stock from any sector.")
    else:
        with st.spinner("Downloading data..."):
            sector_data = fetch_stock_data(selected_stocks, sectors)
            selected_stock_names = [
                ", ".join(stocks) for stocks in selected_stocks.values() if stocks
            ]
            selected_stock_names_str = ", ".join(selected_stock_names)

            st.success(
                f"Data downloaded successfully for selected tickers: {selected_stock_names_str}"
            )

        if sector_data:
            # Converting sector_data dictionary to DataFrame
            sector_df = pd.DataFrame(sector_data)

            # Save to CSV for user download
            sector_df.to_csv("sector_data.csv")
            st.download_button(
                label="Download CSV",
                data=sector_df.to_csv().encode("utf-8"),
                file_name="sector_data.csv",
                mime="text/csv",
            )
# Check if the CSV file exists
try:
    if os.path.exists("sector_data.csv"):
        st.write("Loaded data from previous sessions:")
        loaded_data = pd.read_csv("sector_data.csv", index_col=0, parse_dates=True)
        st.dataframe(loaded_data)
except FileNotFoundError:
    st.error("No data file found! Please download stock data using the sidebar.")
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")

sector_data_dict = sector_data  # Keep the variable for compatibility
# ----------------------------------------- Step 2: Data Preprocessing-----------------------------------------
try:
    # Loading data
    sector_df = pd.read_csv("sector_data.csv", index_col=0, parse_dates=True)

    # Interpolating missing values
    sector_df = sector_df.interpolate(method="time")

    # Normalize data
    normalized_df = (sector_df - sector_df.min()) / (sector_df.max() - sector_df.min())
    normalized_df.to_csv("normalized_sector_data.csv")

except FileNotFoundError:
    st.error("No data file found! Please download stock data using the sidebar.")


# --------------------------------------------Correlation Analysis--------------------------------------
st.subheader("Correlation Analysis")

# Computing correlation matrix
correlation_matrix = sector_df.corr()

st.write("Correlation Matrix:")
st.write(correlation_matrix)

# Plotting heatmap for correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Sector Correlation Matrix")
st.pyplot(plt)

# ---------------------------------------------------- Step 3: Time Series Forecasting----------------------------------

# -------------------------------------- SARIMAX -----------------------------------------------

st.header("SARIMAX Forecasting Results")

# Define seasonal order for SARIMAX
seasonal_order = (
    1,
    1,
    1,
    12,
)  # (p, d, q, s) where s is the seasonality (12 months in this case)
sarimax_results = {}

for sector in normalized_df.columns:
    st.subheader(f"Processing sector: {sector}")
    try:
        # Split into training and testing data
        train_data = normalized_df[sector][:-100]
        test_data = normalized_df[sector][-100:]

        # Fit SARIMAX model
        model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)

        forecast = model_fit.forecast(steps=100)
        sarimax_mae = mean_absolute_error(test_data, forecast)
        sarimax_rmse = np.sqrt(mean_squared_error(test_data, forecast))
        sarimax_r2 = r2_score(test_data, forecast)

        sarimax_results[sector] = {
            "MAE": sarimax_mae,
            "RMSE": sarimax_rmse,
            "R2": sarimax_r2,
            "Model": model,
        }

        # Plot actual vs forecasted
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_data.index, test_data, label="Actual", color="blue")
        ax.plot(test_data.index, forecast, label="Forecast", color="orange")
        ax.set_title(f"SARIMAX Forecast for {sector}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Performance")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    except Exception as e:
        st.write(f"Error processing sector {sector}: {e}")

    st.write(
        f"Sector: {sector} | MAE: {sarimax_mae:.4f} | RMSE: {sarimax_rmse:.4f} | R2: {sarimax_r2:.4f}"
    )

# Display overall results
st.subheader("Sector-wise Results using SARIMAX:")
sarimax_df = pd.DataFrame(sarimax_results).T
st.write(sarimax_df[["MAE", "RMSE", "R2"]])

# --------------------------------------------- LSTM Implementation -----------------------------------------
st.header("LSTM Forecasting Results")

# Hyperparameter configurations for LSTM
hyperparams = {
    "sequence_length": 20,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "hidden_units": [50, 30],
    "dropout_rate": 0.2,
}


def prepare_data(df, sequence_length):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[i : i + sequence_length])
        y.append(df[i + sequence_length])
    return np.array(X), np.array(y)


lstm_results = {}

for sector in normalized_df.columns:
    st.subheader(f"Processing LSTM for sector: {sector}")

    sector_data = normalized_df[sector].values

    # Preparing data
    X, y = prepare_data(sector_data, hyperparams["sequence_length"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Build the LSTM model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                hyperparams["hidden_units"][0],
                activation="relu",
                return_sequences=True,
                input_shape=(X_train.shape[1], 1),
            ),
            tf.keras.layers.Dropout(hyperparams["dropout_rate"]),
            tf.keras.layers.LSTM(hyperparams["hidden_units"][1], activation="relu"),
            tf.keras.layers.Dropout(hyperparams["dropout_rate"]),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"]),
        loss="mean_squared_error",
    )

    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Training the model
    model.fit(
        X_train_reshaped,
        y_train,
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        verbose=1,
    )

    # Predicting on test data
    predictions = model.predict(X_test_reshaped)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    lstm_results[sector] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(y_test)), y_test, label="Actual", color="blue")
    ax.plot(range(len(predictions)), predictions, label="Forecast", color="orange")
    ax.set_title(f"LSTM Forecast for {sector}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Normalized Performance")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.write(f"Sector: {sector} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

st.subheader("Sector-wise Results using LSTM:")
lstm_df = pd.DataFrame(lstm_results).T
st.write(lstm_df[["MAE", "RMSE", "R2"]])

# -----------------------------------LSTM Implementation(same interval on X axis)---------------------------------------------------------
hyperparams = {
    "sequence_length": 20,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "hidden_units": [50, 30],
    "dropout_rate": 0.2,
}


def prepare_data_with_dates(df, sequence_length, dates):
    X, y, y_dates = [], [], []
    for i in range(len(df) - sequence_length):
        X.append(df[i : i + sequence_length])
        y.append(df[i + sequence_length])
        y_dates.append(dates[i + sequence_length])
    return np.array(X), np.array(y), np.array(y_dates)


lstm_results = {}

for sector in normalized_df.columns:
    st.subheader(f"Processing sector: {sector}")

    sector_data = normalized_df[sector].values
    sector_dates = normalized_df.index.values
    train_data = normalized_df[sector][:-100]
    test_data = normalized_df[sector][-100:]
    train_dates = train_data.index
    test_dates = test_data.index

    # Preparing training data
    X_train, y_train, train_y_dates = prepare_data_with_dates(
        train_data.values, hyperparams["sequence_length"], train_dates
    )

    # Preparing testing data
    X_test, y_test, test_y_dates = prepare_data_with_dates(
        test_data.values, hyperparams["sequence_length"], test_dates
    )

    aligned_test_y_dates = test_dates[hyperparams["sequence_length"] :]

    # Step 3: Build the LSTM model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                hyperparams["hidden_units"][0],
                activation="relu",
                return_sequences=True,
                input_shape=(X_train.shape[1], 1),
            ),
            tf.keras.layers.Dropout(hyperparams["dropout_rate"]),
            tf.keras.layers.LSTM(hyperparams["hidden_units"][1], activation="relu"),
            tf.keras.layers.Dropout(hyperparams["dropout_rate"]),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"]),
        loss="mean_squared_error",
    )

    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Training the model
    history = model.fit(
        X_train_reshaped,
        y_train,
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        verbose=1,
    )

    # Predicting on test data
    predictions = model.predict(X_test_reshaped)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    lstm_results[sector] = {"MAE": mae, "RMSE": rmse, "R2": r2, "Model": model}

    st.subheader(f"Plot for {sector}")
    plt.figure(figsize=(10, 6))
    plt.plot(aligned_test_y_dates, y_test, label="Actual", color="blue")
    plt.plot(aligned_test_y_dates, predictions, label="LSTM Forecast", color="orange")
    plt.title(f"LSTM Forecast for {sector}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Performance")
    plt.legend()
    plt.grid()

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Major ticks every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format: Year-Month
    plt.xticks()

    st.pyplot(plt)

    st.write(f"Sector: {sector} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

st.subheader("Sector-wise Results using LSTM:")
result_df = pd.DataFrame.from_dict(lstm_results, orient="index")
st.write(result_df)

# ------------------------------------------- Comparing RMSE, MAE, and R2 Score for SARIMAX, LSTM --------------------------

# Combine SARIMAX and LSTM results into a DataFrame
comparison_df = pd.concat(
    [sarimax_df[["MAE", "RMSE", "R2"]], lstm_df[["MAE", "RMSE", "R2"]]], axis=1
)
comparison_df.columns = [
    "SARIMAX MAE",
    "SARIMAX RMSE",
    "SARIMAX R2",
    "LSTM MAE",
    "LSTM RMSE",
    "LSTM R2",
]

st.subheader("Comparison of SARIMAX and LSTM Results:")
st.write(comparison_df)

# Plotting MAE comparison
st.subheader("MAE Comparison")
fig, ax = plt.subplots(figsize=(12, 6))
comparison_df[["SARIMAX MAE", "LSTM MAE"]].plot(kind="bar", ax=ax)
ax.set_title("MAE Comparison: SARIMAX vs LSTM")
ax.set_ylabel("Mean Absolute Error (MAE)")
ax.set_xlabel("Sector")
ax.legend(title="Model")
ax.grid(True)
st.pyplot(fig)

# Plotting RMSE comparison
st.subheader("RMSE Comparison")
fig, ax = plt.subplots(figsize=(12, 6))
comparison_df[["SARIMAX RMSE", "LSTM RMSE"]].plot(kind="bar", ax=ax)
ax.set_title("RMSE Comparison: SARIMAX vs LSTM")
ax.set_ylabel("Root Mean Squared Error (RMSE)")
ax.set_xlabel("Sector")
ax.legend(title="Model")
ax.grid(True)
st.pyplot(fig)

# Plotting R2 comparison
st.subheader("R2 Comparison")
fig, ax = plt.subplots(figsize=(12, 6))
comparison_df[["SARIMAX R2", "LSTM R2"]].plot(kind="bar", ax=ax)
ax.set_title("R2 Comparison: SARIMAX vs LSTM")
ax.set_ylabel("R2 Score")
ax.set_xlabel("Sector")
ax.legend(title="Model")
ax.grid(True)
st.pyplot(fig)

st.title("Time Series Forecasting and Portfolio Optimization")

# ------------------------------------------- Hybrid Model --------------------------------------------
st.header("Hybrid Model (Combining SARIMAX and LSTM)")

hybrid_results = {}

for sector in normalized_df.columns:
    st.subheader(f"Processing sector: {sector}")

    sarimax_forecast = (
        sarimax_results[sector]["Model"].fit().get_forecast(steps=100).predicted_mean
    )
    lstm_forecast = lstm_results[sector]["Model"].predict(X_test_reshaped)

    min_length = min(len(sarimax_forecast), len(lstm_forecast))
    sarimax_forecast = sarimax_forecast[-min_length:]
    lstm_forecast = lstm_forecast[-min_length:].flatten()

    sarimax_rmse = sarimax_results[sector].get("RMSE", 0)
    lstm_rmse = lstm_results[sector].get("RMSE", 0)

    total_rmse = sarimax_rmse + lstm_rmse
    sarimax_weight = lstm_rmse / total_rmse if total_rmse != 0 else 0.5
    lstm_weight = sarimax_rmse / total_rmse if total_rmse != 0 else 0.5

    hybrid_forecast = (sarimax_forecast * sarimax_weight) + (
        lstm_forecast * lstm_weight
    )

    # Getting the actual test data
    test_data = normalized_df[sector][-min_length:]

    # Evaluating the hybrid forecast
    hybrid_mae = mean_absolute_error(test_data, hybrid_forecast)
    hybrid_rmse = np.sqrt(mean_squared_error(test_data, hybrid_forecast))
    hybrid_r2 = r2_score(test_data, hybrid_forecast)

    hybrid_results[sector] = {
        "MAE": hybrid_mae,
        "RMSE": hybrid_rmse,
        "R2": hybrid_r2,
        "Hybrid Forecast": hybrid_forecast,
    }
    st.write(
        f"Sector: {sector} | Hybrid MAE: {hybrid_mae:.4f} | Hybrid RMSE: {hybrid_rmse:.4f} | Hybrid R2: {hybrid_r2:.4f}"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_data.index, test_data, label="Actual", color="blue")
    ax.plot(
        test_data.index,
        sarimax_forecast,
        label="SARIMAX Forecast",
        color="orange",
        linestyle="--",
    )
    ax.plot(
        test_data.index,
        lstm_forecast,
        label="LSTM Forecast",
        color="green",
        linestyle="--",
    )
    ax.plot(test_data.index, hybrid_forecast, label="Hybrid Forecast", color="red")
    ax.set_title(f"Hybrid Forecast for {sector}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Performance")
    ax.legend()
    ax.grid()
    st.pyplot(fig)
st.subheader("Sector-wise Results using Hybrid Model:")
hybrid_df = pd.DataFrame(hybrid_results).T
st.write(hybrid_df[["MAE", "RMSE", "R2"]])

# -------------------------------------------- Monte Carlo Simulation ------------------------------------
st.header("Monte Carlo Simulation with Annualized Volatility")

np.random.seed(42)

sector_simulations = {}

for sector in normalized_df.columns:
    st.subheader(f"Running Monte Carlo simulation for {sector} sector...")

    last_price = sector_df[sector].iloc[-1]
    returns = normalized_df[sector].pct_change().dropna()

    if returns.empty:
        st.write(f"No returns available for {sector}. Skipping...")
        continue

    returns = returns.clip(lower=-0.99, upper=0.99)  # Clip returns to -99% and +99%

    # Perform Monte Carlo simulation
    simulated_prices = []
    for _ in range(1000):
        daily_returns = np.random.choice(
            returns, size=100, replace=True
        )  # Sample returns
        price_series = [last_price]
        for r in daily_returns:
            price_series.append(price_series[-1] * (1 + r))
        simulated_prices.append(price_series)

    # Convert to a numpy array
    simulated_prices = np.array(simulated_prices)
    sector_simulations[sector] = simulated_prices

    p5 = np.percentile(simulated_prices, 5, axis=0)
    p50 = np.percentile(simulated_prices, 50, axis=0)
    p95 = np.percentile(simulated_prices, 95, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p50, label="Median (50th Percentile)", color="orange")
    ax.fill_between(
        range(len(p5)),
        p5,
        p95,
        color="blue",
        alpha=0.2,
        label="5th-95th Percentile Range",
    )
    ax.set_title(f"Monte Carlo Simulation for {sector} Sector")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.grid()
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 8))
for sector in sector_simulations.keys():
    simulated_prices = sector_simulations[sector]
    p50 = np.percentile(simulated_prices, 50, axis=0)
    ax.plot(p50, label=f"{sector} Median")

ax.set_title("Median Forecast Across Sectors")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.legend(loc="upper left", fontsize=10)
ax.grid()
plt.tight_layout()
st.pyplot(fig)

# Volatility Analysis
st.subheader("Volatility Analysis:")
for sector in sector_simulations.keys():
    simulated_prices = sector_simulations[sector]
    final_prices = simulated_prices[:, -1]  # Prices on the last simulated day
    annualized_volatility = (np.std(final_prices) / np.mean(final_prices)) * np.sqrt(
        252 / 100
    )  # Annualized volatility
    st.write(f"{sector} Sector Volatility (Annualized): {annualized_volatility:.4f}")


# --------------------------------------------------------------------------------------------------------------------
st.title("Volatility Analysis and Comparison")

# Calculate historical volatility using sector_data using raw data
historical_volatility = {}
for sector, data in sector_data_dict.items():
    daily_returns = data.pct_change().dropna()
    # Annualized volatility
    historical_volatility[sector] = np.std(daily_returns) * np.sqrt(252)

# Volatility Analysis and Comparison with Historical Volatility
simulated_volatilities = []
historical_volatilities_list = []
differences = []
sectors_list = list(sector_data_dict.keys())

for sector in sector_simulations.keys():
    simulated_prices = sector_simulations[sector]
    final_prices = simulated_prices[:, -1]

    simulated_volatility = (
        np.std(final_prices) / np.mean(final_prices) * np.sqrt(252 / 100)
    )
    simulated_volatilities.append(simulated_volatility)

    hist_volatility = historical_volatility.get(sector, None)

    if hist_volatility is not None:
        historical_volatilities_list.append(hist_volatility)
        differences.append(simulated_volatility - hist_volatility)
    else:
        historical_volatilities_list.append(np.nan)
        differences.append(np.nan)

x = np.arange(len(sectors_list))
width = 0.25

# Creating a plot with bars for simulated and historical volatility
fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(
    x - width,
    simulated_volatilities,
    width,
    label="Simulated Volatility",
    color="orange",
)
bars2 = ax.bar(
    x, historical_volatilities_list, width, label="Historical Volatility", color="blue"
)

ax.set_xlabel("Sectors")
ax.set_ylabel("Annualized Volatility")
ax.set_title(
    "Comparison of Simulated and Historical Annualized Volatility with Difference"
)
ax.set_xticks(x)
ax.set_xticklabels(sectors_list, rotation=45, ha="right")
ax.legend()


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


add_labels(bars1)
add_labels(bars2)

plt.tight_layout()

st.pyplot(fig)

st.subheader("Volatility Analysis Results")
volatility_comparison_df = pd.DataFrame(
    {
        "Sector": sectors_list,
        "Simulated Volatility": simulated_volatilities,
        "Historical Volatility": historical_volatilities_list,
        "Difference": differences,
    }
)

st.write(volatility_comparison_df)

st.subheader("Differences between Simulated and Historical Volatility")
st.write(volatility_comparison_df[["Sector", "Difference"]])

# -------------------------------------- Portfolio Optimization Using Monte Carlo Simulations -------------------
st.header("Portfolio Optimization Using Monte Carlo Simulations")

# Number of portfolios to simulate
num_portfolios = 5000
results = np.zeros((3, num_portfolios))

# Calculate daily percentage returns for each sector
daily_returns = sector_df.pct_change().dropna()

# Calculate mean returns for each sector annualized and onverting to annualized percentage
sector_means = daily_returns.mean() * 252 * 100

# Covariance matrix annualized
cov_matrix = daily_returns.cov() * 252

# Define risk-free rate annualized
risk_free_rate = 0.02

# Monte Carlo simulations
for i in range(num_portfolios):
    weights = np.random.random(len(sector_df.columns))
    weights /= np.sum(weights)

    portfolio_return = np.dot(weights, sector_means / 100)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Calculating Sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio

max_sharpe_idx = results[2, :].argmax()
min_vol_idx = results[1, :].argmin()

max_sharpe_return = results[0, max_sharpe_idx]
max_sharpe_volatility = results[1, max_sharpe_idx]
min_vol_return = results[0, min_vol_idx]
min_vol_volatility = results[1, min_vol_idx]

# plotting efficient frontier
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(
    results[1, :], results[0, :], c=results[2, :], cmap="viridis", marker="o", alpha=0.6
)
colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label("Sharpe Ratio")

# Highlight maximum Sharpe ratio portfolio
ax.scatter(
    max_sharpe_volatility,
    max_sharpe_return,
    color="red",
    s=100,
    edgecolor="black",
    label="Max Sharpe Ratio",
)

# Highlight minimum volatility portfolio
ax.scatter(
    min_vol_volatility,
    min_vol_return,
    color="blue",
    s=100,
    edgecolor="black",
    label="Min Volatility",
)

ax.set_title("Efficient Frontier", fontsize=16)
ax.set_xlabel("Risk (Volatility)", fontsize=12)
ax.set_ylabel("Return", fontsize=12)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.7)
st.pyplot(fig)

st.subheader("Maximum Sharpe Ratio Portfolio:")
st.write(
    f"Return: {max_sharpe_return:.2%}, Volatility: {max_sharpe_volatility:.2%}, Sharpe Ratio: {results[2, max_sharpe_idx]:.2f}"
)

st.subheader("Minimum Volatility Portfolio:")
st.write(f"Return: {min_vol_return:.2%}, Volatility: {min_vol_volatility:.2%}")

# -----------------------------------Combining SARIMAX and Monte Carlo Simulations----------------------------

# Defining SARIMAX seasonal order
seasonal_order = (1, 1, 1, 12)

# Parameters for Monte Carlo simulation
num_simulations = 1000
forecast_horizon = 100

st.title("SARIMAX Forecasting and Monte Carlo Simulations")
st.write(
    "This app forecasts sector-wise performance using SARIMAX and Monte Carlo simulations."
)

sector_options = normalized_df.columns.tolist()
sector = st.selectbox(
    "Select a Sector", sector_options, key="sector_selectbox_SARIMAX_Monte_Carlo"
)

sarimax_forecast_results = {}


def run_sarimax(sector):
    try:
        # Split into training and testing data
        train_data = normalized_df[sector][:-forecast_horizon]
        test_data = normalized_df[sector][-forecast_horizon:]

        model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)

        forecast = model_fit.forecast(steps=forecast_horizon)

        simulations = np.zeros((num_simulations, forecast_horizon))
        for i in range(num_simulations):
            noise = np.random.normal(
                loc=0, scale=np.std(forecast), size=forecast_horizon
            )
            simulations[i, :] = forecast + noise

        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        r2 = r2_score(test_data, forecast)

        sarimax_forecast_results[sector] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Forecast": forecast,
            "Simulations": simulations,
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_data.index, test_data, label="Actual", color="blue")
        ax.plot(test_data.index, forecast, label="SARIMAX Forecast", color="orange")
        for i in range(num_simulations):
            ax.plot(test_data.index, simulations[i, :], color="lightgray", alpha=0.1)
        ax.set_title(f"SARIMAX Forecast with Monte Carlo Simulations for {sector}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Performance")
        ax.legend()
        ax.grid()

        st.pyplot(fig)

        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"R2: {r2:.4f}")

    except Exception as e:
        st.error(f"Error processing sector {sector}: {e}")


if sector:
    run_sarimax(sector)
# -----------------------------------------Combining LSTM and Monte Carlo Simulation------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

hyperparams = {
    "sequence_length": 20,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "hidden_units": [50, 30],
    "dropout_rate": 0.2,
}


def prepare_data_with_dates(df, sequence_length, dates):
    X, y, y_dates = [], [], []
    for i in range(len(df) - sequence_length):
        X.append(df[i : i + sequence_length])
        y.append(df[i + sequence_length])
        y_dates.append(dates[i + sequence_length])
    return np.array(X), np.array(y), np.array(y_dates)


st.title("LSTM Forecasting and Monte Carlo Simulations")
st.write(
    "This app forecasts sector-wise performance using LSTM and Monte Carlo simulations."
)

sector_options = normalized_df.columns.tolist()
sector = st.selectbox(
    "Select a Sector", sector_options, key="sector_selectbox_LSTM_Monte_Carlo"
)

lstm_forecast_results = {}


def run_lstm(sector):
    try:
        sector_data = normalized_df[sector].values
        sector_dates = normalized_df.index.values

        train_data = normalized_df[sector][:-forecast_horizon]
        test_data = normalized_df[sector][-forecast_horizon:]
        train_dates = train_data.index
        test_dates = test_data.index

        # Prepare training and testing data for LSTM
        X_train, y_train, train_y_dates = prepare_data_with_dates(
            train_data.values, hyperparams["sequence_length"], train_dates
        )
        X_test, y_test, test_y_dates = prepare_data_with_dates(
            test_data.values, hyperparams["sequence_length"], test_dates
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(
                    hyperparams["hidden_units"][0],
                    activation="relu",
                    return_sequences=True,
                    input_shape=(X_train.shape[1], 1),
                ),
                tf.keras.layers.Dropout(hyperparams["dropout_rate"]),
                tf.keras.layers.LSTM(hyperparams["hidden_units"][1], activation="relu"),
                tf.keras.layers.Dropout(hyperparams["dropout_rate"]),
                tf.keras.layers.Dense(1),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hyperparams["learning_rate"]
            ),
            loss="mean_squared_error",
        )

        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Training the model
        model.fit(
            X_train_reshaped,
            y_train,
            epochs=hyperparams["epochs"],
            batch_size=hyperparams["batch_size"],
            verbose=1,
        )

        predictions = model.predict(X_test_reshaped)

        simulations = np.zeros((num_simulations, len(predictions)))
        for i in range(num_simulations):
            noise = np.random.normal(
                loc=0, scale=np.std(predictions), size=len(predictions)
            )
            simulations[i, :] = predictions.flatten() + noise

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        lstm_forecast_results[sector] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Forecast": predictions,
            "Simulations": simulations,
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            test_dates[hyperparams["sequence_length"] :],
            test_data[hyperparams["sequence_length"] :],
            label="Actual",
            color="blue",
        )
        ax.plot(
            test_dates[hyperparams["sequence_length"] :],
            predictions.flatten(),
            label="LSTM Forecast",
            color="orange",
        )
        for i in range(num_simulations):
            ax.plot(
                test_dates[hyperparams["sequence_length"] :],
                simulations[i, :],
                color="lightgray",
                alpha=0.1,
            )

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        ax.set_title(f"LSTM Forecast with Monte Carlo Simulations for {sector}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Performance")
        ax.legend()
        ax.grid()

        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"R2: {r2:.4f}")

    except Exception as e:
        st.error(f"Error processing sector {sector}: {e}")


if sector:
    run_lstm(sector)
# ------------------------------------------------Sector Allocation Strategy--------------------------------------
st.title("Risk-Return Tradeoff")
st.write(
    "This app calculates and visualizes the risk-return tradeoff for different sectors."
)

daily_returns = sector_df.pct_change().dropna()

# Convert mean returns to percentages
sector_means = daily_returns.mean() * 100
# Convert volatilities to percentages
sector_volatilities = daily_returns.std() * 100

# Plotting risk-return tradeoff
plt.figure(figsize=(8, 6))
plt.scatter(sector_volatilities, sector_means, color="blue", s=50, edgecolor="black")

for i, sector in enumerate(sector_df.columns):
    plt.annotate(sector, (sector_volatilities[i], sector_means[i]), fontsize=10)

plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=100))
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))

plt.title("Risk-Return Tradeoff", fontsize=14)
plt.xlabel("Risk (Volatility, %)", fontsize=12)
plt.ylabel("Return (Mean, %)", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()

st.pyplot(plt)
# --------------------------------------------------------------------------------------------------------------------------
st.title("Sector Analysis: Sharpe Ratio, Historical Trends, and Correlation")
st.write("This app performs the following analyses on sector data:")
st.write("1. Sharpe Ratio Calculation")
st.write("2. Historical Trends (Cumulative Returns)")
st.write("3. Correlation Analysis")

# --- 1. Calculate Sharpe Ratio ---
risk_free_rate = 0.02 / 252
sharpe_ratios = {}
for sector in sector_df.columns:
    mean_return = sector_df[sector].mean()
    volatility = sector_df[sector].std()
    sharpe_ratios[sector] = (mean_return - risk_free_rate) / volatility

# Displaying Sharpe ratios
st.subheader("Sharpe Ratios")
st.write(
    pd.DataFrame.from_dict(sharpe_ratios, orient="index", columns=["Sharpe Ratio"])
)

# --- 2. Historical Trends Analysis ---
st.subheader("Historical Trends (Cumulative Returns)")
plt.figure(figsize=(10, 6))
sector_df.cumsum().plot(ax=plt.gca())
plt.title("Cumulative Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend(loc="best")
plt.grid(alpha=0.5)
st.pyplot(plt)

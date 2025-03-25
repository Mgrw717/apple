from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch KRRO stock data from Yahoo Finance
ticker = 'KRRO'
df = yf.download(ticker, start='2024-01-01', end='2025-12-31')
df.columns = df.columns.get_level_values(0)  # Fix MultiIndex issue
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

df.index.name = 'Date'

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.sma1 = self.I(SMA, price, 10)  # Short-term SMA
        self.sma2 = self.I(SMA, price, 20)  # Long-term SMA

    def next(self):
        if crossover(self.sma1, self.sma2):  # Buy signal
            self.buy()
        elif crossover(self.sma2, self.sma1):  # Sell signal
            self.sell()

# Run Backtest
initial_cash = 10000
bt = Backtest(df, SmaCross, cash=initial_cash, commission=0.002, exclusive_orders=True)
stats = bt.run()

# Show results
print(stats)

# Plot results with Buy/Sell signals
plot = bt.plot()
plt.title("KRRO Trading Strategy 2024-2025")
plt.show()

# Display Final Account Balance
final_balance = stats['Equity Final [$]']
print(f"Starting with $10,000, you would have ended with: ${final_balance:.2f}")

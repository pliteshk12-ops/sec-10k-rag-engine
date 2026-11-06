import yfinance as yf
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

def fetch_market_verification(ticker_symbol: str = "AAPL"):
    console.print(f"[bold cyan][*] Fetching live market telemetry for {ticker_symbol} via yfinance & pandas...[/bold cyan]")
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    
    # Structure financial data using pandas DataFrame for analysis
    data = {
        "Metric": ["Market Cap", "Trailing P/E", "Current Price", "52 Week High", "52 Week Low"],
        "Value": [
            info.get("marketCap", "N/A"),
            info.get("trailingPE", "N/A"),
            info.get("currentPrice", "N/A"),
            info.get("fiftyTwoWeekHigh", "N/A"),
            info.get("fiftyTwoWeekLow", "N/A")
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Display using Rich table
    table = Table(title=f"Live Market Validation: {ticker_symbol}")
    table.add_column("Financial Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    for _, row in df.iterrows():
        table.add_row(str(row["Metric"]), str(row["Value"]))
        
    console.print(table)
    return df

if __name__ == "__main__":
    fetch_market_verification("AAPL")
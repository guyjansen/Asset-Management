#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Quantitative stock analysis tool
# Author: Guy Jansen
# Date  : 08/11/2025

"""
A comprehensive stock analysis tool that performs descriptive, fundamental,
quantitative factor, and forward-looking Monte Carlo analysis.
"""

# --- 1. IMPORTS ---
import yfinance as yf
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, Optional
import requests
import io
import zipfile
import textwrap
from tabulate import tabulate
from math import pi


# --- 2. CONFIGURATION ---
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas_datareader")
plt.style.use('seaborn-v0_8-darkgrid')

# This is the global helper function defined after your imports
def plot_styled_table(df, title, palette):
    """
    Renders a pandas DataFrame as a styled image, using a fixed-layout
    approach to create a seamless, professional table without gaps.
    """
    # --- 1. Prepare Text and Figure ---
    # Make the figure wider to accommodate all columns
    fig, ax = plt.subplots(figsize=(14, 0.5 * len(df) + 1.5))
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["background"])
    ax.axis('off')

    # --- 2. Intelligent Header Wrapping ---
    wrapped_headers = [textwrap.fill(header.replace('_', ' '), width=10) for header in df.columns]

    # --- 3. Define Column Widths Manually ---
    # Define a list of widths for the index column + all data columns
    # These values are proportions and can be fine-tuned
    col_widths = [0.12] + [0.1] * len(df.columns)

    # --- 4. Create the Table ---
    table = ax.table(
        cellText=df.values,
        colLabels=wrapped_headers,
        rowLabels=df.index,
        loc='center',
        cellLoc='center', # Center the text within cells
        colWidths=col_widths # Apply the manual widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # --- 5. Style Cells ---
    cell_dict = table.get_celld()
    for (i, j), cell in cell_dict.items():
        cell.set_edgecolor(palette["highlight"]) # Use highlight for a subtle border
        
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color=palette["highlight"])
            cell.set_facecolor(palette["base"])
            cell.set_height(0.15)
        elif j == -1:  # Index (row labels)
            cell.set_text_props(weight='bold', color=palette["base"], ha='left', va='center')
            cell.set_facecolor(palette["background"]) # Match background for seamless look
            cell.set_edgecolor(palette["background"]) # No border for index
            cell.set_width(0.15) # Give more space for ticker names
        else:  # Data cells
            cell.set_text_props(color=palette["base"])
            cell.set_facecolor(palette["highlight"])

    ax.set_title(title, fontsize=18, color=palette["base"], weight='bold', pad=30)
    plt.tight_layout()
    plt.show()
    
class QuantitativeDataFetcher:
    """
    Fetches all necessary financial data without
    printing the status of each individual step.
    """
    def __init__(self, ticker: str, start_date: str = "2000-01-01"):
        self.ticker_symbol = ticker; self.start_date = start_date
        self.stock = yf.Ticker(self.ticker_symbol)
        self.price_data: Optional[pd.DataFrame] = None
        self.fundamental_data: Dict[str, pd.DataFrame] = {}
        self.ff_factors: Optional[pd.DataFrame] = None

    def fetch_all(self):
        self._fetch_price_data()
        self._fetch_fundamental_data()
        self._fetch_fama_french_4_factors()

    def _fetch_price_data(self):
        self.price_data = yf.download(self.ticker_symbol, start=self.start_date, progress=False, auto_adjust=False)
        if self.price_data.empty: raise ValueError(f"No data for ticker {self.ticker_symbol}")
    
    def _fetch_fundamental_data(self):
        self.fundamental_data = {'income_stmt_annual': self.stock.financials, 'balance_sheet_annual': self.stock.balance_sheet, 'cashflow_annual': self.stock.cashflow}

    def _fetch_fama_french_4_factors(self):
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
        self.ff_factors = self._download_and_parse_ff_zip(url)
        if self.ff_factors is not None:
            self.ff_factors.index = self.ff_factors.index.to_timestamp()

    def _download_and_parse_ff_zip(self, url: str) -> Optional[pd.DataFrame]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    content = f.read().decode('utf-8'); lines = content.splitlines()
                    start_row = 0
                    for i, line in enumerate(lines):
                        if 'Mkt-RF' in line and 'SMB' in line and 'HML' in line:
                            start_row = i + 1; break
                    if start_row == 0: return None
                    data = []
                    for i in range(start_row, len(lines)):
                        line = lines[i].strip();
                        if not line: break
                        parts = line.split()
                        if len(parts) >= 5:
                            try: data.append([parts[0]] + [float(v) for v in parts[1:]])
                            except ValueError: continue
                    if not data: return None
                    columns = ['Mkt-RF', 'SMB', 'HML', 'RF', 'MOM']
                    df = pd.DataFrame(data, columns=['Date'] + columns).set_index('Date')
                    df.index = pd.to_datetime(df.index.astype(str), format='%Y%m').to_period('M')
                    return df
        except Exception as e:
            print(f"  > Manual Fama-French download failed. Error: {e}")
            return None
            
class DescriptiveAnalysis:
    """
    Performs descriptive statistical analysis, with a price history plot
    that is focused on the most recent 5 years of data for relevance.
    """
    def __init__(self, price_data: pd.DataFrame, ticker_symbol: str):
        self.price_data = price_data.copy(); self.ticker_symbol = ticker_symbol
        self._prepare_data()
        self.annualized_return, self.annualized_volatility, self.sharpe_ratio, self.max_drawdown = [None]*4

    def _prepare_data(self):
        # Calculations are based on the full history.
        if isinstance(self.price_data.columns, pd.MultiIndex): self.price_data.columns = self.price_data.columns.droplevel(1)
        if 'Adj Close' not in self.price_data.columns: raise ValueError("DataFrame must contain 'Adj Close'")
        self.price_data['daily_return'] = self.price_data['Adj Close'].pct_change()
        self.daily_returns = self.price_data['daily_return'].dropna()

    def run_all(self):
        self._calculate_summary_statistics(); self._display_summary_statistics(); self._plot_summary_visuals()

    def _calculate_summary_statistics(self, risk_free_rate: float = 0.02):
        # Calculations are based on the full history.
        TRADING_DAYS = 252
        self.annualized_return = self.daily_returns.mean() * TRADING_DAYS
        self.annualized_volatility = self.daily_returns.std() * np.sqrt(TRADING_DAYS)
        self.sharpe_ratio = (self.annualized_return - risk_free_rate) / self.annualized_volatility if self.annualized_volatility != 0 else 0
        peak = (1 + self.daily_returns).cumprod().expanding(min_periods=1).max()
        self.max_drawdown = (((1 + self.daily_returns).cumprod() - peak) / peak).min()

    def _display_summary_statistics(self):
        """
        Displays the summary statistics by rendering them as a styled table image.
        """
        PALETTE = {"base": "#1B1464", "secondary": "#2C2F8C", "accent": "#5C6670", "background": "#D9D9D9", "highlight": "#FFFFFF"}
        
        # Create a DataFrame from the metrics
        metrics = {
            "Annualized Return": f"{self.annualized_return:.2%}",
            "Annualized Volatility": f"{self.annualized_volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Maximum Drawdown": f"{self.max_drawdown:.2%}"
        }
        df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        df.index.name = "Metric"
        
        # Call the new plotting function
        plot_styled_table(df, "Key Performance Metrics", PALETTE)

    # THE PLOT WITH THE 5-YEAR VIEW              
    def _plot_summary_visuals(self):
        """
        Generates summary plots. The price chart is zoomed to the last 5 years
        for relevance, while the histogram uses the full return history.
        """
        PALETTE = {"base": "#1B1464", "secondary": "#2C2F8C", "accent": "#5C6670", "background": "#D9D9D9", "highlight": "#FFFFFF"}
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))
        fig.set_facecolor(PALETTE["background"])

        # --- Plot 1: Price History Chart (Last 5 Years) ---
        ax1 = axes[0]
        
        # Filter data for the plot ---
        last_date = self.price_data.index[-1]
        five_years_ago = last_date - pd.DateOffset(years=5)
        plot_data = self.price_data[self.price_data.index >= five_years_ago]
        
        ax1.set_facecolor(PALETTE["background"])
        ax1.plot(plot_data.index, plot_data['Adj Close'], label='Adj Close', color=PALETTE["base"], lw=1)
        ax1.plot(plot_data.index, plot_data['Adj Close'].rolling(50).mean(), label='50-Day SMA', ls='--', color=PALETTE["accent"], lw=0.8)
        ax1.plot(plot_data.index, plot_data['Adj Close'].rolling(200).mean(), label='200-Day SMA', ls='--', color=PALETTE["secondary"], lw=0.7, alpha=0.8)

        ax1.set_title(f'Price History for {self.ticker_symbol} (Last 5 Years)', color=PALETTE["base"], fontsize=16, weight='bold')
        ax1.set_ylabel('Price (USD)', color=PALETTE["accent"]); ax1.tick_params(colors=PALETTE["accent"])
        ax1.grid(True, which='major', axis='both', linestyle='--', color=PALETTE["highlight"])
        
        legend1 = ax1.legend()
        for text in legend1.get_texts(): text.set_color(PALETTE["base"])

        # --- Plot 2: Histogram of Daily Returns (Full History) ---
        ax2 = axes[1]
        ax2.set_facecolor(PALETTE["background"])
        
        # NOTE: the full self.daily_returns is used to get the true long-term distribution
        sns.histplot(self.daily_returns, bins=100, kde=True, ax=ax2, color=PALETTE["secondary"], line_kws={'color': PALETTE['base']})
        ax2.axvline(self.daily_returns.mean(), color=PALETTE["base"], ls='--', lw=2, label=f"Mean: {self.daily_returns.mean():.4f}")

        ax2.set_title('Distribution of Daily Returns (Full History)', color=PALETTE["base"], fontsize=16, weight='bold')
        ax2.set_xlabel('Daily Return', color=PALETTE["accent"]); ax2.set_ylabel('Frequency', color=PALETTE["accent"])
        ax2.tick_params(colors=PALETTE["accent"])
        ax2.grid(True, which='major', axis='both', linestyle='--', color=PALETTE["highlight"])

        legend2 = ax2.legend()
        for text in legend2.get_texts(): text.set_color(PALETTE["base"])
        
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(PALETTE["accent"]); ax.spines['bottom'].set_color(PALETTE["accent"])

        plt.tight_layout()
        plt.show()
        
class FundamentalAnalysis:
    """
    Cleans and coerces all financial statement
    data to a numeric type, preventing future downcasting warnings.
    """
    def __init__(self, fundamental_data: Dict[str, pd.DataFrame], price_data: pd.DataFrame, ticker_symbol: str):
        self.fundamental_data = fundamental_data; self.price_data = price_data.copy(); self.ticker_symbol = ticker_symbol
        self._prepare_data(); self.ratios_df: Optional[pd.DataFrame] = None

    def _prepare_data(self):
        if isinstance(self.price_data.columns, pd.MultiIndex):
            self.price_data.columns = self.price_data.columns.droplevel(1)
            
        self.income_stmt = self.fundamental_data['income_stmt_annual'].transpose().sort_index()
        self.balance_sheet = self.fundamental_data['balance_sheet_annual'].transpose().sort_index()
        
        # Proactively convert all financial data to numeric types.
        # Any values that can't be converted will become NaN (Not a Number).
        self.income_stmt = self.income_stmt.apply(pd.to_numeric, errors='coerce')
        self.balance_sheet = self.balance_sheet.apply(pd.to_numeric, errors='coerce')
        
        self.income_stmt.index = pd.to_datetime(self.income_stmt.index)
        self.balance_sheet.index = pd.to_datetime(self.balance_sheet.index)

    def run_all(self):
        self._calculate_ratios()
        if self.ratios_df is not None and not self.ratios_df.empty:
            self._display_latest_ratios()
            self._plot_historical_ratios()
            self._plot_dupont_analysis()
        else:
            print("Could not calculate ratios due to missing data.")

    def _calculate_ratios(self):
        try:
            merged_data = pd.merge_asof(self.balance_sheet, self.price_data['Adj Close'], left_index=True, right_index=True, direction='nearest')
            market_cap = (merged_data['Share Issued'] * merged_data['Adj Close']).rename('MarketCap')
            income_cols = ['Net Income', 'Total Revenue']; balance_cols = ['Total Equity Gross Minority Interest', 'Total Assets', 'Total Debt']
            combined_df = self.income_stmt[income_cols].join(self.balance_sheet[balance_cols], how='inner').join(market_cap, how='inner')
            ratios = pd.DataFrame(index=combined_df.index)
            ratios['P/E'] = np.where(combined_df['Net Income'] > 0, combined_df['MarketCap'] / combined_df['Net Income'], np.nan)
            ratios['P/S'] = np.where(combined_df['Total Revenue'] > 0, combined_df['MarketCap'] / combined_df['Total Revenue'], np.nan)
            ratios['P/B'] = np.where(combined_df['Total Equity Gross Minority Interest'] > 0, combined_df['MarketCap'] / combined_df['Total Equity Gross Minority Interest'], np.nan)
            ratios['ROE'] = np.where(combined_df['Total Equity Gross Minority Interest'] > 0, combined_df['Net Income'] / combined_df['Total Equity Gross Minority Interest'], np.nan)
            ratios['ROA'] = np.where(combined_df['Total Assets'] > 0, combined_df['Net Income'] / combined_df['Total Assets'], np.nan)
            ratios['Net_Margin'] = np.where(combined_df['Total Revenue'] > 0, combined_df['Net Income'] / combined_df['Total Revenue'], np.nan)
            ratios['Debt_to_Equity'] = np.where(combined_df['Total Equity Gross Minority Interest'] > 0, combined_df['Total Debt'] / combined_df['Total Equity Gross Minority Interest'], np.nan)
            ratios['Asset Turnover'] = np.where(combined_df['Total Assets'] > 0, combined_df['Total Revenue'] / combined_df['Total Assets'], np.nan)
            ratios['Financial Leverage'] = np.where(combined_df['Total Equity Gross Minority Interest'] > 0, combined_df['Total Assets'] / combined_df['Total Equity Gross Minority Interest'], np.nan)
            self.ratios_df = ratios
        except (KeyError, ValueError) as e:
            print(f"Warning: Ratio calculation failed for {self.ticker_symbol}. Missing data or alignment issue: {e}")
            self.ratios_df = None
            
    def _display_latest_ratios(self):
        """
        Displays the latest fundamental ratios by rendering them as a
        styled table image.
        """
        if self.ratios_df is None or self.ratios_df.empty: return
        
        PALETTE = {"base": "#1B1464", "secondary": "#2C2F8C", "accent": "#5C6670", "background": "#D9D9D9", "highlight": "#FFFFFF"}
        
        # Prepare the DataFrame for plotting
        df_to_plot = self.ratios_df.iloc[-1].dropna().to_frame(name='Value')
        df_to_plot['Value'] = df_to_plot['Value'].apply(lambda x: f"{x:.2f}")
        df_to_plot.index.name = "Ratio"
        
        # Call the new plotting function
        plot_styled_table(df_to_plot, "Latest Fundamental Ratios", PALETTE)

    # THE PLOTS WITH THE CUSTOM PALETTE
    
    def _plot_historical_ratios(self):
        """
        Plots a grid of historical ratios, styled with the custom color palette.
        """
        if self.ratios_df is None or self.ratios_df.empty: return

        PALETTE = {"base": "#1B1464", "secondary": "#363AAB", "accent": "#5C6670", "background": "#D9D9D9", "highlight": "#FFFFFF"}
        metrics = ['P/E', 'P/S', 'P/B', 'ROE', 'ROA', 'Debt_to_Equity']
        ratios_to_plot = self.ratios_df[[m for m in metrics if m in self.ratios_df.columns]].dropna(axis=1, how='all')
        if ratios_to_plot.empty: return
        
        num_cols = 3; num_rows = (len(ratios_to_plot.columns) + num_cols - 1) // num_cols
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 4), squeeze=False)
        fig.set_facecolor(PALETTE["background"])
        
        axes = axes.flatten()
        for i, ratio_name in enumerate(ratios_to_plot.columns):
            ax = axes[i]
            ratios_to_plot[ratio_name].plot(ax=ax, kind='line', style='o-', ms=6, color=PALETTE["base"])
            
            # --- Apply Custom Styles ---
            ax.set_facecolor(PALETTE["background"])
            ax.set_title(ratio_name, color=PALETTE["base"], fontsize=14)
            ax.set_ylabel("Value", color=PALETTE["accent"])
            ax.set_xlabel("", color=PALETTE["accent"])
            ax.tick_params(colors=PALETTE["accent"])
            ax.grid(True, color=PALETTE["highlight"], linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(PALETTE["accent"])
            ax.spines['bottom'].set_color(PALETTE["accent"])

        for j in range(i + 1, len(axes)): axes[j].set_visible(False)
        fig.suptitle(f'Historical Fundamental Ratios for {self.ticker_symbol}', fontsize=18, y=1.02, color=PALETTE["base"], weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

    def _plot_dupont_analysis(self):
        """
        Plots the DuPont analysis, styled with the custom color palette.
        """
        if self.ratios_df is None or self.ratios_df.empty: return

        PALETTE = {"base": "#1B1464", "secondary": "#363AAB", "accent": "#5C6670", "background": "#D9D9D9", "highlight": "#FFFFFF"}
        dupont_cols = ['Net_Margin', 'Asset Turnover', 'Financial Leverage', 'ROE']
        if not all(col in self.ratios_df.columns for col in dupont_cols): return
        dupont_df = self.ratios_df[dupont_cols].dropna()
        if dupont_df.empty: return

        labels = dupont_df.index.year
        x = np.arange(len(labels))
        width = 0.25

        fig, ax1 = plt.subplots(figsize=(14, 8))
        fig.set_facecolor(PALETTE["background"])
        ax1.set_facecolor(PALETTE["background"])

        # --- Plot Bars with Custom Colors ---
        ax1.bar(x - width, dupont_df['Net_Margin'], width, label='Net Margin', color=PALETTE["base"], alpha=0.9)
        ax1.bar(x, dupont_df['Asset Turnover'], width, label='Asset Turnover', color=PALETTE["secondary"], alpha=0.9)
        ax1.bar(x + width, dupont_df['Financial Leverage'], width, label='Financial Leverage', color=PALETTE["accent"], alpha=0.9)

        ax2 = ax1.twinx()
        ax2.plot(x, dupont_df['ROE'], color='#FFFFFF', marker='o', linestyle='--', linewidth=3, label='ROE (Right Axis)') # A distinct highlight red

        # --- Apply Custom Styles ---
        ax1.set_ylabel('Component Value (Unitless)', fontsize=12, color=PALETTE["accent"])
        ax1.set_title(f'DuPont Analysis for {self.ticker_symbol}', fontsize=16, color=PALETTE["base"], weight='bold')
        ax1.set_xticks(x); ax1.set_xticklabels(labels)
        ax1.tick_params(axis='x', colors=PALETTE["accent"])
        ax1.tick_params(axis='y', colors=PALETTE["accent"])
        ax1.grid(True, which='major', axis='y', linestyle='--', color=PALETTE["highlight"])
        
        ax2.set_ylabel('Return on Equity (ROE)', fontsize=12, color='#1B1464')
        ax2.tick_params(axis='y', labelcolor='#1B1464')
        
        # --- Clean up Spines ---
        for spine in ['top', 'right', 'left', 'bottom']:
            ax1.spines[spine].set_color(PALETTE["accent"])
            ax2.spines[spine].set_color(PALETTE["accent"])

        # --- Unified Legend ---
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        legend = fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4, frameon=True)
        legend.get_frame().set_facecolor(PALETTE["background"])
        for text in legend.get_texts():
            text.set_color(PALETTE["base"])

        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()

class FactorAnalysis:
    """
    Performs a Carhart 4-Factor regression using a single, unified factor dataset.
    """
    def __init__(self, price_data: pd.DataFrame, ff_factors: pd.DataFrame, ticker_symbol: str):
        self.price_data = price_data.copy()
        self.ff_factors = ff_factors.copy() # contains all 4 factors
        self.ticker_symbol = ticker_symbol
        self.regression_data: Optional[pd.DataFrame] = None
        self.model_results: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None

    def _prepare_data(self):
        """Prepares data using the single 4-factor dataframe."""
        print("Preparing data for 4-factor analysis...")
        if isinstance(self.price_data.columns, pd.MultiIndex): self.price_data.columns = self.price_data.columns.droplevel(1)
        monthly_returns = (1 + self.price_data['Adj Close'].pct_change()).resample('M').prod() - 1
        monthly_returns.name = 'monthly_return'
        
        all_factors = self.ff_factors / 100.0
        all_factors.index = all_factors.index + pd.offsets.MonthEnd(0)
        
        # The merge is now much simpler
        merged_data = pd.merge(monthly_returns, all_factors, left_index=True, right_index=True, how='inner')
        merged_data['excess_return'] = merged_data['monthly_return'] - merged_data['RF']
        self.regression_data = merged_data.dropna()
        
        if self.regression_data.empty: print("Warning: Regression dataset is empty after merging.")
        else: print("...Data preparation and merge complete.")

    def _run_factor_regression(self):
        """Runs the 4-factor OLS regression."""
        if self.regression_data is None or self.regression_data.empty: return
        X = self.regression_data[['Mkt-RF', 'SMB', 'HML', 'MOM']] 
        y = self.regression_data['excess_return']
        X = sm.add_constant(X)
        self.model_results = sm.OLS(y, X).fit()
        
    def run_all(self):
        print("\n--- Running Quantitative 4-Factor Analysis ---"); self._prepare_data()
        print("\n--- Running Static 4-Factor Analysis ---"); self._run_factor_regression(); self._display_regression_summary(); self._plot_factor_betas()
        print("\n--- Quantitative Factor Analysis Complete ---")
    def _display_regression_summary(self):
        if self.model_results is None: return
        monthly_alpha = self.model_results.params.get('const', 0); annualized_alpha = (1 + monthly_alpha)**12 - 1
        print("\n=== Carhart 4-Factor Model Results (Full Period) ==="); print(self.model_results.summary()); print("------------------------------------------")
        print(f"Annualized Alpha: {annualized_alpha:.2%}"); print("==========================================")
    def _plot_factor_betas(self):
        if self.model_results is None: return
        betas = self.model_results.params.drop('const'); conf_int = self.model_results.conf_int().drop('const')
        errors = betas - conf_int.iloc[:, 0]
        fig, ax = plt.subplots(figsize=(10, 6)); betas.plot(kind='bar', ax=ax, yerr=errors, capsize=4, color='skyblue', edgecolor='black')
        ax.set_title(f'Carhart 4-Factor Exposures for {self.ticker_symbol}'); ax.set_ylabel('Beta'); ax.set_xlabel('Factor'); ax.axhline(0, color='grey', ls='--'); plt.xticks(rotation=0); plt.show()
        
class MonteCarloAnalysis:
    """
    This final version removes special Unicode characters from parameter names
    to prevent font-related warnings during plot generation.
    """
    def __init__(self, price_data: pd.DataFrame, ticker_symbol: str):
        self.price_data = price_data.copy()
        self.ticker_symbol = ticker_symbol
        self.simulation_df: Optional[pd.DataFrame] = None
        self._prepare_data()
        self.PALETTE = {"base": "#1B1464", "secondary": "#2C2F8C", "accent": "#5C6670", "background": "#D9D9D9", "highlight": "#FFFFFF", "red": "#FFFFFF"}

    def _prepare_data(self):
        if isinstance(self.price_data.columns, pd.MultiIndex):
            self.price_data.columns = self.price_data.columns.droplevel(1)
        self.price_series = self.price_data['Adj Close'].dropna()
        self.log_returns = np.log(self.price_series / self.price_series.shift(1)).dropna()

    def _estimate_svj_params(self):
        """
        Estimates the parameters for the SVJ model and displays them in a
        styled table image, using standard characters to avoid warnings.
        """
        dt = 1/252
        
        self.mu_hat = self.log_returns.mean() * 252
        self.sigma_hat = self.log_returns.std() * np.sqrt(252)
        threshold = 3 * self.log_returns.std()
        jumps = self.log_returns[abs(self.log_returns) > threshold]
        self.lambda_hat = len(jumps) / len(self.log_returns) * 252
        self.mu_j_hat = jumps.mean()
        self.sigma_j_hat = jumps.std()
        v_daily = self.log_returns**2
        self.theta_v_hat = self.sigma_hat**2
        v_daily_values = v_daily.values
        kappa_X_values = (v_daily.mean() - v_daily_values[:-1]) * dt
        delta_v_values = v_daily_values[1:] - v_daily_values[:-1]
        self.kappa_hat = np.sum(delta_v_values * kappa_X_values) / np.sum(kappa_X_values**2)
        resid = delta_v_values - self.kappa_hat * kappa_X_values
        self.xi_hat = np.std(resid) / np.sqrt(dt)

        params = {
            "Annual Drift (μ)": f"{self.mu_hat:.4f}",
            "Annual Volatility (σ)": f"{self.sigma_hat:.4f}",
            "Jump Intensity (λ)": f"{self.lambda_hat:.4f} per year",
            "Mean Jump Size (μj)": f"{self.mu_j_hat:.4f}",     
            "Jump Volatility (σj)": f"{self.sigma_j_hat:.4f}", 
            "Reversion Speed (κ)": f"{self.kappa_hat:.4f}",
            "Vol-of-Vol (ξ)": f"{self.xi_hat:.4f}"
        }
        params_df = pd.DataFrame.from_dict(params, orient='index', columns=['Estimated Value'])
        params_df.index.name = "Parameter"
        
        plot_styled_table(params_df, "Monte Carlo Model Parameters", self.PALETTE)

    def _run_gbm_svj_simulation(self, num_simulations: int = 500, num_days: int = 252):
        S0 = self.price_series.iloc[-1]; dt = 1/252
        all_paths = np.zeros((num_days, num_simulations))
        for i in range(num_simulations):
            S = np.zeros(num_days); S[0] = S0; v = np.zeros(num_days); v[0] = self.sigma_hat**2
            for t in range(1, num_days):
                v[t] = np.abs(v[t-1] + self.kappa_hat * (self.theta_v_hat - v[t-1]) * dt + self.xi_hat * np.sqrt(v[t-1]) * np.sqrt(dt) * np.random.randn())
                num_jumps = np.random.poisson(self.lambda_hat * dt); jump_factor = np.sum(np.random.normal(self.mu_j_hat, self.sigma_j_hat, num_jumps)) if num_jumps > 0 else 0
                S[t] = S[t-1] * np.exp((self.mu_hat - 0.5 * v[t]) * dt + np.sqrt(v[t]) * np.sqrt(dt) * np.random.randn() + jump_factor)
            all_paths[:, i] = S
        time_index = pd.date_range(start=self.price_series.index[-1] + pd.Timedelta(days=1), periods=num_days, freq='B')
        self.simulation_df = pd.DataFrame(all_paths, index=time_index)
        
    def _apply_custom_style(self, fig, ax, title):
        fig.set_facecolor(self.PALETTE["background"]); ax.set_facecolor(self.PALETTE["background"])
        ax.set_title(title, fontsize=16, color=self.PALETTE["base"], weight='bold')
        ax.set_xlabel('Date', color=self.PALETTE["accent"]); ax.set_ylabel('Price ($)', color=self.PALETTE["accent"])
        ax.tick_params(colors=self.PALETTE["accent"]); ax.grid(True, which='major', axis='both', linestyle='--', color=self.PALETTE["highlight"])
        for spine in ax.spines.values(): spine.set_color(self.PALETTE["accent"])

    def _plot_simulation_paths(self):
        fig, ax = plt.subplots(figsize=(14, 8)); self._apply_custom_style(fig, ax, f'{self.ticker_symbol} Monte Carlo Paths')
        ax.plot(self.simulation_df.index, self.simulation_df.values, color=self.PALETTE["accent"], alpha=0.2, lw=0.8)
        ax.plot(self.simulation_df.index, self.simulation_df.mean(axis=1), color=self.PALETTE["red"], linewidth=2, label='Mean Path')
        legend = ax.legend(); legend.get_frame().set_facecolor(self.PALETTE["background"]); legend.get_frame().set_edgecolor('none')
        for text in legend.get_texts(): text.set_color(self.PALETTE["base"])
        plt.tight_layout(); plt.show()

    def _plot_confidence_intervals(self):
        percentiles = self.simulation_df.quantile([0.05, 0.25, 0.50, 0.75, 0.95], axis=1).T
        fig, ax = plt.subplots(figsize=(14, 8)); self._apply_custom_style(fig, ax, f'{self.ticker_symbol} Monte Carlo CI')
        ax.plot(percentiles.index, percentiles[0.50], color=self.PALETTE["red"], label='Median Path', lw=2)
        ax.fill_between(percentiles.index, percentiles[0.05], percentiles[0.95], color=self.PALETTE["accent"], alpha=0.2, label='5-95% CI')
        ax.fill_between(percentiles.index, percentiles[0.25], percentiles[0.75], color=self.PALETTE["secondary"], alpha=0.3, label='25-75% CI')
        legend = ax.legend(); legend.get_frame().set_facecolor(self.PALETTE["background"]); legend.get_frame().set_edgecolor('none')
        for text in legend.get_texts(): text.set_color(self.PALETTE["base"])
        plt.tight_layout(); plt.show()
        
    def run_all(self):
        self._estimate_svj_params()
        self._run_gbm_svj_simulation()
        if self.simulation_df is not None:
            self._plot_simulation_paths()
            self._plot_confidence_intervals()
        

class PeerAnalysis:
    """
    Performs a comparative analysis with a final, corrected implementation of
    the grouped horizontal bar charts, styled with a custom professional color palette.
    """
    def __init__(self, primary_ticker: str, peer_tickers: list):
        self.primary_ticker = primary_ticker
        self.all_tickers = [primary_ticker] + peer_tickers
        self.comparison_df: Optional[pd.DataFrame] = None

    def __init__(self, primary_ticker: str, peer_tickers: list):
        self.primary_ticker = primary_ticker
        self.all_tickers = [primary_ticker] + peer_tickers
        self.comparison_df: Optional[pd.DataFrame] = None

    def _gather_peer_data(self):
        """
        Gathers peer data silently by calling specific calculation methods
        instead of the verbose run_all() methods.
        """
        all_metrics = []
        for ticker in self.all_tickers:
            try:
                # Fetch data silently
                fetcher = QuantitativeDataFetcher(ticker=ticker)
                fetcher.fetch_all()
                
                price_data, fundamental_data = fetcher.price_data, fetcher.fundamental_data
                metrics = {'Ticker': ticker} # Initialize the dictionary
                
                if price_data is not None:
                    # --- THE CRITICAL FIX: Call the calculation method directly ---
                    desc_analyzer = DescriptiveAnalysis(price_data, ticker)
                    desc_analyzer._calculate_summary_statistics() # This method is quiet
                    metrics['Volatility'] = desc_analyzer.annualized_volatility
                    metrics['Sharpe Ratio'] = desc_analyzer.sharpe_ratio
                
                if fundamental_data and not all(df.empty for df in fundamental_data.values()):
                    # --- THE CRITICAL FIX: Call the calculation method directly ---
                    fund_analyzer = FundamentalAnalysis(fundamental_data, price_data, ticker)
                    fund_analyzer._calculate_ratios() # This method is quiet
                    
                    if fund_analyzer.ratios_df is not None:
                        metrics.update(fund_analyzer.ratios_df.iloc[-1].to_dict())
                        revenue = fund_analyzer.income_stmt.get('Total Revenue')
                        if revenue is not None and len(revenue) > 1:
                            metrics['YoY Revenue Growth'] = revenue.pct_change().iloc[-1]
                
                all_metrics.append(metrics)

            except Exception as e:
                print(f"Could not analyze ticker {ticker}. Error: {e}")

        self.comparison_df = pd.DataFrame(all_metrics).set_index('Ticker')

    def _display_comparison_table(self):
        """
        Displays the peer comparison data by rendering it as a styled table image.
        """
        if self.comparison_df is None or self.comparison_df.empty: return
        
        PALETTE = {"base": "#1B1464", "secondary": "#2C2F8C", "accent": "#5C6670", "background": "#D9D9D9", "highlight": "#FFFFFF"}
        display_cols = ['P/E', 'P/S', 'ROE', 'Net_Margin', 'Debt_to_Equity', 'Volatility', 'Sharpe Ratio', 'YoY Revenue Growth']
        display_cols = [col for col in display_cols if col in self.comparison_df.columns]
        
        df_to_plot = self.comparison_df[display_cols].copy()
        
        # Format columns appropriately
        for col in df_to_plot.columns:
            if "Growth" in col or "Volatility" in col:
                df_to_plot[col] = df_to_plot[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            else:
                df_to_plot[col] = df_to_plot[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        plot_styled_table(df_to_plot, "Peer Comparison Table", PALETTE)

    def _plot_grouped_bar_charts(self):
        pass
    def run_all(self):
        self._gather_peer_data()
        self._display_comparison_table()
        self._plot_grouped_bar_charts()
    # THE PLOT WITH THE CUSTOM PALETTE             
    def _plot_grouped_bar_charts(self):
        """
        Creates a series of horizontal bar charts styled with the specified
        custom color palette for a professional, branded look.
        """
        if self.comparison_df is None or self.comparison_df.empty: return
        
        # --- Define the Custom Color Palette ---
        PALETTE = {
            "base": "#1B1464",         # For peer bars & main text
            "secondary": "#363AAB",    # For the primary ticker bar
            "accent": "#5C6670",       # For axis lines and less important text
            "background": "#D9D9D9",   # For the chart background
            "highlight": "#FFFFFF"     # For grid lines
        }

        chart_groups = {
            "Valuation": ['P/E', 'P/S'],
            "Profitability": ['ROE', 'Net_Margin', 'Sharpe Ratio'],
            "Risk & Leverage": ['Volatility', 'Debt_to_Equity'],
            "Growth": ['YoY Revenue Growth']
        }

        for title, metrics in chart_groups.items():
            plot_metrics = [m for m in metrics if m in self.comparison_df.columns]
            if not plot_metrics: continue
            plot_df = self.comparison_df[plot_metrics].copy().dropna(subset=plot_metrics, how='all')
            if plot_df.empty: continue

            num_metrics = len(plot_metrics)
            
            fig, axes = plt.subplots(
                nrows=1, ncols=num_metrics, 
                figsize=(7 * num_metrics, 0.5 * len(plot_df.index) + 1.5),
                sharey=False
            )
            
            if num_metrics == 1: axes = [axes]
            
            # --- Apply Global Figure Styles ---
            fig.set_facecolor(PALETTE["background"])
            fig.suptitle(title, fontsize=18, y=1.05, color=PALETTE["base"], weight='bold')
            
            for i, metric in enumerate(plot_metrics):
                ax = axes[i]
                sorted_df = plot_df[[metric]].dropna().sort_values(by=metric, ascending=True)
                
                # --- Apply Bar Colors ---
                colors = [PALETTE["secondary"] if ticker == self.primary_ticker else PALETTE["base"] for ticker in sorted_df.index]
                
                ax.barh(sorted_df.index, sorted_df[metric], color=colors)

                # --- Apply Axis & Text Styles ---
                ax.set_facecolor(PALETTE["background"])
                ax.set_title(metric, color=PALETTE["base"], fontsize=14)
                ax.set_xlabel("Value", color=PALETTE["accent"])
                ax.tick_params(axis='x', colors=PALETTE["accent"])
                ax.tick_params(axis='y', colors=PALETTE["base"], labelsize=12)
                
                # --- Apply Grid and Spine Styles ---
                ax.grid(True, which='major', axis='x', linestyle='--', color=PALETTE["highlight"])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color(PALETTE["accent"])
                ax.spines['bottom'].set_color(PALETTE["accent"])

                if i == 0:
                    ax.set_ylabel("Ticker", color=PALETTE["accent"])

                # Add data labels
                for bar in ax.patches:
                    ax.text(
                        bar.get_width() + bar.get_width() * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f'{bar.get_width():.2f}',
                        va='center',
                        color=PALETTE["base"]
                    )
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def run_all(self):
        self._gather_peer_data()
        self._display_comparison_table()
        self._plot_grouped_bar_charts()

class ScoringModel:
    """
    Synthesizes peer comparison data into a quantitative scorecard
    and generates a brief, rules-based investment thesis.
    This version is robust against MultiIndex columns and missing data.
    """
    def __init__(self, comparison_df: pd.DataFrame, price_data: pd.DataFrame, primary_ticker: str):
        self.raw_data = comparison_df.dropna(how='all', axis=1)
        self.price_data = price_data.copy()
        self.primary_ticker = primary_ticker
        self.scorecard: Optional[pd.DataFrame] = None
        
    def _calculate_scores(self):
        """Calculates normalized scores (1-10) for each factor based on peer ranks."""
        ranks = pd.DataFrame(index=self.raw_data.index)
        factor_map = {
            'Value': [('P/E', True), ('P/S', True)],
            'Quality': [('ROE', False), ('Net_Margin', False), ('Debt_to_Equity', True)],
            'Growth': [('YoY Revenue Growth', False)],
            'Momentum': [('Sharpe Ratio', False)]
        }
        
        for factor, metrics in factor_map.items():
            factor_ranks = pd.DataFrame(index=self.raw_data.index)
            # Only process metrics that exist in the gathered data
            valid_metrics = [m for m in metrics if m[0] in self.raw_data.columns]
            if not valid_metrics: continue

            for metric, asc in valid_metrics:
                factor_ranks[metric] = self.raw_data[metric].rank(ascending=asc, method='first')
            ranks[factor] = factor_ranks.mean(axis=1)

        # Normalize ranks to a 1-10 score, handling cases with few peers
        num_peers = len(ranks)
        if num_peers > 1:
            self.scorecard = ((ranks - 1) / (num_peers - 1)) * 9 + 1
        else: # Handle case with only one ticker
            self.scorecard = pd.DataFrame(5.0, index=ranks.index, columns=ranks.columns)

        self.scorecard['Overall'] = self.scorecard.mean(axis=1, skipna=True) # skipna handles missing Growth
        
    def _generate_thesis(self):
        """Generates a simple text summary based on the primary ticker's scores."""
        if self.scorecard is None: return
        
        ticker_scores = self.scorecard.loc[self.primary_ticker]
        
        def score_to_text(name, score):
            if pd.isna(score): return f"has **insufficient data for a {name} score**"
            if score >= 8: return f"scores exceptionally on **{name} ({score:.1f}/10)**"
            if score >= 6: return f"scores well on **{name} ({score:.1f}/10)**"
            if score >= 4: return f"is average on **{name} ({score:.1f}/10)**"
            return f"scores poorly on **{name} ({score:.1f}/10)**"

        # 1. Defensively handle potential MultiIndex columns
        price_df = self.price_data.copy()
        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.droplevel(1)
            
        # 2. Ensure we get a single float value, not a Series
        sma200 = price_df['Adj Close'].rolling(window=200).mean().iloc[-1]
        current_price = price_df['Adj Close'].iloc[-1]
        sentiment = "Positive (Price > 200-day SMA)" if current_price > sma200 else "Negative (Price < 200-day SMA)"

        print("\n\n--- Quantitative Investment Thesis ---")
        print(f"**Ticker:** {self.primary_ticker}")
        print(f"**Overall Score:** {ticker_scores['Overall']:.1f}/10")
        print("-" * 36)
        print(f"* **Valuation:** The stock {score_to_text('Value', ticker_scores.get('Value'))}.")
        print(f"* **Quality:** The business {score_to_text('Quality', ticker_scores.get('Quality'))}.")
        print(f"* **Growth:** The company {score_to_text('Growth', ticker_scores.get('Growth'))}.")
        print(f"* **Momentum:** The stock's risk-adjusted returns {score_to_text('Momentum', ticker_scores.get('Momentum'))}. Current market sentiment appears **{sentiment}**.")
        print("----------------------------------------")

    def display_scorecard(self):
        if self.scorecard is None: return
        print("\n\n=== Quantitative Scorecard (1=Worst, 10=Best) ===")
        print(tabulate(self.scorecard.sort_values(by='Overall', ascending=False).round(1), headers='keys', tablefmt='grid'))
        print("==================================================")
        
    def run_all(self):
        self._calculate_scores()
        self.display_scorecard()
        self._generate_thesis()

class AdvancedRiskAnalysis:
    # ... (all calculation methods are unchanged) ...
    def __init__(self, descriptive_analyzer: DescriptiveAnalysis, monte_carlo_analyzer: MonteCarloAnalysis):
        self.daily_returns = descriptive_analyzer.daily_returns
        self.simulation_df = monte_carlo_analyzer.simulation_df
        self.ticker_symbol = descriptive_analyzer.ticker_symbol
        self.risk_metrics = {}

    def _calculate_historical_var_cvar(self, confidence_level: float = 0.95):
        alpha = 1 - confidence_level
        self.risk_metrics['Historical'] = {'VaR': self.daily_returns.quantile(alpha), 'CVaR': self.daily_returns[self.daily_returns <= self.daily_returns.quantile(alpha)].mean()}
    def _calculate_parametric_var_cvar(self, confidence_level: float = 0.95):
        from scipy.stats import norm
        alpha = 1 - confidence_level; mu = self.daily_returns.mean(); sigma = self.daily_returns.std()
        self.risk_metrics['Parametric (Normal)'] = {'VaR': norm.ppf(alpha, loc=mu, scale=sigma), 'CVaR': mu - sigma * (norm.pdf(norm.ppf(alpha)) / alpha)}
    def _calculate_monte_carlo_var_cvar(self, confidence_level: float = 0.95):
        if self.simulation_df is None: return
        alpha = 1 - confidence_level; all_sim_returns = self.simulation_df.pct_change().dropna().values.flatten()
        var = np.quantile(all_sim_returns, alpha)
        self.risk_metrics['Monte Carlo'] = {'VaR': var, 'CVaR': all_sim_returns[all_sim_returns <= var].mean()}

    def display_risk_metrics(self, confidence_level: float = 0.95):
        """
        Displays the risk metrics by rendering them as a styled table image.
        """
        title = f"Advanced Risk Analysis ({int(confidence_level*100)}% Confidence)"
        df = pd.DataFrame(self.risk_metrics).T
        df.index.name = "Method"
        
        df_percent = df.map(lambda x: f"{x:.2%}")
        
        # Call the global plotting function
        plot_styled_table(df_percent, title, self.PALETTE) 
        
        print("\n* VaR (Value at Risk): The most you can expect to lose on a given day.")
        print("* CVaR (Conditional VaR): The average loss on days that are worse than the VaR.")
        print("======================================================")
        
    def run_all(self, confidence_level: float = 0.95):
        self.PALETTE = {"base": "#1B1464", "secondary": "#2C2F8C", "accent": "#5C6670", "background": "#D9D9D9", "highlight": "#FFFFFF"}
        self._calculate_historical_var_cvar(confidence_level)
        self._calculate_parametric_var_cvar(confidence_level)
        self._calculate_monte_carlo_var_cvar(confidence_level)
        self.display_risk_metrics(confidence_level)
        
# --- 4. MAIN EXECUTION ---
def main(ticker: str, peer_list: list = None):
    """
    Main function to orchestrate the entire stock analysis workflow, including
    the new Advanced Risk Analysis module.
    """
    print(f"--- Starting Full Analysis for Primary Ticker: {ticker} ---")

    data_fetcher = QuantitativeDataFetcher(ticker=ticker)
    data_fetcher.fetch_all()
    
    price_data = data_fetcher.price_data
    fundamental_data = data_fetcher.fundamental_data
    ff_factors = data_fetcher.ff_factors

    desc_analyzer = None # Initialize to None
    if price_data is not None:
        desc_analyzer = DescriptiveAnalysis(price_data, ticker_symbol=ticker)
        desc_analyzer.run_all()

    if fundamental_data and not all(df.empty for df in fundamental_data.values()):
        fund_analyzer = FundamentalAnalysis(fundamental_data, price_data, ticker_symbol=ticker)
        fund_analyzer.run_all()
        
    if price_data is not None and ff_factors is not None:
        factor_analyzer = FactorAnalysis(price_data, ff_factors, ticker_symbol=ticker)
        factor_analyzer.run_all()
        
    mc_analyzer = None # Initialize to None
    if price_data is not None:
        mc_analyzer = MonteCarloAnalysis(price_data, ticker_symbol=ticker)
        mc_analyzer.run_all()
        
    if desc_analyzer is not None and mc_analyzer is not None:
        risk_analyzer = AdvancedRiskAnalysis(desc_analyzer, mc_analyzer)
        risk_analyzer.run_all()
        
    comparison_data = None
    if peer_list:
        peer_analyzer = PeerAnalysis(primary_ticker=ticker, peer_tickers=peer_list)
        peer_analyzer.run_all()
        comparison_data = peer_analyzer.comparison_df
    
    if comparison_data is not None and not comparison_data.empty:
        scorer = ScoringModel(comparison_data, price_data, primary_ticker=ticker)
        scorer.run_all()
        

if __name__ == "__main__":
    TICKER_TO_ANALYZE = "AAPL"
    PEER_GROUP = ["MSFT", "GOOGL", "NVDA", "META"]
    main(ticker=TICKER_TO_ANALYZE, peer_list=PEER_GROUP)


# In[ ]:





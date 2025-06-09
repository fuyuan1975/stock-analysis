# app.py - 加入除錯功能版本
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# 設定頁面
st.set_page_config(
    page_title="專業股票分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂CSS樣式
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 快取裝飾器
@st.cache_data(ttl=300)  # 快取5分鐘
def get_stock_data(symbol: str, period: str = "1y"):
    """快取股票數據獲取"""
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period=period)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)  # 快取10分鐘
def get_stock_info(symbol: str):
    """快取股票基本資訊"""
    try:
        stock = yf.Ticker(symbol)
        return stock.info or {}
    except:
        return {}

# StockAnalyzer 類別
class StockAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
        self._load_data()
    
    def _load_data(self):
        """載入所需資料"""
        # 使用快取功能提升效能
        self.info = get_stock_info(self.symbol)
        
        # 載入歷史數據
        self.hist = get_stock_data(self.symbol, "1y")
        if self.hist.empty:
            # 嘗試載入更短的時間期間
            self.hist = get_stock_data(self.symbol, "3mo")
            if self.hist.empty:
                self.hist = get_stock_data(self.symbol, "1mo")
        
        # 載入財務資料（不使用快取，因為更新頻率低）
        try:
            self.financials = self.stock.financials
        except:
            self.financials = None
            
        try:
            self.balance_sheet = self.stock.balance_sheet
        except:
            self.balance_sheet = None
            
        try:
            self.cashflow = self.stock.cashflow
        except:
            self.cashflow = None
    
    def calculate_financial_ratios(self) -> dict:
        """計算財務比率"""
        ratios = {}
        
        try:
            if hasattr(self, 'financials') and self.financials is not None and not self.financials.empty:
                revenue = self._get_first_available_value(
                    self.financials, 
                    ['Total Revenue', 'Revenue', 'Net Revenue']
                )
                net_income = self._get_first_available_value(
                    self.financials,
                    ['Net Income', 'Net Income Common Stockholders']
                )
                
                if revenue and net_income:
                    ratios['淨利率'] = (net_income / revenue) * 100
            
            if hasattr(self, 'balance_sheet') and self.balance_sheet is not None and not self.balance_sheet.empty:
                total_assets = self._get_first_available_value(
                    self.balance_sheet,
                    ['Total Assets']
                )
                total_equity = self._get_first_available_value(
                    self.balance_sheet,
                    ['Total Stockholder Equity', 'Total Equity', 'Stockholders Equity']
                )
                total_debt = self._get_first_available_value(
                    self.balance_sheet,
                    ['Total Debt', 'Long Term Debt', 'Total Liabilities']
                )
                
                if hasattr(self, 'financials') and self.financials is not None:
                    net_income = self._get_first_available_value(
                        self.financials,
                        ['Net Income', 'Net Income Common Stockholders']
                    )
                    
                    if net_income and total_assets:
                        ratios['ROA'] = (net_income / total_assets) * 100
                    if net_income and total_equity:
                        ratios['ROE'] = (net_income / total_equity) * 100
                
                if total_debt and total_assets:
                    ratios['負債比率'] = (total_debt / total_assets) * 100
            
            # 市場指標
            ratios['P/E'] = self.info.get('trailingPE', np.nan)
            ratios['P/B'] = self.info.get('priceToBook', np.nan)
            ratios['股息率'] = self.info.get('dividendYield', 0) * 100 if self.info.get('dividendYield') else 0
            
        except Exception as e:
            st.error(f"計算比率時發生錯誤: {e}")
            
        return ratios
    
    def _get_first_available_value(self, df, field_names):
        """取得第一個可用值"""
        for field in field_names:
            if field in df.index:
                try:
                    if len(df.columns) > 0:
                        value = df.loc[field].iloc[0]
                        if pd.notna(value):
                            return value
                except:
                    continue
        return None
    
    def calculate_technical_indicators(self) -> pd.DataFrame:
        """計算技術指標"""
        df = self.hist.copy()
        
        # 移動平均
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['Signal'] = self._calculate_macd(df['Close'])
        df['MACD_Histogram'] = df['MACD'] - df['Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """計算 RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """計算 MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

# 主要應用程式邏輯
def main():
    # 標題
    st.title("📊 專業股票分析系統")
    st.markdown("### 為資深基金經理人打造的快速決策工具")
    
    # 側邊欄設定
    with st.sidebar:
        st.header("📌 分析設定")
        
        # 股票選擇
        market = st.radio("選擇市場", ["美股", "台股", "港股", "陸股"])
        
        # 預設股票列表
        stock_dict = {
            "美股": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"],
            "台股": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2412.TW"],
            "港股": ["0700.HK", "0005.HK", "0939.HK", "0941.HK", "1299.HK"],
            "陸股": ["BABA", "BIDU", "JD", "PDD", "NIO"]
        }
        
        # 股票輸入
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol = st.selectbox(
                "選擇股票",
                stock_dict.get(market, []),
                help="選擇預設股票或輸入自訂代碼"
            )
        with col2:
            custom_symbol = st.text_input("自訂", "")
            
        if custom_symbol:
            symbol = custom_symbol
            
        # 分析期間
        period_options = {
            "1週": "5d",
            "1個月": "1mo",
            "3個月": "3mo",
            "6個月": "6mo",
            "1年": "1y",
            "2年": "2y",
            "5年": "5y"
        }
        
        period_text = st.select_slider(
            "分析期間",
            options=list(period_options.keys()),
            value="1年"
        )
        period = period_options[period_text]
        
        # 技術指標選擇
        st.markdown("### 📈 技術指標")
        show_ma = st.checkbox("移動平均線 (MA)", value=True)
        show_volume = st.checkbox("成交量", value=True)
        show_rsi = st.checkbox("RSI", value=True)
        show_macd = st.checkbox("MACD", value=False)
        show_bb = st.checkbox("布林通道", value=False)
        
        # 分析按鈕
        analyze_button = st.button("🔍 開始分析", type="primary", use_container_width=True)
        
        # 除錯按鈕
        st.markdown("---")
        debug_mode = st.checkbox("🔍 除錯模式", value=False)
    
    # 主要內容區
    if analyze_button or st.session_state.get('analyzed', False):
        st.session_state['analyzed'] = True
        
        try:
            # 建立分析器
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text('正在載入股票資訊...')
            progress_bar.progress(20)
            
            analyzer = StockAnalyzer(symbol)
            
            status_text.text('正在更新歷史數據...')
            progress_bar.progress(50)
            
            # 更新指定期間的數據
            if period != "1y":
                analyzer.hist = get_stock_data(symbol, period)
            
            status_text.text('正在計算指標...')
            progress_bar.progress(80)
            
            # 清除進度条
            progress_bar.progress(100)
            status_text.text('分析完成！')
            time.sleep(0.5)  # 短暫顯示
            
            progress_bar.empty()
            status_text.empty()
                
            # 公司資訊
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                company_name = analyzer.info.get('longName', symbol)
                st.markdown(f"## {company_name}")
                st.markdown(f"**產業:** {analyzer.info.get('industry', 'N/A')} | "
                          f"**部門:** {analyzer.info.get('sector', 'N/A')}")
            
            # 除錯資訊顯示
            if debug_mode:
                st.markdown("### 🔍 除錯資訊")
                debug_col1, debug_col2 = st.columns(2)
                
                with debug_col1:
                    st.markdown("**原始 info 數據:**")
                    st.write(f"trailingPE: {analyzer.info.get('trailingPE')}")
                    st.write(f"priceToBook: {analyzer.info.get('priceToBook')}")
                    st.write(f"marketCap: {analyzer.info.get('marketCap')}")
                    st.write(f"returnOnEquity: {analyzer.info.get('returnOnEquity')}")
                    st.write(f"dividendYield: {analyzer.info.get('dividendYield')}")
                    st.write(f"sharesOutstanding: {analyzer.info.get('sharesOutstanding')}")
                
                with debug_col2:
                    st.markdown("**歷史數據狀態:**")
                    st.write(f"歷史數據長度: {len(analyzer.hist)}")
                    if not analyzer.hist.empty:
                        st.write(f"最新價格: {analyzer.hist['Close'].iloc[-1]}")
                        st.write(f"數據範圍: {analyzer.hist.index[0]} 到 {analyzer.hist.index[-1]}")
                    
                    ratios = analyzer.calculate_financial_ratios()
                    st.markdown("**計算出的比率:**")
                    for key, value in ratios.items():
                        st.write(f"{key}: {value}")
            
            # 關鍵指標卡片
            st.markdown("### 📊 關鍵指標")
            
            # 先確保有歷史數據
            if len(analyzer.hist) == 0:
                st.warning("無法取得股價數據，請檢查股票代碼")
                return
                
            # 先計算財務比率
            try:
                ratios = analyzer.calculate_financial_ratios()
            except Exception as e:
                st.warning(f"計算財務比率時發生錯誤: {e}")
                ratios = {}
            
            # 建立六個欄位 - 強制顯示版本
            metrics_cols = st.columns(6)
            
            # 指標 1: 現價
            with metrics_cols[0]:
                try:
                    current_price = float(analyzer.hist['Close'].iloc[-1])
                    if len(analyzer.hist) > 1:
                        prev_close = float(analyzer.hist['Close'].iloc[-2])
                        price_change_pct = ((current_price - prev_close) / prev_close) * 100
                    else:
                        price_change_pct = 0.0
                    
                    st.metric(
                        "現價",
                        f"${current_price:.2f}",
                        f"{price_change_pct:+.2f}%"
                    )
                    if debug_mode:
                        st.caption(f"Debug: 價格={current_price}")
                except Exception as e:
                    st.metric("現價", "錯誤", "0.00%")
                    if debug_mode:
                        st.caption(f"錯誤: {e}")
            
            # 指標 2: 本益比
            with metrics_cols[1]:
                try:
                    pe_ratio = analyzer.info.get('trailingPE')
                    if pe_ratio and pe_ratio > 0:
                        st.metric("本益比 (P/E)", f"{float(pe_ratio):.2f}")
                        if debug_mode:
                            st.caption(f"Debug: PE={pe_ratio}")
                    else:
                        st.metric("本益比 (P/E)", "N/A")
                        if debug_mode:
                            st.caption(f"Debug: PE為空或≤0")
                except Exception as e:
                    st.metric("本益比 (P/E)", "錯誤")
                    if debug_mode:
                        st.caption(f"錯誤: {e}")
            
            # 指標 3: 股價淨值比
            with metrics_cols[2]:
                try:
                    pb_ratio = analyzer.info.get('priceToBook')
                    if pb_ratio and pb_ratio > 0:
                        st.metric("股價淨值比 (P/B)", f"{float(pb_ratio):.2f}")
                        if debug_mode:
                            st.caption(f"Debug: PB={pb_ratio}")
                    else:
                        st.metric("股價淨值比 (P/B)", "N/A")
                        if debug_mode:
                            st.caption(f"Debug: PB為空或≤0")
                except Exception as e:
                    st.metric("股價淨值比 (P/B)", "錯誤")
                    if debug_mode:
                        st.caption(f"錯誤: {e}")
            
            # 指標 4: ROE
            with metrics_cols[3]:
                try:
                    roe_info = analyzer.info.get('returnOnEquity')
                    if roe_info and roe_info > 0:
                        st.metric("ROE", f"{float(roe_info)*100:.1f}%")
                        if debug_mode:
                            st.caption(f"Debug: ROE={roe_info}")
                    else:
                        roe_calc = ratios.get('ROE')
                        if roe_calc and not pd.isna(roe_calc):
                            st.metric("ROE", f"{float(roe_calc):.1f}%")
                            if debug_mode:
                                st.caption(f"Debug: 計算ROE={roe_calc}")
                        else:
                            st.metric("ROE", "N/A")
                            if debug_mode:
                                st.caption(f"Debug: ROE無法取得")
                except Exception as e:
                    st.metric("ROE", "錯誤")
                    if debug_mode:
                        st.caption(f"錯誤: {e}")
            
            # 指標 5: 股息率
            with metrics_cols[4]:
                try:
                    dividend_yield = analyzer.info.get('dividendYield')
                    if dividend_yield and dividend_yield > 0:
                        st.metric("股息率", f"{float(dividend_yield)*100:.2f}%")
                        if debug_mode:
                            st.caption(f"Debug: 股息率={dividend_yield}")
                    else:
                        trailing_yield = analyzer.info.get('trailingAnnualDividendYield')
                        if trailing_yield and trailing_yield > 0:
                            st.metric("股息率", f"{float(trailing_yield)*100:.2f}%")
                            if debug_mode:
                                st.caption(f"Debug: 年股息率={trailing_yield}")
                        else:
                            st.metric("股息率", "0.00%")
                            if debug_mode:
                                st.caption(f"Debug: 無股息")
                except Exception as e:
                    st.metric("股息率", "錯誤")
                    if debug_mode:
                        st.caption(f"錯誤: {e}")
            
            # 指標 6: 市值
            with metrics_cols[5]:
                try:
                    market_cap = analyzer.info.get('marketCap')
                    if market_cap and market_cap > 0:
                        market_cap = float(market_cap)
                        if market_cap >= 1e12:
                            st.metric("市值", f"${market_cap/1e12:.2f}T")
                        elif market_cap >= 1e9:
                            st.metric("市值", f"${market_cap/1e9:.2f}B")
                        elif market_cap >= 1e6:
                            st.metric("市值", f"${market_cap/1e6:.2f}M")
                        else:
                            st.metric("市值", f"${market_cap:,.0f}")
                        if debug_mode:
                            st.caption(f"Debug: 市值={market_cap}")
                    else:
                        shares = analyzer.info.get('sharesOutstanding')
                        current_price = analyzer.hist['Close'].iloc[-1]
                        if shares and current_price:
                            calc_market_cap = float(shares) * float(current_price)
                            if calc_market_cap >= 1e9:
                                st.metric("市值", f"${calc_market_cap/1e9:.2f}B")
                            else:
                                st.metric("市值", f"${calc_market_cap/1e6:.2f}M")
                            if debug_mode:
                                st.caption(f"Debug: 計算市值={calc_market_cap}")
                        else:
                            st.metric("市值", "N/A")
                            if debug_mode:
                                st.caption(f"Debug: 無市值數據")
                except Exception as e:
                    st.metric("市值", "錯誤")
                    if debug_mode:
                        st.caption(f"錯誤: {e}")
            
            # 標籤頁
            tab1, tab2, tab3, tab4 = st.tabs(["📈 價格走勢", "🔧 技術分析", "💰 財務分析", "📊 詳細數據"])
            
            with tab1:
                # 價格走勢圖
                tech_df = analyzer.calculate_technical_indicators()
                
                if len(tech_df) > 0:
                    fig = make_subplots(
                        rows=2 if show_volume else 1,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3] if show_volume else [1],
                        subplot_titles=("價格走勢", "成交量") if show_volume else ("價格走勢",)
                    )
                    
                    # K線圖
                    fig.add_trace(
                        go.Candlestick(
                            x=tech_df.index,
                            open=tech_df['Open'],
                            high=tech_df['High'],
                            low=tech_df['Low'],
                            close=tech_df['Close'],
                            name="價格",
                            increasing_line_color='red',
                            decreasing_line_color='green'
                        ),
                        row=1, col=1
                    )
                    
                    # 移動平均線
                    if show_ma:
                        if 'MA5' in tech_df.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=tech_df.index,
                                    y=tech_df['MA5'],
                                    name="MA5",
                                    line=dict(color='orange', width=1)
                                ),
                                row=1, col=1
                            )
                        if 'MA20' in tech_df.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=tech_df.index,
                                    y=tech_df['MA20'],
                                    name="MA20",
                                    line=dict(color='blue', width=1)
                                ),
                                row=1, col=1
                            )
                        if 'MA60' in tech_df.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=tech_df.index,
                                    y=tech_df['MA60'],
                                    name="MA60",
                                    line=dict(color='purple', width=1)
                                ),
                                row=1, col=1
                            )
                    
                    # 成交量
                    if show_volume:
                        colors = ['red' if tech_df['Close'].iloc[i] >= tech_df['Open'].iloc[i] 
                                 else 'green' for i in range(len(tech_df))]
                        
                        fig.add_trace(
                            go.Bar(
                                x=tech_df.index,
                                y=tech_df['Volume'],
                                name="成交量",
                                marker_color=colors,
                                opacity=0.5
                            ),
                            row=2, col=1
                        )
                    
                    fig.update_xaxes(rangeslider_visible=False)
                    fig.update_layout(
                        title=f"{symbol} 價格走勢圖",
                        yaxis_title="價格",
                        template="plotly_white",
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("無法取得足夠的歷史數據")
            
            with tab2:
                st.markdown("### 技術指標詳細分析")
                st.info("技術指標功能正常運作中...")
            
            with tab3:
                st.markdown("### 💰 財務分析")
                st.info("財務分析功能正常運作中...")
            
            with tab4:
                st.markdown("### 📊 詳細數據")
                st.info("詳細數據功能正常運作中...")
                
        except Exception as e:
            st.error(f"分析過程中發生錯誤: {str(e)}")
            st.info("請檢查股票代碼是否正確，或稍後再試。")
    
    else:
        # 歡迎頁面
        st.markdown("""
        ### 👋 歡迎使用專業股票分析系統
        
        本系統提供：
        - 📊 即時股價與技術指標分析
        - 💰 完整財務比率計算
        - 📈 互動式圖表視覺化
        - 🔍 多市場股票支援
        
        **開始使用：**
        1. 在左側選擇市場和股票代碼
        2. 設定分析期間和技術指標
        3. 點擊「開始分析」按鈕
        
        ---
        💡 **提示：** 可以輸入自訂股票代碼進行分析
        """)

if __name__ == "__main__":
    main()
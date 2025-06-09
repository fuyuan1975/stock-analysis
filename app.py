# app.py - 完整增強版股票分析系統
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import requests
from typing import Dict, Optional, Tuple
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
        border-left: 4px solid #1f77b4;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
    }
    .metric-positive {
        color: #00C853;
    }
    .metric-negative {
        color: #FF1744;
    }
    .analysis-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 快取裝飾器
@st.cache_data(ttl=300)  # 快取5分鐘
def get_stock_data(symbol: str, period: str = "1y"):
    """快取股票數據獲取"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"獲取股票數據失敗: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)  # 快取10分鐘
def get_stock_info(symbol: str):
    """快取股票基本資訊"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return info if info else {}
    except Exception as e:
        st.error(f"獲取股票資訊失敗: {e}")
        return {}

@st.cache_data(ttl=1800)  # 快取30分鐘
def get_financial_data(symbol: str):
    """獲取財務報表數據"""
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        return financials, balance_sheet, cashflow
    except Exception as e:
        st.warning(f"獲取財務數據時發生錯誤: {e}")
        return None, None, None

# 增強版股票分析器
class EnhancedStockAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.stock = yf.Ticker(self.symbol)
        self._load_data()
    
    def _load_data(self):
        """載入所需資料"""
        # 使用快取功能提升效能
        self.info = get_stock_info(self.symbol)
        
        # 載入歷史數據
        self.hist = get_stock_data(self.symbol, "1y")
        if self.hist.empty:
            self.hist = get_stock_data(self.symbol, "3mo")
            if self.hist.empty:
                self.hist = get_stock_data(self.symbol, "1mo")
        
        # 載入財務資料
        self.financials, self.balance_sheet, self.cashflow = get_financial_data(self.symbol)
        
        # 計算基本統計數據
        if not self.hist.empty:
            self._calculate_basic_stats()
    
    def _calculate_basic_stats(self):
        """計算基本統計數據"""
        if len(self.hist) > 0:
            self.current_price = float(self.hist['Close'].iloc[-1])
            
            if len(self.hist) > 1:
                prev_close = float(self.hist['Close'].iloc[-2])
                self.daily_change = self.current_price - prev_close
                self.daily_change_pct = (self.daily_change / prev_close) * 100
            else:
                self.daily_change = 0
                self.daily_change_pct = 0
                
            # 週期性變化
            if len(self.hist) >= 5:
                week_ago_price = float(self.hist['Close'].iloc[-5])
                self.weekly_change_pct = ((self.current_price - week_ago_price) / week_ago_price) * 100
            else:
                self.weekly_change_pct = 0
                
            if len(self.hist) >= 20:
                month_ago_price = float(self.hist['Close'].iloc[-20])
                self.monthly_change_pct = ((self.current_price - month_ago_price) / month_ago_price) * 100
            else:
                self.monthly_change_pct = 0
                
            # 價格統計
            self.high_52w = float(self.hist['High'].max())
            self.low_52w = float(self.hist['Low'].min())
            self.avg_volume = float(self.hist['Volume'].mean())
            
    def get_enhanced_metrics(self) -> Dict:
        """獲取增強版指標"""
        metrics = {}
        
        if self.hist.empty:
            return metrics
            
        # 基本價格指標
        metrics['current_price'] = getattr(self, 'current_price', 0)
        metrics['daily_change'] = getattr(self, 'daily_change', 0)
        metrics['daily_change_pct'] = getattr(self, 'daily_change_pct', 0)
        metrics['weekly_change_pct'] = getattr(self, 'weekly_change_pct', 0)
        metrics['monthly_change_pct'] = getattr(self, 'monthly_change_pct', 0)
        
        # 市場指標 - 多重來源嘗試
        metrics['pe_ratio'] = self._get_safe_value(
            [
                self.info.get('trailingPE'),
                self.info.get('forwardPE'),
                self._calculate_pe_ratio()
            ]
        )
        
        metrics['pb_ratio'] = self._get_safe_value(
            [
                self.info.get('priceToBook'),
                self._calculate_pb_ratio()
            ]
        )
        
        # 市值計算
        market_cap = self.info.get('marketCap')
        if not market_cap:
            shares = self.info.get('sharesOutstanding', self.info.get('impliedSharesOutstanding'))
            if shares and hasattr(self, 'current_price'):
                market_cap = shares * self.current_price
        metrics['market_cap'] = market_cap
        
        # ROE指標
        metrics['roe'] = self._get_safe_value(
            [
                self.info.get('returnOnEquity'),
                self._calculate_roe()
            ]
        )
        
        # 股息率
        metrics['dividend_yield'] = self._get_safe_value(
            [
                self.info.get('dividendYield'),
                self.info.get('trailingAnnualDividendYield'),
                self.info.get('fiveYearAvgDividendYield')
            ]
        )
        
        # 其他重要指標
        metrics['beta'] = self.info.get('beta')
        metrics['eps'] = self.info.get('trailingEps', self.info.get('forwardEps'))
        metrics['revenue_growth'] = self.info.get('revenueGrowth')
        metrics['profit_margin'] = self.info.get('profitMargins')
        
        return metrics
    
    def _get_safe_value(self, value_list):
        """安全獲取數值"""
        for value in value_list:
            if value is not None and not pd.isna(value) and value != 0:
                return float(value)
        return None
    
    def _calculate_pe_ratio(self):
        """計算本益比"""
        try:
            eps = self.info.get('trailingEps')
            if eps and hasattr(self, 'current_price') and eps > 0:
                return self.current_price / eps
        except:
            pass
        return None
    
    def _calculate_pb_ratio(self):
        """計算股價淨值比"""
        try:
            book_value = self.info.get('bookValue')
            if book_value and hasattr(self, 'current_price') and book_value > 0:
                return self.current_price / book_value
        except:
            pass
        return None
    
    def _calculate_roe(self):
        """計算ROE"""
        try:
            if self.financials is not None and self.balance_sheet is not None:
                net_income = self._get_financial_value(self.financials, 
                    ['Net Income', 'Net Income Common Stockholders'])
                shareholders_equity = self._get_financial_value(self.balance_sheet,
                    ['Total Stockholder Equity', 'Stockholders Equity'])
                
                if net_income and shareholders_equity and shareholders_equity != 0:
                    return (net_income / shareholders_equity) * 100
        except:
            pass
        return None
    
    def _get_financial_value(self, df, field_names):
        """從財務報表獲取數值"""
        if df is None or df.empty:
            return None
            
        for field in field_names:
            if field in df.index:
                try:
                    value = df.loc[field].iloc[0]
                    if pd.notna(value) and value != 0:
                        return float(value)
                except:
                    continue
        return None
    
    def calculate_technical_indicators(self) -> pd.DataFrame:
        """計算技術指標"""
        if self.hist.empty:
            return pd.DataFrame()
            
        df = self.hist.copy()
        
        try:
            # 移動平均
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
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
            
            # 成交量移動平均
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            
        except Exception as e:
            st.warning(f"計算技術指標時發生錯誤: {e}")
            
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """計算 RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, 0.0001)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """計算 MACD"""
        try:
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line
        except:
            return pd.Series(index=prices.index, dtype=float), pd.Series(index=prices.index, dtype=float)
    
    def get_analysis_summary(self) -> Dict:
        """獲取分析摘要"""
        summary = {
            'trend': '中性',
            'strength': '一般',
            'recommendation': '觀望',
            'key_points': []
        }
        
        if self.hist.empty:
            return summary
            
        try:
            tech_df = self.calculate_technical_indicators()
            metrics = self.get_enhanced_metrics()
            
            # 趨勢分析
            if hasattr(self, 'daily_change_pct'):
                if self.daily_change_pct > 2:
                    summary['trend'] = '強烈上漲'
                elif self.daily_change_pct > 0.5:
                    summary['trend'] = '上漲'
                elif self.daily_change_pct < -2:
                    summary['trend'] = '強烈下跌'
                elif self.daily_change_pct < -0.5:
                    summary['trend'] = '下跌'
            
            # RSI分析
            if 'RSI' in tech_df.columns and not tech_df['RSI'].empty:
                current_rsi = tech_df['RSI'].iloc[-1]
                if not pd.isna(current_rsi):
                    if current_rsi > 70:
                        summary['key_points'].append('RSI顯示超買狀態')
                    elif current_rsi < 30:
                        summary['key_points'].append('RSI顯示超賣狀態')
            
            # 移動平均分析
            if all(col in tech_df.columns for col in ['MA5', 'MA20']):
                current_price = tech_df['Close'].iloc[-1]
                ma5 = tech_df['MA5'].iloc[-1]
                ma20 = tech_df['MA20'].iloc[-1]
                
                if not any(pd.isna([current_price, ma5, ma20])):
                    if current_price > ma5 > ma20:
                        summary['key_points'].append('價格位於短期均線之上')
                    elif current_price < ma5 < ma20:
                        summary['key_points'].append('價格位於短期均線之下')
            
            # 估值分析
            pe_ratio = metrics.get('pe_ratio')
            if pe_ratio:
                if pe_ratio < 15:
                    summary['key_points'].append('本益比相對較低')
                elif pe_ratio > 25:
                    summary['key_points'].append('本益比相對較高')
                    
        except Exception as e:
            summary['key_points'].append(f'分析過程中發生錯誤: {str(e)}')
            
        return summary

# 輔助函數
def format_number(value, format_type="currency"):
    """格式化數字顯示"""
    if value is None or pd.isna(value):
        return "N/A"
    
    try:
        value = float(value)
        if format_type == "currency":
            if abs(value) >= 1e12:
                return f"${value/1e12:.2f}T"
            elif abs(value) >= 1e9:
                return f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.2f}M"
            elif abs(value) >= 1e3:
                return f"${value/1e3:.2f}K"
            else:
                return f"${value:.2f}"
        elif format_type == "percentage":
            return f"{value:.2f}%"
        elif format_type == "ratio":
            return f"{value:.2f}"
        else:
            return f"{value:,.2f}"
    except:
        return "錯誤"

def create_price_chart(analyzer, show_ma=True, show_volume=True, show_bb=False):
    """創建價格圖表"""
    tech_df = analyzer.calculate_technical_indicators()
    
    if len(tech_df) == 0:
        return None
    
    # 創建子圖
    rows = 2 if show_volume else 1
    fig = make_subplots(
        rows=rows,
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
            increasing_line_color='#FF6B6B',
            decreasing_line_color='#4ECDC4'
        ),
        row=1, col=1
    )
    
    # 移動平均線
    if show_ma:
        ma_lines = [
            ('MA5', '#FFA726', 'MA5日'),
            ('MA20', '#42A5F5', 'MA20日'),
            ('MA50', '#AB47BC', 'MA50日')
        ]
        
        for ma_col, color, name in ma_lines:
            if ma_col in tech_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=tech_df.index,
                        y=tech_df[ma_col],
                        name=name,
                        line=dict(color=color, width=1.5),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
    
    # 布林通道
    if show_bb and all(col in tech_df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(
                x=tech_df.index,
                y=tech_df['BB_Upper'],
                name="布林上軌",
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=tech_df.index,
                y=tech_df['BB_Lower'],
                name="布林下軌",
                line=dict(color='gray', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 成交量
    if show_volume:
        colors = ['#FF6B6B' if tech_df['Close'].iloc[i] >= tech_df['Open'].iloc[i] 
                 else '#4ECDC4' for i in range(len(tech_df))]
        
        fig.add_trace(
            go.Bar(
                x=tech_df.index,
                y=tech_df['Volume'],
                name="成交量",
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        
        # 成交量移動平均
        if 'Volume_MA' in tech_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_df.index,
                    y=tech_df['Volume_MA'],
                    name="成交量MA",
                    line=dict(color='orange', width=2),
                    opacity=0.8
                ),
                row=2, col=1
            )
    
    # 圖表設定
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        title=f"{analyzer.symbol} 技術分析圖表",
        template="plotly_white",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_technical_indicators_chart(analyzer):
    """創建技術指標圖表"""
    tech_df = analyzer.calculate_technical_indicators()
    
    if len(tech_df) == 0:
        return None
    
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("RSI", "MACD")
    )
    
    # RSI
    if 'RSI' in tech_df.columns:
        fig.add_trace(
            go.Scatter(
                x=tech_df.index,
                y=tech_df['RSI'],
                name="RSI",
                line=dict(color='purple', width=2)
            ),
            row=1, col=1
        )
        
        # RSI 參考線
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)
    
    # MACD
    if all(col in tech_df.columns for col in ['MACD', 'Signal', 'MACD_Histogram']):
        fig.add_trace(
            go.Scatter(
                x=tech_df.index,
                y=tech_df['MACD'],
                name="MACD",
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=tech_df.index,
                y=tech_df['Signal'],
                name="Signal",
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # MACD 柱狀圖
        colors = ['green' if val >= 0 else 'red' for val in tech_df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=tech_df.index,
                y=tech_df['MACD_Histogram'],
                name="MACD Histogram",
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title=f"{analyzer.symbol} 技術指標",
        template="plotly_white",
        height=500,
        showlegend=True
    )
    
    return fig

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
            "美股": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "CRM"],
            "台股": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2412.TW", "2881.TW"],
            "港股": ["0700.HK", "0005.HK", "0939.HK", "0941.HK", "1299.HK", "0388.HK"],
            "陸股": ["BABA", "BIDU", "JD", "PDD", "NIO", "XPEV", "LI"]
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
            symbol = custom_symbol.upper()
            
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
        
        # 清除快取按鈕
        if st.button("🔄 清除快取"):
            st.cache_data.clear()
            st.success("快取已清除！")
    
    # 主要內容區
    if analyze_button or st.session_state.get('analyzed', False):
        st.session_state['analyzed'] = True
        
        try:
            # 進度顯示
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text('正在載入股票資訊...')
                progress_bar.progress(20)
                
                # 建立分析器
                analyzer = EnhancedStockAnalyzer(symbol)
                
                status_text.text('正在更新歷史數據...')
                progress_bar.progress(50)
                
                # 更新指定期間的數據
                if period != "1y":
                    analyzer.hist = get_stock_data(symbol, period)
                    analyzer._calculate_basic_stats()
                
                status_text.text('正在計算指標...')
                progress_bar.progress(80)
                
                # 獲取指標
                metrics = analyzer.get_enhanced_metrics()
                analysis_summary = analyzer.get_analysis_summary()
                
                progress_bar.progress(100)
                status_text.text('分析完成！')
                time.sleep(0.5)
                
                # 清除進度條
                progress_container.empty()
            
            # 檢查數據有效性
            if analyzer.hist.empty:
                st.error(f"無法取得股票 {symbol} 的數據，請檢查股票代碼是否正確。")
                return
            
            # 公司資訊標題
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                company_name = analyzer.info.get('longName', symbol)
                st.markdown(f"## {company_name}")
                industry = analyzer.info.get('industry', 'N/A')
                sector = analyzer.info.get('sector', 'N/A')
                st.markdown(f"**產業:** {industry} | **部門:** {sector}")
            
            with col2:
                # 趨勢指示器
                trend = analysis_summary.get('trend', '中性')
                if '上漲' in trend:
                    st.markdown(f'<p style="color: green; font-size: 20px;">📈 {trend}</p>', 
                              unsafe_allow_html=True)
                elif '下跌' in trend:
                    st.markdown(f'<p style="color: red; font-size: 20px;">📉 {trend}</p>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<p style="color: gray; font-size: 20px;">➡️ {trend}</p>', 
                              unsafe_allow_html=True)
            
            with col3:
                # 建議指示器
                recommendation = analysis_summary.get('recommendation', '觀望')
                if recommendation == '買入':
                    st.markdown(f'<p style="color: green; font-size: 18px;">💡 建議: {recommendation}</p>', 
                              unsafe_allow_html=True)
                elif recommendation == '賣出':
                    st.markdown(f'<p style="color: red; font-size: 18px;">💡 建議: {recommendation}</p>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<p style="color: orange; font-size: 18px;">💡 建議: {recommendation}</p>', 
                              unsafe_allow_html=True)
            
            # 除錯資訊顯示
            if debug_mode:
                with st.expander("🔍 除錯資訊", expanded=False):
                    debug_col1, debug_col2 = st.columns(2)
                    
                    with debug_col1:
                        st.markdown("**原始 info 數據:**")
                        key_info = {
                            'trailingPE': analyzer.info.get('trailingPE'),
                            'priceToBook': analyzer.info.get('priceToBook'),
                            'marketCap': analyzer.info.get('marketCap'),
                            'returnOnEquity': analyzer.info.get('returnOnEquity'),
                            'dividendYield': analyzer.info.get('dividendYield'),
                            'sharesOutstanding': analyzer.info.get('sharesOutstanding'),
                            'trailingEps': analyzer.info.get('trailingEps'),
                            'bookValue': analyzer.info.get('bookValue')
                        }
                        for key, value in key_info.items():
                            st.write(f"{key}: {value}")
                    
                    with debug_col2:
                        st.markdown("**歷史數據狀態:**")
                        st.write(f"歷史數據長度: {len(analyzer.hist)}")
                        if not analyzer.hist.empty:
                            st.write(f"最新價格: {analyzer.hist['Close'].iloc[-1]:.2f}")
                            st.write(f"數據範圍: {analyzer.hist.index[0].date()} 到 {analyzer.hist.index[-1].date()}")
                        
                        st.markdown("**計算出的指標:**")
                        for key, value in metrics.items():
                            st.write(f"{key}: {value}")
            
            # 關鍵指標卡片
            st.markdown("### 📊 關鍵指標")
            
            # 建立六個欄位的指標顯示
            metrics_cols = st.columns(6)
            
            # 指標 1: 現價
            with metrics_cols[0]:
                current_price = metrics.get('current_price', 0)
                daily_change_pct = metrics.get('daily_change_pct', 0)
                if current_price > 0:
                    st.metric(
                        "現價",
                        f"${current_price:.2f}",
                        f"{daily_change_pct:+.2f}%"
                    )
                else:
                    st.metric("現價", "N/A", "0.00%")
            
            # 指標 2: 本益比
            with metrics_cols[1]:
                pe_ratio = metrics.get('pe_ratio')
                if pe_ratio and pe_ratio > 0:
                    st.metric("本益比 (P/E)", f"{pe_ratio:.2f}")
                else:
                    st.metric("本益比 (P/E)", "N/A")
            
            # 指標 3: 股價淨值比
            with metrics_cols[2]:
                pb_ratio = metrics.get('pb_ratio')
                if pb_ratio and pb_ratio > 0:
                    st.metric("股價淨值比 (P/B)", f"{pb_ratio:.2f}")
                else:
                    st.metric("股價淨值比 (P/B)", "N/A")
            
            # 指標 4: ROE
            with metrics_cols[3]:
                roe = metrics.get('roe')
                if roe is not None:
                    if isinstance(roe, float) and roe < 1:  # 如果是小數形式
                        st.metric("ROE", f"{roe*100:.1f}%")
                    else:  # 如果已經是百分比形式
                        st.metric("ROE", f"{roe:.1f}%")
                else:
                    st.metric("ROE", "N/A")
            
            # 指標 5: 股息率
            with metrics_cols[4]:
                dividend_yield = metrics.get('dividend_yield')
                if dividend_yield is not None and dividend_yield > 0:
                    if dividend_yield < 1:  # 小數形式
                        st.metric("股息率", f"{dividend_yield*100:.2f}%")
                    else:  # 百分比形式
                        st.metric("股息率", f"{dividend_yield:.2f}%")
                else:
                    st.metric("股息率", "0.00%")
            
            # 指標 6: 市值
            with metrics_cols[5]:
                market_cap = metrics.get('market_cap')
                st.metric("市值", format_number(market_cap, "currency"))
            
            # 額外指標行
            st.markdown("#### 📈 週期表現")
            perf_cols = st.columns(4)
            
            with perf_cols[0]:
                weekly_change = metrics.get('weekly_change_pct', 0)
                st.metric("週變化", f"{weekly_change:+.2f}%")
            
            with perf_cols[1]:
                monthly_change = metrics.get('monthly_change_pct', 0)
                st.metric("月變化", f"{monthly_change:+.2f}%")
            
            with perf_cols[2]:
                beta = metrics.get('beta')
                if beta:
                    st.metric("Beta係數", f"{beta:.2f}")
                else:
                    st.metric("Beta係數", "N/A")
            
            with perf_cols[3]:
                profit_margin = metrics.get('profit_margin')
                if profit_margin:
                    if profit_margin < 1:
                        st.metric("利潤率", f"{profit_margin*100:.1f}%")
                    else:
                        st.metric("利潤率", f"{profit_margin:.1f}%")
                else:
                    st.metric("利潤率", "N/A")
            
            # 分析摘要卡片
            if analysis_summary.get('key_points'):
                st.markdown("### 💡 分析要點")
                for point in analysis_summary['key_points']:
                    st.markdown(f"• {point}")
            
            # 標籤頁
            tab1, tab2, tab3, tab4 = st.tabs(["📈 價格走勢", "🔧 技術分析", "💰 財務分析", "📊 詳細數據"])
            
            with tab1:
                st.markdown("#### 價格與成交量分析")
                
                # 創建價格圖表
                price_chart = create_price_chart(analyzer, show_ma, show_volume, show_bb)
                if price_chart:
                    st.plotly_chart(price_chart, use_container_width=True)
                else:
                    st.warning("無法創建價格圖表")
                
                # 價格統計
                if hasattr(analyzer, 'high_52w') and hasattr(analyzer, 'low_52w'):
                    st.markdown("#### 價格區間分析")
                    range_cols = st.columns(3)
                    
                    with range_cols[0]:
                        st.metric("52週最高", f"${analyzer.high_52w:.2f}")
                    
                    with range_cols[1]:
                        st.metric("52週最低", f"${analyzer.low_52w:.2f}")
                    
                    with range_cols[2]:
                        current_vs_high = ((current_price - analyzer.high_52w) / analyzer.high_52w) * 100
                        st.metric("距離高點", f"{current_vs_high:.1f}%")
            
            with tab2:
                st.markdown("#### 技術指標分析")
                
                if show_rsi or show_macd:
                    # 創建技術指標圖表
                    tech_chart = create_technical_indicators_chart(analyzer)
                    if tech_chart:
                        st.plotly_chart(tech_chart, use_container_width=True)
                
                # 技術指標數值
                tech_df = analyzer.calculate_technical_indicators()
                if not tech_df.empty:
                    st.markdown("#### 當前技術指標數值")
                    tech_cols = st.columns(4)
                    
                    with tech_cols[0]:
                        if 'RSI' in tech_df.columns:
                            current_rsi = tech_df['RSI'].iloc[-1]
                            if not pd.isna(current_rsi):
                                rsi_status = "超買" if current_rsi > 70 else "超賣" if current_rsi < 30 else "正常"
                                st.metric("RSI", f"{current_rsi:.1f}", rsi_status)
                            else:
                                st.metric("RSI", "N/A")
                        else:
                            st.metric("RSI", "N/A")
                    
                    with tech_cols[1]:
                        if 'MACD' in tech_df.columns:
                            current_macd = tech_df['MACD'].iloc[-1]
                            if not pd.isna(current_macd):
                                st.metric("MACD", f"{current_macd:.3f}")
                            else:
                                st.metric("MACD", "N/A")
                        else:
                            st.metric("MACD", "N/A")
                    
                    with tech_cols[2]:
                        if 'MA20' in tech_df.columns:
                            ma20 = tech_df['MA20'].iloc[-1]
                            if not pd.isna(ma20):
                                ma20_distance = ((current_price - ma20) / ma20) * 100
                                st.metric("MA20距離", f"{ma20_distance:+.1f}%")
                            else:
                                st.metric("MA20距離", "N/A")
                        else:
                            st.metric("MA20距離", "N/A")
                    
                    with tech_cols[3]:
                        if hasattr(analyzer, 'avg_volume'):
                            current_volume = tech_df['Volume'].iloc[-1]
                            volume_ratio = (current_volume / analyzer.avg_volume) if analyzer.avg_volume > 0 else 0
                            st.metric("成交量比率", f"{volume_ratio:.1f}x")
                        else:
                            st.metric("成交量比率", "N/A")
            
            with tab3:
                st.markdown("#### 財務健康度分析")
                
                # 基本財務指標
                fin_cols = st.columns(3)
                
                with fin_cols[0]:
                    st.markdown("**盈利能力**")
                    eps = metrics.get('eps')
                    if eps:
                        st.write(f"每股盈餘 (EPS): ${eps:.2f}")
                    else:
                        st.write("每股盈餘 (EPS): N/A")
                    
                    profit_margin = metrics.get('profit_margin')
                    if profit_margin:
                        margin_pct = profit_margin * 100 if profit_margin < 1 else profit_margin
                        st.write(f"利潤率: {margin_pct:.1f}%")
                    else:
                        st.write("利潤率: N/A")
                
                with fin_cols[1]:
                    st.markdown("**成長性**")
                    revenue_growth = metrics.get('revenue_growth')
                    if revenue_growth:
                        growth_pct = revenue_growth * 100 if revenue_growth < 1 else revenue_growth
                        st.write(f"營收成長率: {growth_pct:.1f}%")
                    else:
                        st.write("營收成長率: N/A")
                
                with fin_cols[2]:
                    st.markdown("**估值水準**")
                    pe_ratio = metrics.get('pe_ratio')
                    if pe_ratio:
                        if pe_ratio < 15:
                            valuation = "低估"
                        elif pe_ratio > 25:
                            valuation = "高估"
                        else:
                            valuation = "合理"
                        st.write(f"估值判斷: {valuation}")
                    else:
                        st.write("估值判斷: 無法評估")
                
                # 財務報表數據（如果有的話）
                if analyzer.financials is not None and not analyzer.financials.empty:
                    st.markdown("#### 財務報表摘要")
                    with st.expander("查看詳細財務數據"):
                        st.markdown("**損益表 (最近期間)**")
                        st.dataframe(analyzer.financials.head())
                        
                        if analyzer.balance_sheet is not None:
                            st.markdown("**資產負債表 (最近期間)**")
                            st.dataframe(analyzer.balance_sheet.head())
            
            with tab4:
                st.markdown("#### 完整數據總覽")
                
                # 股票基本資訊
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**公司基本資訊**")
                    basic_info = {
                        "公司全名": analyzer.info.get('longName', 'N/A'),
                        "產業": analyzer.info.get('industry', 'N/A'),
                        "部門": analyzer.info.get('sector', 'N/A'),
                        "國家": analyzer.info.get('country', 'N/A'),
                        "員工數": analyzer.info.get('fullTimeEmployees', 'N/A'),
                        "網站": analyzer.info.get('website', 'N/A')
                    }
                    
                    for key, value in basic_info.items():
                        st.write(f"**{key}:** {value}")
                
                with col2:
                    st.markdown("**市場數據**")
                    market_info = {
                        "交易所": analyzer.info.get('exchange', 'N/A'),
                        "貨幣": analyzer.info.get('currency', 'N/A'),
                        "時區": analyzer.info.get('timeZone', 'N/A'),
                        "52週高點": f"${analyzer.info.get('fiftyTwoWeekHigh', 0):.2f}" if analyzer.info.get('fiftyTwoWeekHigh') else 'N/A',
                        "52週低點": f"${analyzer.info.get('fiftyTwoWeekLow', 0):.2f}" if analyzer.info.get('fiftyTwoWeekLow') else 'N/A',
                        "平均成交量": f"{analyzer.info.get('averageVolume', 0):,}" if analyzer.info.get('averageVolume') else 'N/A'
                    }
                    
                    for key, value in market_info.items():
                        st.write(f"**{key}:** {value}")
                
                # 歷史數據表格
                if not analyzer.hist.empty:
                    st.markdown("#### 歷史價格數據 (最近20天)")
                    recent_data = analyzer.hist.tail(20).round(2)
                    st.dataframe(recent_data)
                
                # 下載數據按鈕
                if not analyzer.hist.empty:
                    csv = analyzer.hist.to_csv()
                    st.download_button(
                        label="📥 下載歷史數據 (CSV)",
                        data=csv,
                        file_name=f"{symbol}_historical_data.csv",
                        mime="text/csv"
                    )
                
        except Exception as e:
            st.error(f"分析過程中發生錯誤: {str(e)}")
            st.info("請檢查股票代碼是否正確，或稍後再試。")
            if debug_mode:
                st.exception(e)
    
    else:
        # 歡迎頁面
        st.markdown("""
        <div class="analysis-card">
        <h3>👋 歡迎使用專業股票分析系統</h3>
        
        <p><strong>本系統提供：</strong></p>
        <ul>
        <li>📊 即時股價與技術指標分析</li>
        <li>💰 完整財務比率計算</li>
        <li>📈 互動式圖表視覺化</li>
        <li>🔍 多市場股票支援</li>
        <li>🤖 智能分析摘要</li>
        </ul>
        
        <p><strong>開始使用：</strong></p>
        <ol>
        <li>在左側選擇市場和股票代碼</li>
        <li>設定分析期間和技術指標</li>
        <li>點擊「開始分析」按鈕</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # 示例股票
        st.markdown("### 🔥 熱門股票快速分析")
        sample_cols = st.columns(4)
        
        popular_stocks = ["AAPL", "TSLA", "GOOGL", "MSFT"]
        for i, stock in enumerate(popular_stocks):
            with sample_cols[i]:
                if st.button(f"分析 {stock}", key=f"sample_{stock}"):
                    st.session_state['analyzed'] = True
                    st.rerun()

if __name__ == "__main__":
    main()
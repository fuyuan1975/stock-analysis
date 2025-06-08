
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from datetime import datetime

# 頁面設定
st.set_page_config(page_title="專業股票分析系統", page_icon="📈", layout="wide")

# 股票代碼選單（可擴充）
stock_dict = {
    "台積電 (2330)": "2330.TW",
    "聯發科 (2454)": "2454.TW",
    "環球晶 (6488)": "6488.TWO",
    "鴻海 (2317)": "2317.TW",
    "長榮 (2603)": "2603.TW"
}

# 輸入選項
st.sidebar.title("股票設定")
selected_stock_name = st.sidebar.selectbox("選擇股票", list(stock_dict.keys()))
symbol = stock_dict[selected_stock_name]

# 分析區間
period_options = {
    "1 週": "7d",
    "1 個月": "1mo",
    "3 個月": "3mo",
    "半年": "6mo",
    "1 年": "1y",
    "5 年": "5y"
}
selected_period_label = st.sidebar.select_slider("分析期間", options=list(period_options.keys()), value="3 個月")
selected_period = period_options[selected_period_label]

# 技術指標選擇
st.sidebar.title("技術指標")
show_ma = st.sidebar.checkbox("移動平均線 (MA)", value=True)
show_volume = st.sidebar.checkbox("成交量", value=True)
show_rsi = st.sidebar.checkbox("RSI", value=False)

# 資料讀取
@st.cache_data(ttl=300)
def get_data(sym, period):
    try:
        data = yf.download(sym, period=period)
        return data
    except:
        return pd.DataFrame()

data = get_data(symbol, selected_period)

# 錯誤處理：若抓不到資料
if data.empty:
    st.error("⚠️ 無法取得該股票資料，請確認代碼是否正確或是否為上櫃股票。")
    st.stop()

st.title(f"{selected_stock_name} 股票分析")

# 計算技術指標
if show_ma:
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA60"] = data["Close"].rolling(window=60).mean()

if show_rsi:
    rsi = RSIIndicator(close=data["Close"], window=14)
    data["RSI"] = rsi.rsi()

# 繪圖
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="股價"
))

if show_ma:
    fig.add_trace(go.Scatter(x=data.index, y=data["MA5"], name="MA5"))
    fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=data.index, y=data["MA60"], name="MA60"))

fig.update_layout(title="股價走勢圖", xaxis_title="日期", yaxis_title="價格")

st.plotly_chart(fig, use_container_width=True)

# RSI 額外圖
if show_rsi:
    st.subheader("RSI 指標")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="orange")))
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
    fig_rsi.update_layout(yaxis_title="RSI 值")
    st.plotly_chart(fig_rsi, use_container_width=True)

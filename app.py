import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(
    page_title="Dự báo giá cổ phiếu bằng AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Giao diện ----------
st.markdown("""
<style>
    .main-title {font-size: 2rem; font-weight: 750; margin-bottom: .2rem;}
    .sub-title {color: #777; margin-bottom: 1.5rem;}
    div.stButton > button[kind="primary"] {
        background: #ff4b4b; border: 0; color: white; font-weight: 650;
        border-radius: 9px; min-height: 3rem;
    }
    div[data-testid="stMetric"] {
        background: #f7f8fb; border: 1px solid #e7e9ef;
        padding: 14px; border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

STOCKS = {
    "Vinamilk (VNM)": "VNM.VN",
    "FPT (FPT)": "FPT.VN",
    "Vietcombank (VCB)": "VCB.VN",
    "Hòa Phát (HPG)": "HPG.VN",
    "MBBank (MBB)": "MBB.VN",
    "Masan (MSN)": "MSN.VN",
    "Vingroup (VIC)": "VIC.VN",
    "Petrolimex (PLX)": "PLX.VN",
    "REE (REE)": "REE.VN",
    "Sabeco (SAB)": "SAB.VN",
    "VN-Index (^VNINDEX)": "^VNINDEX",
}

@st.cache_data(ttl=900, show_spinner=False)
def load_stock_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    # Yahoo Finance dùng ngày kết thúc không bao gồm end_date, nên cộng thêm một ngày.
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    if "Close" not in df.columns:
        return pd.DataFrame()
    df = df[["Date", "Close"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna().sort_values("Date").reset_index(drop=True)

def build_model(lookback: int, units: int, dropout: float):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout),
        LSTM(max(16, units // 2), return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def make_sequences(values: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i, 0])
        y.append(values[i, 0])
    return np.array(X), np.array(y)

def train_and_forecast(df: pd.DataFrame, forecast_days: int, epochs: int,
                       batch_size: int, lookback: int, units: int, dropout: float):
    prices = df["Close"].values.astype("float32").reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = make_sequences(scaled, lookback)
    if len(X) < 30:
        raise ValueError("Không đủ dữ liệu để huấn luyện. Hãy chọn khoảng thời gian dài hơn.")

    X = X.reshape((X.shape[0], X.shape[1], 1))
    split = max(int(len(X) * 0.8), 1)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model(lookback, units, dropout)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": batch_size,
        "shuffle": False,
        "verbose": 0,
        "callbacks": [early_stop],
    }
    if len(X_test) >= 5:
        fit_kwargs["validation_data"] = (X_test, y_test)
    with st.spinner("Đang huấn luyện mô hình LSTM..."):
        model.fit(**fit_kwargs)

    predicted_test = model.predict(X_test, verbose=0) if len(X_test) else np.array([])
    actual_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten() if len(y_test) else np.array([])
    predicted_test = scaler.inverse_transform(predicted_test).flatten() if len(predicted_test) else np.array([])

    # Dự báo đệ quy: dùng các giá trị dự báo gần nhất cho phiên tiếp theo.
    window = scaled[-lookback:].copy()
    future_scaled = []
    for _ in range(forecast_days):
        x_input = window.reshape(1, lookback, 1)
        next_value = float(model.predict(x_input, verbose=0)[0, 0])
        future_scaled.append(next_value)
        window = np.vstack([window[1:], [[next_value]]])

    future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({"Ngày dự báo": future_dates, "Giá dự báo": future_prices})

    test_dates = pd.to_datetime(df["Date"].iloc[lookback + split:lookback + split + len(actual_test)]).reset_index(drop=True)
    evaluation = None
    if len(actual_test) > 0:
        evaluation = {
            "mae": float(mean_absolute_error(actual_test, predicted_test)),
            "rmse": float(np.sqrt(mean_squared_error(actual_test, predicted_test))),
            "dates": test_dates,
            "actual": actual_test,
            "predicted": predicted_test,
        }
    return model, forecast_df, evaluation

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ CẤU HÌNH DỰ BÁO")
    selected_stock = st.selectbox("Chọn cổ phiếu", list(STOCKS.keys()), index=0)
    ticker = STOCKS[selected_stock]
    default_start = date(2018, 1, 1)
    default_end = date.today()
    start_date = st.date_input("Ngày bắt đầu", value=default_start, max_value=date.today())
    end_date = st.date_input("Ngày kết thúc", value=default_end, min_value=start_date, max_value=date.today())
    forecast_days = st.slider("Số phiên muốn dự báo", min_value=1, max_value=30, value=7)
    epochs = st.slider("Số Epochs", min_value=5, max_value=50, value=10, step=5)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    lookback = st.slider("Số phiên lịch sử đưa vào mô hình", min_value=20, max_value=120, value=60, step=10)
    units = st.select_slider("Độ phức tạp mô hình (LSTM units)", options=[32, 50, 64, 96, 128], value=50)
    run_button = st.button("🚀 BẮT ĐẦU DỰ BÁO", type="primary", use_container_width=True)
    st.caption("Dữ liệu lấy từ Yahoo Finance. Mã cổ phiếu Việt Nam thường dùng hậu tố .VN.")
    st.caption("Kết quả chỉ phục vụ học tập/tham khảo, không phải khuyến nghị đầu tư.")

# ---------- Nội dung chính ----------
st.markdown('<div class="main-title">📈 Ứng dụng dự báo giá cổ phiếu bằng trí tuệ nhân tạo LSTM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Tải dữ liệu giá lịch sử, huấn luyện mô hình LSTM và ước lượng giá đóng cửa cho các phiên tiếp theo.</div>', unsafe_allow_html=True)
st.divider()

if start_date >= end_date:
    st.error("Ngày kết thúc phải sau ngày bắt đầu.")
    st.stop()

if run_button:
    try:
        with st.spinner(f"Đang tải dữ liệu {ticker} từ Yahoo Finance..."):
            df = load_stock_data(ticker, start_date, end_date)
        if df.empty:
            st.error("Không tải được dữ liệu. Hãy kiểm tra mã cổ phiếu hoặc thử lại sau. Một số mã có thể không có trên Yahoo Finance.")
            st.stop()
        if len(df) <= lookback + 35:
            st.error(f"Chỉ có {len(df)} phiên dữ liệu. Hãy chọn khoảng thời gian dài hơn hoặc giảm số phiên lịch sử xuống.")
            st.stop()

        st.session_state["stock_df"] = df
        st.session_state["stock_meta"] = {
            "stock": selected_stock, "ticker": ticker, "start": start_date,
            "end": end_date, "forecast_days": forecast_days,
            "epochs": epochs, "batch_size": batch_size, "lookback": lookback,
            "units": units,
        }
        with st.spinner("Đang huấn luyện và tạo dự báo..."):
            model, forecast_df, evaluation = train_and_forecast(
                df, forecast_days, epochs, batch_size, lookback, units, 0.2
            )
        st.session_state["forecast_df"] = forecast_df
        st.session_state["evaluation"] = evaluation
        st.session_state["model_ready"] = True
    except Exception as exc:
        st.error(f"Đã xảy ra lỗi khi dự báo: {exc}")
        st.info("Nếu lỗi liên quan đến tải dữ liệu, hãy thử lại sau. Nếu lỗi cài đặt TensorFlow, kiểm tra requirements.txt và phiên bản Python.")

if st.session_state.get("model_ready", False):
    df = st.session_state["stock_df"]
    forecast_df = st.session_state["forecast_df"]
    meta = st.session_state["stock_meta"]
    evaluation = st.session_state.get("evaluation")

    st.subheader(f"📊 Kết quả dự báo: {meta['stock']} ({meta['ticker']})")
    latest_price = float(df["Close"].iloc[-1])
    first_forecast = float(forecast_df["Giá dự báo"].iloc[0])
    change_pct = (first_forecast / latest_price - 1) * 100 if latest_price else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá đóng cửa gần nhất", f"{latest_price:,.2f}")
    c2.metric("Dự báo phiên kế tiếp", f"{first_forecast:,.2f}", f"{change_pct:+.2f}%")
    c3.metric("Số phiên dự báo", f"{len(forecast_df)}")
    c4.metric("Số phiên dữ liệu", f"{len(df):,}")

    tab1, tab2, tab3 = st.tabs(["📈 Biểu đồ", "🔮 Bảng dự báo", "🧪 Đánh giá mô hình"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Giá đóng cửa lịch sử",
                                 line=dict(color="#2563eb", width=2)))
        fig.add_trace(go.Scatter(x=forecast_df["Ngày dự báo"], y=forecast_df["Giá dự báo"],
                                 name="Giá LSTM dự báo", line=dict(color="#ff4b4b", width=3, dash="dash"),
                                 mode="lines+markers"))
        fig.add_trace(go.Scatter(
            x=[df["Date"].iloc[-1], forecast_df["Ngày dự báo"].iloc[0]],
            y=[latest_price, first_forecast], name="Nối điểm",
            line=dict(color="#ff4b4b", width=1, dash="dot"), showlegend=False
        ))
        fig.update_layout(
            title="Giá đóng cửa lịch sử và giá dự báo",
            xaxis_title="Ngày", yaxis_title="Giá (đơn vị theo nguồn dữ liệu)",
            hovermode="x unified", template="plotly_white", height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lưu ý: giá dự báo là kết quả mô hình học từ dữ liệu quá khứ; không thể phản ánh đầy đủ tin tức, sự kiện hoặc biến động bất ngờ.")
    with tab2:
        display_forecast = forecast_df.copy()
        display_forecast["Ngày dự báo"] = display_forecast["Ngày dự báo"].dt.strftime("%d/%m/%Y")
        display_forecast["Giá dự báo"] = display_forecast["Giá dự báo"].round(2)
        st.dataframe(display_forecast, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Tải bảng dự báo CSV",
            data=display_forecast.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"du_bao_{meta['ticker'].replace('^','')}.csv",
            mime="text/csv",
        )
    with tab3:
        st.markdown("Mô hình được đánh giá trên phần dữ liệu cuối chuỗi, không dùng để đảm bảo độ chính xác trong tương lai.")
        if evaluation:
            a, b = st.columns(2)
            a.metric("MAE", f"{evaluation['mae']:,.3f}")
            b.metric("RMSE", f"{evaluation['rmse']:,.3f}")
            eval_fig = go.Figure()
            eval_fig.add_trace(go.Scatter(x=evaluation["dates"], y=evaluation["actual"], name="Giá thực tế"))
            eval_fig.add_trace(go.Scatter(x=evaluation["dates"], y=evaluation["predicted"], name="Giá mô hình dự đoán"))
            eval_fig.update_layout(title="So sánh thực tế và dự đoán trên tập kiểm tra",
                                   xaxis_title="Ngày", yaxis_title="Giá", template="plotly_white", height=420)
            st.plotly_chart(eval_fig, use_container_width=True)
        else:
            st.warning("Chưa đủ dữ liệu kiểm tra để tính chỉ số đánh giá.")

    with st.expander("Xem dữ liệu lịch sử đã tải"):
        st.dataframe(df.tail(100).sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Tải dữ liệu lịch sử CSV",
                           data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"lich_su_{meta['ticker'].replace('^','')}.csv",
                           mime="text/csv")
else:
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("## 📊 Tổng quan ứng dụng")
        st.write("Ứng dụng dùng mạng nơ-ron hồi tiếp LSTM để học mối quan hệ trong chuỗi giá đóng cửa lịch sử và tạo dự báo cho các phiên tiếp theo.")
        st.info("Chọn mã cổ phiếu, thời gian dữ liệu và số phiên dự báo ở thanh bên trái, sau đó nhấn **BẮT ĐẦU DỰ BÁO**.")
        st.markdown("###  Quy trình hoạt động")
        st.markdown("""
        1. **Chọn mã cổ phiếu và khoảng thời gian.**
        2. **Tải dữ liệu** giá đóng cửa từ Yahoo Finance.
        3. **Chuẩn hóa dữ liệu** bằng MinMaxScaler.
        4. **Tạo chuỗi đầu vào** từ các phiên gần nhất.
        5. **Huấn luyện mô hình LSTM** trên dữ liệu lịch sử.
        6. **Dự báo đệ quy** giá cho các phiên tiếp theo và hiển thị biểu đồ.
        """)
    with right:
        st.markdown("### ⚠️ Lưu ý quan trọng")
        st.warning("Dự báo giá cổ phiếu có độ bất định cao. Mô hình minh họa này chỉ phục vụ học tập/nghiên cứu, không phải lời khuyên mua hoặc bán.")
        st.markdown("### 💡 Mẹo sử dụng")
        st.write("- Nên chọn dữ liệu lịch sử đủ dài.")
        st.write("- Có thể thử nhiều số Epochs và số phiên đầu vào để so sánh.")
        st.write("- Mã Việt Nam có thể tải không ổn định tùy khả năng truy cập Yahoo Finance.")


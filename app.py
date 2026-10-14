
# ============================================================
# UNG DUNG DU BAO GIA CO PHIEU - LSTM + STREAMLIT
# ============================================================

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------- CAU HINH --------------------

SEED = 42
WINDOW_SIZE = 60

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# -------------------- GIAO DIEN --------------------

st.title("📈 ỨNG DỤNG DỰ ĐOÁN GIÁ CỔ PHIẾU")
st.markdown(
    "Dự báo giá cổ phiếu bằng mô hình **LSTM (Deep Learning)** "
    "từ dữ liệu lịch sử."
)

st.info(
    "Mô hình phục vụ mục đích học tập và nghiên cứu. "
    "Kết quả dự báo không phải cam kết lợi nhuận hay khuyến nghị đầu tư."
)

# -------------------- TAI DU LIEU --------------------

@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df.empty:
        return pd.DataFrame()

    # Xu ly du lieu MultiIndex tu cac phien ban yfinance
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df.xs("Close", axis=1, level=0)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            return pd.DataFrame()
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df["Close"]

    result = pd.DataFrame({"Close": close})
    result.index = pd.to_datetime(result.index)
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()
    result["Close"] = pd.to_numeric(
        result["Close"], errors="coerce"
    )
    result = result.dropna()

    return result


# -------------------- TAO DU LIEU CHUOI --------------------

def create_sequences(data, window_size=WINDOW_SIZE):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.asarray(X), np.asarray(y)


# -------------------- XAY DUNG MO HINH --------------------

def build_model():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.1),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


def predict_future(model, scaled_data, scaler, steps=10):
    # Lay 60 phien gan nhat
    current_window = scaled_data[-WINDOW_SIZE:].reshape(
        1, WINDOW_SIZE, 1
    )

    predictions = []

    for _ in range(steps):
        next_value = model.predict(
            current_window, verbose=0
        )[0, 0]

        predictions.append(next_value)

        next_value_array = np.array(
            next_value
        ).reshape(1, 1, 1)

        current_window = np.concatenate(
            [current_window[:, 1:, :], next_value_array],
            axis=1
        )

    predictions = np.array(predictions).reshape(-1, 1)

    return scaler.inverse_transform(predictions).flatten()


# -------------------- NHAP THONG TIN --------------------

with st.sidebar:
    st.header("⚙️ Cài đặt dự báo")

    ticker = st.text_input(
        "Mã cổ phiếu Yahoo Finance",
        value="VNM.VN",
        help="Ví dụ: VNM.VN, FPT.VN, HPG.VN, VCB.VN"
    ).strip().upper()

    start_date = st.date_input(
        "Ngày bắt đầu lấy dữ liệu",
        value=date(2018, 1, 1),
        max_value=date.today()
    )

    forecast_days = st.slider(
        "Số phiên cần dự báo",
        min_value=1,
        max_value=30,
        value=10
    )

    epochs = st.slider(
        "Số epochs tối đa",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )

    run_button = st.button(
        "🚀 Chạy mô hình dự báo",
        type="primary",
        use_container_width=True
    )

    st.caption(
        "Mỗi lần chạy có thể cần vài phút tùy dữ liệu "
        "và tài nguyên máy chủ."
    )

# -------------------- CHAY DU BAO --------------------

if run_button:
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if start_date >= date.today():
        st.error("Ngày bắt đầu phải trước ngày hiện tại.")
        st.stop()

    end_date = date.today() + timedelta(days=1)

    try:
        with st.spinner("Đang tải dữ liệu cổ phiếu..."):
            df = download_stock_data(
                ticker,
                start_date.isoformat(),
                end_date.isoformat()
            )

        if df.empty:
            st.error(
                "Không tải được dữ liệu. Hãy kiểm tra mã cổ phiếu "
                "và thử lại, ví dụ VNM.VN hoặc FPT.VN."
            )
            st.stop()

        data = df[["Close"]].values.astype(np.float32)

        if len(data) < 150:
            st.error(
                f"Chỉ có {len(data)} phiên dữ liệu. "
                "Hãy chọn khoảng thời gian dài hơn để mô hình "
                "có đủ dữ liệu huấn luyện và kiểm tra."
            )
            st.stop()

        # Chia du lieu theo thu tu thoi gian
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]

        # Chi fit scaler tren tap train de tranh ro ri du lieu
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        X_train, y_train = create_sequences(train_scaled)

        test_combined = np.vstack([
            train_scaled[-WINDOW_SIZE:],
            test_scaled
        ])

        X_test, y_test = create_sequences(test_combined)

        # Chia validation tu cuoi tap train
        val_size = max(1, int(0.1 * len(X_train)))

        if len(X_train) <= val_size:
            st.error("Không đủ dữ liệu để chia tập huấn luyện.")
            st.stop()

        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]

        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        model = build_model()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        with st.spinner("Đang huấn luyện mô hình LSTM..."):
            history = model.fit(
                X_train_final,
                y_train_final,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

        with st.spinner("Đang tính toán kết quả dự báo..."):
            test_pred_scaled = model.predict(
                X_test, verbose=0
            )

            test_pred = scaler.inverse_transform(
                test_pred_scaled
            ).flatten()

            y_test_real = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()

            rmse = np.sqrt(
                mean_squared_error(y_test_real, test_pred)
            )
            mae = mean_absolute_error(y_test_real, test_pred)

            if len(y_test_real) > 1:
                r2 = r2_score(y_test_real, test_pred)
            else:
                r2 = float("nan")

            # Huan luyen tren tap train va du bao tu 60 phien cuoi
            all_scaled = np.vstack([train_scaled, test_scaled])

            future_predictions = predict_future(
                model,
                all_scaled,
                scaler,
                steps=forecast_days
            )

        # -------------------- KET QUA --------------------

        st.success(f"Đã chạy mô hình cho mã {ticker}")

        last_close = float(data[-1, 0])
        last_date = pd.Timestamp(df.index[-1]).strftime("%d/%m/%Y")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Giá đóng cửa gần nhất",
            f"{last_close:,.2f}"
        )
        col2.metric("RMSE", f"{rmse:,.3f}")
        col3.metric("MAE", f"{mae:,.3f}")
        col4.metric("R²", f"{r2:.4f}" if np.isfinite(r2) else "N/A")

        st.caption(
            f"Dữ liệu gần nhất: {last_date} | "
            f"Tổng số phiên: {len(data):,} | "
            f"Train: {len(train_data):,} | "
            f"Test: {len(test_data):,}"
        )

        # -------------------- BIEU DO 1: LOSS --------------------

        st.subheader("1. Loss Train và Validation")

        loss_fig = go.Figure()

        loss_fig.add_trace(go.Scatter(
            y=history.history["loss"],
            mode="lines+markers",
            name="Train Loss"
        ))

        loss_fig.add_trace(go.Scatter(
            y=history.history["val_loss"],
            mode="lines+markers",
            name="Validation Loss"
        ))

        loss_fig.update_layout(
            title="Loss Train vs Validation",
            xaxis_title="Epoch",
            yaxis_title="Mean Squared Error",
            hovermode="x unified",
            template="plotly_white",
            height=420
        )

        st.plotly_chart(loss_fig, use_container_width=True)

        # -------------------- BIEU DO 2: TEST --------------------

        st.subheader("2. Giá thực tế so với giá dự báo trên tập Test")

        test_dates = df.index[train_size:]

        # So khop ngay voi ket qua test
        test_dates = test_dates[-len(y_test_real):]

        test_fig = go.Figure()

        test_fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_test_real,
            mode="lines",
            name="Giá thực tế"
        ))

        test_fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_pred,
            mode="lines",
            name="Giá dự báo"
        ))

        test_fig.update_layout(
            title=f"So sánh giá thực tế và dự báo - {ticker}",
            xaxis_title="Ngày",
            yaxis_title="Giá đóng cửa",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(test_fig, use_container_width=True)

        # -------------------- BIEU DO 3: DU BAO TUONG LAI --------------------

        st.subheader(
            f"3. Dự báo {forecast_days} phiên giao dịch tiếp theo"
        )

        # Ngay lam viec tham khao; khong tinh lich nghi le
        future_dates = pd.bdate_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days
        )

        future_df = pd.DataFrame({
            "Ngày dự báo": future_dates,
            "Giá dự báo": future_predictions
        })

        future_fig = go.Figure()

        # Hien thi mot doan gia lich su gan nhat
        history_count = min(100, len(df))

        future_fig.add_trace(go.Scatter(
            x=df.index[-history_count:],
            y=df["Close"].iloc[-history_count:],
            mode="lines",
            name="Giá lịch sử"
        ))

        # Noi duong du bao voi gia dong cua gan nhat
        forecast_x = [df.index[-1]] + list(future_dates)
        forecast_y = [last_close] + list(future_predictions)

        future_fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode="lines+markers",
            name="Giá dự báo tương lai",
            line=dict(dash="dash")
        ))

        future_fig.add_vline(
            x=df.index[-1],
            line_dash="dot",
            annotation_text="Bắt đầu dự báo"
        )

        future_fig.update_layout(
            title=f"Dự báo giá {ticker}",
            xaxis_title="Ngày",
            yaxis_title="Giá đóng cửa",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(future_fig, use_container_width=True)

        # -------------------- BANG DU BAO --------------------

        st.subheader("Bảng giá dự báo")

        future_df["Giá dự báo"] = future_df["Giá dự báo"].round(2)

        st.dataframe(
            future_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = future_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "📥 Tải bảng dự báo CSV",
            data=csv_data,
            file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
            mime="text/csv"
        )

        # -------------------- LICH SU GIA --------------------

        with st.expander("Xem dữ liệu giá lịch sử"):
            st.dataframe(
                df.tail(100).sort_index(ascending=False),
                use_container_width=True
            )

        st.caption(
            "Lưu ý: R² có thể âm nếu mô hình dự báo kém hơn "
            "mốc dự báo bằng giá trung bình. Giá dự báo nhiều bước "
            "được tạo nối tiếp từ chính các dự báo trước đó nên "
            "sai số có thể tích lũy. Ngày làm việc hiển thị chưa "
            "loại trừ các ngày nghỉ lễ của thị trường chứng khoán Việt Nam."
        )

    except Exception as e:
        st.error(
            "Có lỗi khi tải dữ liệu hoặc chạy mô hình. "
            "Hãy thử lại hoặc kiểm tra log của ứng dụng."
        )
        st.exception(e)

else:
    st.markdown("### Hướng dẫn sử dụng")
    st.write(
        "1. Nhập mã cổ phiếu ở thanh bên trái.\n"
        "2. Chọn ngày bắt đầu lấy dữ liệu.\n"
        "3. Chọn số phiên muốn dự báo và số epochs tối đa.\n"
        "4. Nhấn **Chạy mô hình dự báo** và đợi kết quả."
    )

    st.warning(
        "Ứng dụng chỉ chạy mô hình sau khi bạn nhấn nút. "
        "Dự báo là kết quả mô hình học máy, không phải giá chắc chắn "
        "sẽ xảy ra trên thị trường."
    )

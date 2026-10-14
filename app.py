
# ============================================================
# UNG DUNG DU BAO GIA CO PHIEU BANG LSTM
# Streamlit + TensorFlow + Yahoo Finance
# ============================================================

import os
import random
from datetime import date, timedelta

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# 1. CAU HINH
# ============================================================

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

st.title("📈 ỨNG DỤNG DỰ ĐOÁN GIÁ CỔ PHIẾU")
st.markdown(
    "Ứng dụng sử dụng mô hình **LSTM Deep Learning** "
    "để phân tích giá đóng cửa lịch sử và dự báo các phiên tiếp theo."
)

st.info(
    "Ứng dụng phục vụ học tập và nghiên cứu. "
    "Giá dự báo không phải giá chắc chắn sẽ xảy ra "
    "và không phải khuyến nghị mua bán cổ phiếu."
)

# ============================================================
# 2. TAI DU LIEU CO PHIEU
# ============================================================

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

    if df is None or df.empty:
        return pd.DataFrame()

    # Xu ly ca cot thong thuong va MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return pd.DataFrame()

        close = df.xs("Close", axis=1, level=0)

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
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

    return result.dropna()


# ============================================================
# 3. TAO DU LIEU CHUOI 60 PHIEN
# ============================================================

def create_sequences(data, window_size=WINDOW_SIZE):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # LSTM can du lieu 3 chieu: mau, thoi gian, dac trung
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y


# ============================================================
# 4. XAY DUNG MO HINH LSTM
# ============================================================

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


# ============================================================
# 5. DU BAO NHIEU PHIEN TIEP THEO
# ============================================================

def predict_future(model, scaled_data, scaler, steps):
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
            [[[next_value]]], dtype=np.float32
        )

        current_window = np.concatenate(
            [current_window[:, 1:, :], next_value_array],
            axis=1
        )

    predictions = np.asarray(
        predictions, dtype=np.float32
    ).reshape(-1, 1)

    return scaler.inverse_transform(predictions).flatten()


# ============================================================
# 6. THANH CAI DAT
# ============================================================

with st.sidebar:
    st.header("⚙️ Cài đặt mô hình")

    ticker = st.text_input(
        "Mã cổ phiếu Yahoo Finance",
        value="VNM.VN",
        help="Ví dụ: VNM.VN, FPT.VN, HPG.VN, VCB.VN"
    ).strip().upper()

    if ticker and "." not in ticker:
        ticker = ticker + ".VN"
        
    start_date = st.date_input(
        "Ngày bắt đầu dữ liệu",
        value=date(2018, 1, 1),
        max_value=date.today(),
        key="start_date"
    )

    start_date = st.date_input(
        "Ngày bắt đầu dữ liệu",
        value=date(2018, 1, 1),
        max_value=date.today(),
        key="start_date"
    )

    end_date = st.date_input(
        "Ngày kết thúc dữ liệu",
        value=date.today(),
        min_value=start_date,
        max_value=date.today(),
        key="end_date"
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
        value=15,
        step=5
    )

    run_button = st.button(
        "🚀 Chạy mô hình dự báo",
        type="primary",
        use_container_width=True
    )

    st.caption(
        "Lần chạy đầu có thể lâu hơn vì ứng dụng phải huấn luyện LSTM."
    )


# ============================================================
# 7. CHAY UNG DUNG
# ============================================================

if run_button:

    if not ticker:
        st.error("Bạn chưa nhập mã cổ phiếu.")
        st.stop()

    if start_date >= end_date:
    st.error("Ngày kết thúc phải sau ngày bắt đầu.")
    st.stop()

# Yahoo Finance khong bao gom ngay end trong ket qua
download_end_date = end_date + timedelta(days=1)

    try:
        # ----------------------------------------------------
        # BUOC A: TAI DU LIEU
        # ----------------------------------------------------

        with st.spinner("Đang tải dữ liệu từ Yahoo Finance..."):
            df = download_stock_data(
                ticker,
                start_date.isoformat(),
                 download_end_date.isoformat()
            )

        if df.empty:
            st.error(
                "Không tải được dữ liệu cho mã "
                f"{ticker}. Hãy kiểm tra mã cổ phiếu, "
                "kết nối Internet hoặc thử mã VNM.VN."
            )
            st.stop()

        data = df[["Close"]].values.astype(np.float32)

        if len(data) < 150:
            st.error(
                f"Dữ liệu chỉ có {len(data)} phiên. "
                "Hãy chọn ngày bắt đầu sớm hơn để có ít nhất "
                "150 phiên dữ liệu."
            )
            st.stop()

        # ----------------------------------------------------
        # BUOC B: CHIA TRAIN VA TEST THEO THOI GIAN
        # ----------------------------------------------------

        train_size = int(len(data) * 0.8)

        train_data = data[:train_size]

        # Scaler chi hoc tren tap train
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)

        # Chuyen toan bo du lieu sang cung thang do
        all_scaled = scaler.transform(data)

        X_train, y_train = create_sequences(train_scaled)

        # Tao test voi 60 phien lien truoc ngay dau test
        test_combined = np.vstack([
            train_scaled[-WINDOW_SIZE:],
            all_scaled[train_size:]
        ])

        X_test, y_test = create_sequences(test_combined)

        # ----------------------------------------------------
        # BUOC C: CHIA VALIDATION
        # ----------------------------------------------------

        val_size = max(1, int(len(X_train) * 0.1))

        if len(X_train) <= val_size:
            st.error("Không đủ dữ liệu để chia tập validation.")
            st.stop()

        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]

        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

        # ----------------------------------------------------
        # BUOC D: HUAN LUYEN LSTM
        # ----------------------------------------------------

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        model = build_model()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        progress = st.progress(0)
        status = st.empty()

        class StreamlitProgress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress.progress(
                    min((epoch + 1) / epochs, 1.0)
                )
                status.text(
                    f"Đang huấn luyện: epoch {epoch + 1}/{epochs}"
                )

        with st.spinner("Đang huấn luyện mô hình LSTM..."):
            history = model.fit(
                X_train_final,
                y_train_final,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                shuffle=False,
                callbacks=[early_stop, StreamlitProgress()],
                verbose=0
            )

        progress.empty()
        status.empty()

        # ----------------------------------------------------
        # BUOC E: DANH GIA TREN TAP TEST
        # ----------------------------------------------------

        with st.spinner("Đang đánh giá và tạo dự báo..."):
            test_pred_scaled = model.predict(
                X_test, verbose=0
            )

            test_pred = scaler.inverse_transform(
                test_pred_scaled.reshape(-1, 1)
            ).flatten()

            y_test_real = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()

            rmse = float(np.sqrt(
                mean_squared_error(y_test_real, test_pred)
            ))

            mae = float(
                mean_absolute_error(y_test_real, test_pred)
            )

            r2 = float(
                r2_score(y_test_real, test_pred)
            )

            # Huan luyen lai tren toan bo du lieu de du bao tuong lai.
            # Dung so epoch da hoc tot nhat o mo hinh validation.
            best_epoch = len(history.history["loss"])

            future_X, future_y = create_sequences(all_scaled)

            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(SEED)

            future_model = build_model()

            future_model.fit(
                future_X,
                future_y,
                epochs=max(1, best_epoch),
                batch_size=32,
                shuffle=False,
                verbose=0
            )

            future_predictions = predict_future(
                future_model,
                all_scaled,
                scaler,
                forecast_days
            )

        # ====================================================
        # 8. HIEN THI CAC CHI SO
        # ====================================================

        st.success(f"Đã chạy mô hình thành công cho {ticker}.")

        last_close = float(data[-1, 0])
        last_date = pd.Timestamp(df.index[-1])

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Giá đóng cửa gần nhất",
            f"{last_close:,.2f}"
        )

        col2.metric("RMSE", f"{rmse:,.3f}")
        col3.metric("MAE", f"{mae:,.3f}")
        col4.metric("R²", f"{r2:.4f}")

        st.caption(
            f"Ngày dữ liệu cuối: {last_date:%d/%m/%Y} | "
            f"Số phiên dữ liệu: {len(data)} | "
            f"Epoch tốt nhất: {best_epoch}"
        )

        # ====================================================
        # 9. BIEU DO LOSS
        # ====================================================

        st.subheader("1. Loss Train và Validation")

        loss_fig = go.Figure()

        loss_fig.add_trace(go.Scatter(
            x=list(range(1, len(history.history["loss"]) + 1)),
            y=history.history["loss"],
            mode="lines+markers",
            name="Train Loss"
        ))

        loss_fig.add_trace(go.Scatter(
            x=list(range(1, len(history.history["val_loss"]) + 1)),
            y=history.history["val_loss"],
            mode="lines+markers",
            name="Validation Loss"
        ))

        loss_fig.update_layout(
            title="Loss Train vs Validation",
            xaxis_title="Epoch",
            yaxis_title="Mean Squared Error",
            template="plotly_white",
            hovermode="x unified",
            height=420
        )

        st.plotly_chart(
            loss_fig,
            use_container_width=True
        )

        # ====================================================
        # 10. BIEU DO GIA THUC TE VA DU BAO TEST
        # ====================================================

        st.subheader("2. Giá thực tế so với giá dự báo trên tập Test")

        test_dates = df.index[train_size:]

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
            title=f"Giá thực tế và dự báo - {ticker}",
            xaxis_title="Ngày",
            yaxis_title="Giá đóng cửa",
            template="plotly_white",
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(
            test_fig,
            use_container_width=True
        )

        # ====================================================
        # 11. BIEU DO DU BAO TUONG LAI
        # ====================================================

        st.subheader(
            f"3. Dự báo {forecast_days} phiên tiếp theo"
        )

        # Ngay lam viec tham khao, chua loai ngay nghi le
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days
        )

        future_fig = go.Figure()

        history_count = min(100, len(df))

        future_fig.add_trace(go.Scatter(
            x=df.index[-history_count:],
            y=df["Close"].iloc[-history_count:],
            mode="lines",
            name="Giá lịch sử"
        ))

        future_fig.add_trace(go.Scatter(
            x=[last_date] + list(future_dates),
            y=[last_close] + list(future_predictions),
            mode="lines+markers",
            name="Giá dự báo",
            line=dict(dash="dash")
        ))

        
future_fig.add_shape(
    type="line",
    x0=last_date,
    x1=last_date,
    y0=0,
    y1=1,
    xref="x",
    yref="paper",
    line=dict(
        color="gray",
        dash="dot",
        width=2
    )
)

future_fig.add_annotation(
    x=last_date,
    y=1,
    xref="x",
    yref="paper",
    text="Bắt đầu dự báo",
    showarrow=False,
    yshift=12
)
```

Cách này vẽ đường thẳng đứng trực tiếp trên trục ngày tháng, tránh lỗi phép cộng/trừ `Timestamp` ở `add_vline()`.

## 4. Sửa phần tải dữ liệu để biết chính xác nguyên nhân

Tìm hàm `download_stock_data()` ở đầu file và thay toàn bộ hàm bằng đoạn sau. Đoạn này thử tải dữ liệu lại tối đa 3 lần nếu xảy ra lỗi, đồng thời xử lý tên cột `Close` và cấu trúc dữ liệu khác nhau của Yahoo Finance.

<CodeBlock language="python" editable>
@st.cache_data(ttl=900, show_spinner=False)
def download_stock_data(ticker, start_date, end_date):
    import time

    last_error = None

    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    if "Close" not in df.columns.get_level_values(0):
                        return pd.DataFrame()

                    close = df.xs(
                        "Close", axis=1, level=0
                    )

                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                else:
                    if "Close" not in df.columns:
                        return pd.DataFrame()

                    close = df["Close"]

                result = pd.DataFrame({"Close": close})
                result.index = pd.to_datetime(result.index)
                result = result.sort_index()
                result = result[
                    ~result.index.duplicated(keep="last")
                ]
                result["Close"] = pd.to_numeric(
                    result["Close"], errors="coerce"
                )
                result = result.dropna()

                if not result.empty:
                    return result

        except Exception as e:
            last_error = str(e)

        if attempt < 2:
            time.sleep(2)

    if last_error:
        raise RuntimeError(
            f"Yahoo Finance không tải được {ticker}: {last_error}"
        )

    return pd.DataFrame()

        future_fig.update_layout(
            title=f"Dự báo giá tương lai - {ticker}",
            xaxis_title="Ngày",
            yaxis_title="Giá đóng cửa",
            template="plotly_white",
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(
            future_fig,
            use_container_width=True
        )

        # ====================================================
        # 12. BANG DU BAO
        # ====================================================

        st.subheader("Bảng giá dự báo")

        future_df = pd.DataFrame({
            "Ngày dự báo": future_dates.strftime("%d/%m/%Y"),
            "Giá dự báo": np.round(future_predictions, 2)
        })

        st.dataframe(
            future_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="📥 Tải bảng dự báo CSV",
            data=future_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
            mime="text/csv"
        )

        with st.expander("Xem 100 phiên dữ liệu gần nhất"):
            st.dataframe(
                df.tail(100).sort_index(ascending=False),
                use_container_width=True
            )

        st.warning(
            "Mô hình dự báo nhiều phiên theo cách nối tiếp, "
            "vì vậy sai số có thể tăng theo thời gian. "
            "Ngày dự báo chưa loại trừ ngày nghỉ lễ của thị trường."
        )

    except Exception as e:
        st.error(
            "Ứng dụng gặp lỗi khi tải dữ liệu hoặc chạy mô hình. "
            "Thông tin kỹ thuật:"
        )
        st.exception(e)

else:
    st.subheader("Hướng dẫn sử dụng")

    st.markdown(
        """
        1. Nhập mã cổ phiếu, ví dụ `VNM.VN`, `FPT.VN`, `HPG.VN`.
        2. Chọn ngày bắt đầu lấy dữ liệu.
        3. Chọn số phiên muốn dự báo.
        4. Nhấn **Chạy mô hình dự báo**.
        5. Chờ quá trình tải dữ liệu và huấn luyện LSTM hoàn tất.
        """
    )

    st.caption(
        "Ứng dụng cần kết nối Internet để lấy dữ liệu từ Yahoo Finance."
    )

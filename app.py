import streamlit as st
import pickle
import numpy as np

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🌬️ 대기질 CO 농도 예측 시스템")
st.write("센서값을 입력하면 일산화탄소(CO) 농도를 예측합니다.")

st.sidebar.header("센서값 입력")

pt08_s1 = st.sidebar.slider("PT08.S1(CO) 센서값", 463, 1737, 1100)
c6h6 = st.sidebar.slider("C6H6(GT) 벤젠 농도", 0.0, 32.0, 10.0)
pt08_s2 = st.sidebar.slider("PT08.S2(NMHC) 센서값", 155, 1723, 939)
nox = st.sidebar.slider("NOx(GT) 질소산화물", 0, 830, 248)
pt08_s3 = st.sidebar.slider("PT08.S3(NOx) 센서값", 77, 1594, 836)
no2 = st.sidebar.slider("NO2(GT) 이산화질소", 0, 245, 113)
pt08_s4 = st.sidebar.slider("PT08.S4(NO2) 센서값", 441, 2471, 1456)
pt08_s5 = st.sidebar.slider("PT08.S5(O3) 오존센서", 0, 2194, 1023)
temp = st.sidebar.slider("T 온도 (℃)", -8.0, 44.0, 18.0)
rh = st.sidebar.slider("RH 상대습도 (%)", 0.0, 100.0, 49.0)
ah = st.sidebar.slider("AH 절대습도", 0.0, 2.2, 1.0)

input_data = np.array([[pt08_s1, c6h6, pt08_s2, nox, pt08_s3, no2, pt08_s4, pt08_s5, temp, rh, ah]])
prediction = model.predict(input_data)

st.subheader("예측 결과")
st.metric(label="CO 농도 예측값 (mg/m³)", value=f"{prediction[0]:.2f}")

if prediction[0] < 1.5:
    st.success("✅ 안전 수준입니다.")
elif prediction[0] < 3.0:
    st.warning("⚠️ 주의 수준입니다.")
else:
    st.error("🚨 위험 수준입니다!")
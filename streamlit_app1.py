import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances_argmin

# 1. 모델 및 리소스 불러오기
model = joblib.load("best_model.pkl")
features = joblib.load("feature_columns.pkl")
scaler = joblib.load("scaler.pkl")
cluster_centers = np.load("cluster_centers.npy")
cluster_features = joblib.load("cluster_features.pkl")

# 2. 앱 제목
st.title("💳 고객 이탈 예측 앱 (XGBoost 기반)")

st.markdown("고객 정보를 입력하면, 신용카드 이탈 가능성을 예측하고, 유지 고객 유사 조건을 제안합니다.")

# 3. 사용자 입력받기 (중요 변수 위주)
age = st.slider("고객 나이 (Customer Age)", 18, 90, 45)
trans_ct = st.slider("총 거래 수 (Total_Trans_Ct)", 0, 150, 60)
revolving_bal = st.number_input("리볼빙 잔액 (Total_Revolving_Bal)", 0, 10000, 1000)
trans_amt = st.number_input("총 거래 금액 (Total_Trans_Amt)", 0, 20000, 3000)
ct_chg = st.slider("분기 거래수 변화율 (Total_Ct_Chng_Q4_Q1)", 0.0, 2.0, 0.5)

# 4. 입력값 구성
user_input = {
    'Customer_Age': age,
    'Total_Trans_Ct': trans_ct,
    'Total_Revolving_Bal': revolving_bal,
    'Total_Trans_Amt': trans_amt,
    'Total_Ct_Chng_Q4_Q1': ct_chg
}

# 5. 전체 피처에 맞게 입력값 구성
input_df = pd.DataFrame([user_input])
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[features]

# 6. 예측 수행
proba = model.predict_proba(input_df)[0][1]
pred = model.predict(input_df)[0]

st.subheader("📈 이탈 예측 결과")
st.metric(label="이탈 확률", value=f"{proba*100:.2f}%")
st.write(f"예측 결과: **{'이탈' if pred == 1 else '유지'} 고객**")

# 7. 유지 유사 클러스터 추천 (이탈 고객만 대상)
if pred == 1:
    st.warning("이 고객은 이탈할 가능성이 높습니다.")
    st.markdown("🔎 **유사한 유지 고객 클러스터를 분석하여, 유지 전환 조건을 제시합니다.**")

    # 클러스터링용 피처 스케일링
    user_df_cluster = pd.DataFrame([user_input])
    for col in cluster_features:
        if col not in user_df_cluster.columns:
            user_df_cluster[col] = 0
    user_df_cluster = user_df_cluster[cluster_features]

    scaled_input = scaler.transform(user_df_cluster)
    closest_cluster = pairwise_distances_argmin(scaled_input, cluster_centers)[0]

    st.markdown(f"✅ 가장 유사한 유지 고객 클러스터: **Cluster {closest_cluster}**")

    # 유사 클러스터와 차이 큰 항목 제시
    cluster_center = cluster_centers[closest_cluster]
    diffs = scaled_input[0] - cluster_center
    diff_df = pd.DataFrame({
        'Feature': cluster_features,
        'Difference': diffs
    }).set_index('Feature').abs().sort_values(by='Difference', ascending=False)

    top_n = 3
    st.markdown(f"🔁 유지 고객과의 차이가 큰 상위 {top_n}개 항목:")
    for feature in diff_df.head(top_n).index:
        current = user_df_cluster[feature].values[0]
        target = round(cluster_center[cluster_features.index(feature)], 2)
        st.markdown(f"- `{feature}`: 현재 값 **{current}** → 유지 고객 평균 **{target}**")

# 8. 참고 정보
with st.expander("ℹ️ 예측에 사용된 주요 변수"):
    st.write(input_df.T)

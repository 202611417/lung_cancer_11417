import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import numpy as np
import koreanize_matplotlib

# 1. 페이지 설정
st.set_page_config(
    page_title="AI Lung Health Analyzer",
    page_icon="🫁",
    layout="wide"
)


# 커스텀 CSS (세련된 다크/블루 테마)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stNumberInput, .stSlider { border-radius: 10px; }
    .result-card {
        padding: 30px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 모델 및 스케일러 로드 함수
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('lung_model.pkl')
        scaler = joblib.load('lung_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("파일을 찾을 수 없습니다: lung_model.pkl 또는 lung_scaler.pkl")
        return None, None

model, scaler = load_assets()


# 🌟 [수정됨] 3. 기존 데이터셋 로드 (lung_cancer_examples.csv 반영)
@st.cache_data
def load_original_data():
    try:
        # 말씀해주신 파일명으로 수정되었습니다.
        return pd.read_csv('lung_cancer_examples.csv') 
    except FileNotFoundError:
        # 파일이 없을 경우 에러 방지용 임시 데이터 생성
        st.sidebar.warning("⚠️ 'lung_cancer_examples.csv' 파일이 없어 임시 데이터로 시각화합니다.")
        dummy_data = pd.DataFrame({
            '나이': np.random.randint(20, 80, 200),
            '흡연량': np.random.uniform(0, 40, 200),
            '지역환경지수': np.random.randint(10, 90, 200),
            'cluster': np.random.randint(0, 4, 200) # 0, 1, 2, 3번 군집 무작위
        })
        return dummy_data

# df 변수 정의
df = load_original_data()


# 4. 사이드바: 프로젝트 정보
with st.sidebar:
    st.title("🫁 AI Diagnostic Unit")
    st.info("이 시스템은 나이, 흡연량, 지역 환경 지수를 분석하여 환자의 건강 위험 군집을 예측합니다.")
    st.divider()
    st.caption("© 2024 AI Healthcare Solutions")

# 5. 메인 화면
st.title("폐 건강 위험 군집 예측 시스템")
st.subheader("실시간 환자 데이터 분석 및 AI 리포트")

# 데이터 입력 구역 (좌측 입력, 우측 결과)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.write("### 📋 환자 프로필 입력")
    age = st.slider("나이 (Age)", 0, 100, 45)
    smokes = st.number_input("일일 흡연량 (Smoking Amount)", min_value=0.0, value=10.0, step=0.1)
    areaQ = st.slider("지역 환경 지수 (Area Quality Index)", 0, 100, 50)
    
    predict_btn = st.button("분석 실행", use_container_width=True, type="primary")

with col2:
    st.write("### 📊 분석 결과")
    if predict_btn and model is not None:
        with st.spinner('AI 분석 중...'):
            # 데이터 준비
            new_patient = pd.DataFrame([[age, smokes, areaQ]], columns=['나이', '흡연량', '지역환경지수'])
            
            # 스케일링 및 예측
            new_patient_scaled = scaler.transform(new_patient)
            pred_cluster = model.predict(new_patient_scaled)
            
            time.sleep(1) # 시각적 효과를 위한 딜레이
            
            # 결과 표시 카드
            st.markdown(f"""
                <div class="result-card">
                    <p style="font-size: 1.2rem; opacity: 0.9;">예측된 환자 위험 군집</p>
                    <h1 style="font-size: 4rem; margin: 10px 0;">{pred_cluster[0]} <span style="font-size: 1.5rem;">번</span></h1>
                    <p style="font-size: 1rem;">해당 환자는 {pred_cluster[0]}번 그룹의 특성을 보입니다.<br>정밀 진단이 권장됩니다.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 6. 군집별 결과 해석 안내
            st.write("---")
            st.markdown("""
            ##### 💡 환자 군집 분석 가이드
            * **0번 군집:** 🟢 매우 건강군
            * **1번 군집:** 🟡 건강군
            * **2번 군집:** 🟠 중간 그룹 (주의 요망)
            * **3번 군집:** 🔴 강한 폐암 위험군 (정밀 검진 필수)
            """)
            
            # 7. 산점도(Scatter Plot) 시각화 코드
            st.write("---")
            st.write("##### 📍 환자 위치 시각화 (나이 vs 흡연량)")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # 기존 데이터 플롯 (df 사용)
            scatter = ax.scatter(df['나이'], df['흡연량'], c=df['cluster'], cmap='coolwarm', alpha=0.5, label='기존 데이터')
            
            # 현재 입력된 새 환자 위치 플롯
            ax.scatter(age, smokes, c='red', s=250, marker='X', label='새 환자 (Current Patient)')
            
            ax.set_xlabel('나이 (Age)')
            ax.set_ylabel('흡연량 (Smoking Amount)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 화면에 출력
            st.pyplot(fig)
            
            # 입력 데이터 요약 표
            st.write("---")
            st.write("##### 📝 입력 데이터 요약")
            st.table(new_patient)
    else:
        st.info("왼쪽에서 환자 데이터를 입력하고 '분석 실행' 버튼을 눌러주세요.")

st.divider()
st.write("⚠️ **주의:** 이 결과는 AI 모델에 기반한 예측치이며, 반드시 전문의의 상담이 필요합니다.")

# 생활지수 예측 베이스라인 모델 (Baseline Model)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 1️⃣ 데이터 불러오기
df = pd.read_csv('data/preprocessed_data.csv')

# -----------------------------
# 2️⃣ 데이터 확인 및 전처리
# -----------------------------
# print("데이터 요약:")
# print(df.info())
# print("\n결측치 확인:")
# print(df.isnull().sum())

# 전운량평균의 결측치(비어있는 값)는 평균으로 대체
df['전운량평균'].fillna(df['전운량평균'].mean(), inplace=True)

# 필요 없는 열(예: 날짜, 지역)은 제거
df = df.drop(columns=['날짜','지역']) # 지역은 세종 하나뿐이기에 제거

# -----------------------------
# 3️⃣ 타깃 변수 / 입력 변수 분리
# -----------------------------
X = df.drop(columns=['생활지수'])
y = df['생활지수']

# -----------------------------
# 4️⃣ 범주형 변수 인코딩
# -----------------------------
# '생활지수'가 object형 → 숫자로 변환 필요

le_target = LabelEncoder()
y = le_target.fit_transform(y)

# -----------------------------
# 5️⃣ 학습용 / 검증용 데이터 분리 (8:2)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_test.shape}")

# -----------------------------
# 6️⃣ 스케일링 (표준화)
# -----------------------------
# 수치형 변수만 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 7️⃣ 모델 학습 (RandomForest)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,       # 트리 개수
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 8️⃣ 예측 및 성능 평가
# -----------------------------
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\n정확도(Accuracy): {acc:.4f}")

# 9️⃣ 혼동행렬 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.xlabel('예측값 (Predicted)')
plt.ylabel('실제값 (Actual)')
plt.title('생활지수 예측 혼동행렬')
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("혼동행렬 이미지가 'confusion_matrix.png' 파일로 저장되었습니다.")
# 미세먼지 농도 회귀 예측 베이스라인
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

# 1️⃣ 데이터 준비
df = pd.read_csv('data/preprocessed_data.csv')

# 2️⃣ 결측치 처리
df['전운량평균'].fillna(df['전운량평균'].mean(), inplace=True)

# 불필요한 열 제거
df = df.drop(columns=['날짜', '지역']) # 지역은 세종시만 존재하므로 제거

# -----------------------------
# 3️⃣ 타깃 / 피처 분리
# -----------------------------
target = '미세먼지'
X = df.drop(columns=[target, '생활지수', '초미세먼지'])  # 생활지수는 분류용이므로 제외
y = df[target]

# -----------------------------
# 5️⃣ 학습용/검증용 데이터 분할 (8:2)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"학습 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_test.shape}")


# 6️⃣ 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 7️⃣ 회귀 모델 학습 (RandomForest)
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)


# 8️⃣ 예측 및 평가
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n MAE  (평균절대오차): {mae:.3f}")
print(f" RMSE (평균제곱근오차): {rmse:.3f}")


# 9️⃣ 실제값 vs 예측값 시각화
# -----------------------------
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel('실제 미세먼지 (Actual)')
plt.ylabel('예측 미세먼지 (Predicted)')
plt.title('실제 vs 예측 미세먼지 농도')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

plt.savefig("scatter_plot.png", dpi=300)
plt.close()
print("산점도 이미지가 'scatter_plot.png' 파일로 저장되었습니다.")
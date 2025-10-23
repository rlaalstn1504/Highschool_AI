# Matplotlib & Seaborn 기본 시각화 예제
# -----------------------------------------------
# 목적: 더미데이터로 다양한 그래프를 그려보며
#       시각화의 기본 개념을 익힌다.
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

# -------------------------------
# 1️⃣ 더미데이터 생성
# -------------------------------
# 예시: 5개 지역의 평균 기온, 습도, 미세먼지 농도
data = {
    '지역': ['서울', '부산', '대구', '광주', '춘천', '전라', '대전', '울산'],
    '기온평균': [18.5, 20.3, 21.1, 19.7, 17.8, 21.1, 19.7, 17.8],
    '습도평균': [55, 65, 58, 62, 60, 58, 62, 60],
    '미세먼지': [45, 52, 48, 48, 48, 48, 43, 50]
}

df = pd.DataFrame(data)

# --------------------------------------------
# 2️⃣ 막대그래프 (Bar Chart)
# --------------------------------------------
# 카테고리별 수치 비교에 적합한 그래프
plt.figure(figsize=(6, 4)) # 그래프 크기 (가로 6인치, 세로 4인치)
# x: 가로축, y: 세로축, hue: 색상 구분 기준
# palette: 색상 조합, legend=False: 범례 표시 안 함
sns.barplot(x='지역', y='기온평균', hue='지역', data=df, palette='Blues_d', legend=False)
plt.title('지역별 평균 기온')
plt.xlabel('지역')
plt.ylabel('평균 기온(°C)')
plt.tight_layout()
plt.savefig('bar_chart.png', dpi=300)
plt.close()
print("막대그래프(bar_chart.png) 저장 완료")

# --------------------------------------------
# 3️⃣ 산점도 (Scatter Plot)
# --------------------------------------------
# 두 변수 간의 관계(상관성)를 확인할 때 사용
plt.figure(figsize=(6, 4))
# x: 가로축 변수, y: 세로축 변수
# s: 점 크기, color: 색상, edgecolor: 테두리 색
sns.scatterplot(x='기온평균', y='습도평균', data=df, s=100, color='orange', edgecolor='black')
plt.title('기온 vs 습도 관계')
plt.xlabel('평균 기온(°C)')   # X축 이름
plt.ylabel('평균 습도(%)')    # Y축 이름
plt.tight_layout() # 그래프 요소 간 여백 자동 조정
plt.savefig('scatter_plot.png', dpi=300)
plt.close()
print("산점도(scatter_plot.png) 저장 완료")

# --------------------------------------------
# 4️⃣ 히스토그램 (Histogram)
# --------------------------------------------
# 데이터의 분포(빈도)를 확인할 때 사용
# 예: 미세먼지 농도의 분포 형태 
# bins : 구간 개수 (막대의 개수)
# kde=True : 분포 곡선(kernel density estimate) 추가
plt.figure(figsize=(6, 4))
sns.histplot(df['미세먼지'], bins=8, kde=True) 
plt.title('미세먼지 농도 분포')
plt.xlabel('미세먼지 (㎍/㎥)')  # X축 이름
plt.ylabel('빈도 (Frequency)')  # Y축 이름
plt.tight_layout()
plt.savefig('histogram.png', dpi=300)
plt.close()
print("히스토그램(histogram.png) 저장 완료")

# --------------------------------------------
# 요약 설명
# --------------------------------------------
# • 막대그래프 : 범주(지역) 간 수치 비교에 유용
# • 산점도     : 두 변수 간의 관계(상관성) 파악
# • 히스토그램 : 단일 변수의 분포 형태 확인
# --------------------------------------------

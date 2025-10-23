import pandas as pd
import os 

# 데이터 불러오기
weather = pd.read_csv("data/sejong_weather.csv", parse_dates=["일시"])
air = pd.read_csv("data/sejong_air.csv", parse_dates=["측정일시"])
weather_index = pd.read_csv("data/sejong_weather_index.csv", parse_dates=["날짜"])

# 데이터 확인하기
print(weather.head())
print(air.head())
print(weather_index.head())


# Weather 일 단위 집계
weather_daily = weather.groupby(weather["일시"].dt.date).agg({
    "기온(°C)": ["mean","max","min"],
    "강수량(mm)": "sum",
    "풍속(m/s)": "mean",
    "습도(%)": "mean",
    "일조(hr)": "sum",
    "적설(cm)": "sum",
    "전운량(10분위)": "mean"
})
weather_daily.columns = [
    "기온평균","기온최고","기온최저",
    "강수량합","풍속평균","습도평균",
    "일조합","적설합","전운량평균"
]
weather_daily = weather_daily.reset_index().rename(columns={"일시":"날짜"})

# 날짜 타입 통일
weather_daily["날짜"] = pd.to_datetime(weather_daily["날짜"])
air["측정일시"] = pd.to_datetime(air["측정일시"])

# 대기 에너지의 일 단위 집계
# 프로젝트를 열게 되면, 해당 부분이 없기 때문에 학생과 같이 실습을 진행하세요.)

air_daily = air.groupby(air["측정일시"].dt.date).agg({
    "아황산가스":"mean",
    "미세먼지":"mean",
    "오존":"mean",
    "이산화질소":"mean",
    "일산화탄소":"mean",
    "초미세먼지":"mean"
}).reset_index().rename(columns={"측정일시":"날짜"})

# 날짜 타입 통일 (에러 방지용)
air_daily["날짜"] = pd.to_datetime(air_daily["날짜"])
weather_index["날짜"] = pd.to_datetime(weather_index["날짜"])

# 날씨, 대기, 생활 지수 데이터 병합 진행
data = pd.merge(weather_daily, air_daily, on="날짜")
final_data = pd.merge(data, weather_index, on="날짜")

# 최종 전처리 데이터 저장
final_data.to_csv("data/preprocessed_data.csv", index=False, encoding="utf-8-sig")

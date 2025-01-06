import streamlit as st
import pandas as pd
import requests

from lib import Model

st.set_page_config(
    page_title="Hava Durumu tahmini",
    page_icon=":partly_sunny:",
    layout="wide",
    menu_items={
        "About": "**Yapay zeka** kullanarak önceden tanımlanmış veri setlerini kullanarak **hava tahminlerini** gösterir."
    },
)

hide_menu_style = """<style>#MainMenu,.stAppDeployButton,footer {display: none !important; visibility: hidden !important;}</style>"""
st.markdown(hide_menu_style, unsafe_allow_html=True)


# ---


st.title("Hava Durumu Tahmini")

st.caption("Bu uygulama, Yapay zeka kullanarak hava durumu tahminleri yapar.")

# ---

st.subheader("Veri Seti")

df = pd.read_csv("../datasets/weather-extended.csv")
st.dataframe(df, use_container_width=True)

st.line_chart(df.drop(["weather", "city"], axis=1))


@st.cache_data
def get_lat_lon(cities):
    lat_lon = []
    for city in cities:
        response = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=9b06695880876d3dcead8840099e691c&units=metric"
        )

        if response.status_code == 200:
            data = response.json()
            lat_lon.append([data["coord"]["lat"], data["coord"]["lon"]])

    return lat_lon


df_map = pd.DataFrame(
    get_lat_lon(df["city"].unique()),
    columns=["lat", "lon"],
)

st.map(df_map, size=2000)

# ---

st.subheader("Model Tahmini", help="XGBClassifier")

min_precipitation = df["precipitation"].min()
max_precipitation = df["precipitation"].max()
if "precipitation" not in st.session_state:
    st.session_state["precipitation"] = (min_precipitation + max_precipitation) / 2

min_temp_max = df["temp_max"].min()
max_temp_max = df["temp_max"].max()
if "temp_max" not in st.session_state:
    st.session_state["temp_max"] = (min_temp_max + max_temp_max) / 2

min_temp_min = df["temp_min"].min()
max_temp_min = df["temp_min"].max()
if "temp_min" not in st.session_state:
    st.session_state["temp_min"] = (min_temp_min + max_temp_min) / 2

min_wind = df["wind"].min()
max_wind = df["wind"].max()
if "wind" not in st.session_state:
    st.session_state["wind"] = (min_wind + max_wind) / 2


if st.button("Rasgele", type="primary", icon=":material/refresh:"):
    random_row = df.sample()

    st.session_state["city"] = random_row["city"].values[0]
    st.session_state["precipitation"] = random_row["precipitation"].values[0]
    st.session_state["temp_max"] = random_row["temp_max"].values[0]
    st.session_state["temp_min"] = random_row["temp_min"].values[0]
    st.session_state["wind"] = random_row["wind"].values[0]

if st.button("Veri al", type="primary", icon="🔍"):
    response = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?q={st.session_state['city']}&appid=9b06695880876d3dcead8840099e691c&units=metric"
    )

    if response.status_code == 200:
        st.session_state["precipitation"] = response.json()["main"]["humidity"]
        st.session_state["temp_max"] = response.json()["main"]["temp_max"]
        st.session_state["temp_min"] = response.json()["main"]["temp_min"]
        st.session_state["wind"] = response.json()["wind"]["speed"]

city = st.selectbox("Şehir", df["city"].unique(), key="city", help="city")

cols = st.columns(2)

with cols[0]:
    precipitation = st.slider(
        "Yağış miktarı",
        min_precipitation,
        max_precipitation,
        key="precipitation",
        help="precipitation",
    )
    temp_min = st.slider(
        "Minimum sıcaklık",
        min_temp_min,
        max_temp_min,
        key="temp_min",
        help="temp_min",
    )

with cols[1]:
    temp_max = st.slider(
        "Maksimum sıcaklık",
        min_temp_max,
        max_temp_max,
        key="temp_max",
        help="temp_max",
    )
    wind = st.slider("Rüzgar hızı", min_wind, max_wind, key="wind", help="temp_max")

# ---


@st.cache_data
def model():
    return Model()


try:
    pred = model().predict(city, [precipitation, temp_max, temp_min, wind])

    st.write(f"Tahmin: **{pred}**")
except Exception as e:
    print("Error: ", e)

    st.write("Oops, try again")

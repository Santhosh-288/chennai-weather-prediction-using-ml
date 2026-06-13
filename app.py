import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# PAGE CONFIG

st.set_page_config(
    page_title="Chennai Weather Prediction",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Chennai Weather Prediction & Alert System")
st.write("Predict whether it will rain tomorrow in Chennai using Machine Learning.")

# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv("chennai_weather.csv")

    # Chennai only
    df = df[df["city"] == "Chennai"]

    return df

df = load_data()
df = df[df["city"] == "Chennai"]


# FEATURES & TARGET

features = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m"
]

X = df[features]
y = df["rain_tomorrow"]

# TRAIN MODEL

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

# SIDEBAR INPUTS

st.sidebar.header("Weather Parameters")

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    value=30.0
)

humidity = st.sidebar.number_input(
    "Relative Humidity (%)",
    value=80.0
)

precipitation = st.sidebar.number_input(
    "Precipitation",
    value=5.0
)

pressure = st.sidebar.number_input(
    "Surface Pressure",
    value=1005.0
)

cloud_cover = st.sidebar.number_input(
    "Cloud Cover (%)",
    value=75.0
)

wind_speed = st.sidebar.number_input(
    "Wind Speed",
    value=15.0
)

# PREDICTION

if st.sidebar.button("Predict Weather"):

    sample = pd.DataFrame(
        [[
            temperature,
            humidity,
            precipitation,
            pressure,
            cloud_cover,
            wind_speed
        ]],
        columns=[
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m"
        ]
    )

    prediction = model.predict(sample)[0]

    probability = (
        model.predict_proba(sample)[0][1]
        * 100
    )

    st.subheader("Prediction Result")

    st.metric(
        "Chance of Rain Tomorrow",
        f"{probability:.2f}%"
    )

    if prediction == 1:
        st.success("🌧️ Rain Expected Tomorrow")

        if probability > 80:
            st.error(
                "⚠️ Heavy Rain Alert! Carry an umbrella."
            )

    else:
        st.success("☀️ No Rain Expected Tomorrow")

# MODEL PERFORMANCE

st.subheader("Model Performance")

st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)


# DATA PREVIEW

st.subheader("Dataset Preview")

st.dataframe(df.head())

# BASIC STATS

st.subheader("Dataset Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Records",
        len(df)
    )

with col2:
    st.metric(
        "Rainy Days",
        int(df["rain_tomorrow"].sum())
    )

with col3:
    st.metric(
        "Features Used",
        len(features)
    )

# FEATURE IMPORTANCE


st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

st.caption(
    "Random Forest Classifier trained using historical Chennai weather data."
)

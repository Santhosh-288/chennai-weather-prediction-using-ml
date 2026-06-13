 Chennai Rain Prediction & Alert System

 Overview

The Chennai Rain Prediction & Alert System is a Machine Learning-based web application developed using Python and Streamlit. The system predicts whether it will rain tomorrow in Chennai based on current weather parameters such as temperature, humidity, precipitation, surface pressure, cloud cover, and wind speed.

A Random Forest Classifier is trained on historical Chennai weather data to generate predictions and estimate the probability of rainfall. The application also provides rain alerts when the predicted chance of rain is high.

 Features

* Predicts whether it will rain tomorrow in Chennai
* Calculates probability of rainfall
* Provides weather alerts for high rainfall chances
* Interactive user interface built with Streamlit
* Displays model accuracy and performance metrics
* Visualizes feature importance using charts
* Shows dataset preview and statistics

Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* Random Forest Classifier

Input Parameters

The model uses the following weather parameters:

* Temperature (°C)
* Relative Humidity (%)
* Precipitation
* Surface Pressure
* Cloud Cover (%)
* Wind Speed

Machine Learning Model

The project uses a Random Forest Classifier trained on historical Chennai weather data.

Dataset Split:

* Training Data: 80%
* Testing Data: 20%

The model predicts the target variable:

* `rain_tomorrow`

  * 1 = Rain Expected
  * 0 = No Rain Expected

## How to Run

1. Install the required packages:

pip install -r requirements.txt

2. Start the Streamlit application:

streamlit run app.py

3. Open the generated local URL in your browser.

roject Structure

├── app.py
├── requirements.txt
├── README.md
├── weather.csv

Objective

The objective of this project is to demonstrate the application of Machine Learning in rainfall prediction using historical weather data and to provide users with a simple weather alert system for Chennai.

 Author
Santhosh C
Santhosh M
BE Computer Science and Engineering

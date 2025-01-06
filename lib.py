import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class Model:
    def __init__(self):
        self.init_model()

    def init_model(self):
        self.df = pd.read_csv("../datasets/weather-extended.csv")

        Q1 = self.df.quantile(0.25, numeric_only=True)
        Q3 = self.df.quantile(0.75, numeric_only=True)
        IQR = Q3 - Q1

        a_aligned_df, a_aligned_Q1 = self.df.align(Q1 - 1.5 * IQR, axis=1, copy=False)
        b_aligned_df, b_aligned_Q3 = self.df.align(Q3 + 1.5 * IQR, axis=1, copy=False)

        outliers = (a_aligned_df < a_aligned_Q1) | (b_aligned_df > b_aligned_Q3)

        self.df = self.df[~outliers.any(axis=1)]

        self.df["precipitation"] = np.sqrt(self.df["precipitation"])
        self.df["wind"] = np.sqrt(self.df["wind"])

    def get_xgb(self, city):
        df = self.df[self.df["city"] == city]
        df = df.drop("city", axis=1)

        encoder = LabelEncoder()
        df["weather"] = encoder.fit_transform(df["weather"])

        x = ((df.loc[:, df.columns != "weather"]).astype(int)).values[:, 0:]
        y = df["weather"].values

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.1, random_state=2
        )

        xgb = XGBClassifier()
        xgb.fit(x_train, y_train)

        return xgb, encoder

    def predict(self, city, data):
        xgb, encoder = self.get_xgb(city)

        pred = xgb.predict([data])
        pred = encoder.inverse_transform(pred)[0]

        icons = {"drizzle": "🌧️", "fog": "🌫️", "rain": "🌧️", "snow": "❄️", "sun": "☀️"}

        return f"{pred} {icons[pred]}"

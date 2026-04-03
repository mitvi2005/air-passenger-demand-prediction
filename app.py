from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import math

app = Flask(__name__)
model = pickle.load(open("model1.pkl", "rb"))


def retrain_model():
    global model
    df = pd.read_csv("new_air_passangers.csv")
    df = df.drop_duplicates()
    X = df[['month', 'year']]
    y = df['Passengers']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    print(
        f"Model retrained! R2 Score: {r2:.4f} | "
        f"MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f}"
    )
    pickle.dump(model, open("model1.pkl", "wb"))
    return {
        'r2': round(r2, 4),
        'mae': round(mae, 2),
        'mse': round(mse, 2),
        'rmse': round(rmse, 2),
        'rows': len(df)
        }

def get_model_stats():
    df = pd.read_csv("new_air_passangers.csv")
    df = df.drop_duplicates()
    X = df[['month', 'year']]
    y = df['Passengers']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model_temp = LinearRegression()
    model_temp.fit(X_train, y_train)
    y_pred = model_temp.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    return {
        'r2': round(r2, 4),
        'mae': round(mae, 2),
        'mse': round(mse, 2),
        'rmse': round(rmse, 2),
        'rows': len(df),
        'actual': y_test.tolist()[:30],
        'predicted': [round(p, 1) for p in y_pred.tolist()[:30]]
    }

@app.route('/')
def home():
    stats = get_model_stats()
    return render_template("index.html", stats=stats)


@app.route('/information')
def information():
    stats = get_model_stats()
    return render_template("information.html", stats=stats)


@app.route('/predictor')
def predictor():
    stats = get_model_stats()
    return render_template("predictor.html", stats=stats, show_result=False, error=None)


@app.route('/about')
def about():
    stats = get_model_stats()
    return render_template("about.html", stats=stats)


@app.route('/api/stats')
def api_stats():
    try:
        stats = get_model_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/historical')
def api_historical():
    try:
        df = pd.read_csv("new_air_passangers.csv")
        df = df.drop_duplicates().sort_values(['year', 'month'])
        result = []
        for _, row in df.iterrows():
            result.append({
                'year': int(row['year']),
                'month': int(row['month']),
                'passengers': round(float(row['Passengers']), 0)
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        month = int(request.form['month'])
        year = int(request.form['year'])
        data = pd.DataFrame({
            'month': [month],
            'year': [year]
        })
        prediction = model.predict(data)
        pred_value = round(float(prediction[0]), 0)

        new_data = pd.DataFrame({
            'Passengers': [pred_value],
            'month': [month],
            'year': [year]
        })

        file_path = "new_air_passangers.csv"
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            duplicate = existing_df[
                (existing_df['month'] == month) &
                (existing_df['year'] == year)]
            if duplicate.empty:
                new_data.to_csv(file_path, mode='a', header=False, index=False)
        else:
            new_data.to_csv(file_path, index=False)
        
        stats = retrain_model()

        months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_name = months_names[month - 1]

        return render_template(
            "predictor.html",
            prediction=pred_value,
            month=month,
            year=year,
            month_name=month_name,
            show_result=True,
            stats=stats
        )
    except Exception as e:
        return render_template(
            "predictor.html",
            error=str(e),
            stats=get_model_stats(),
            show_result=False)


if __name__ == "__main__":
    app.run(debug=True)


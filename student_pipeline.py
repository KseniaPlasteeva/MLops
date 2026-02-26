from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import joblib
import os

DATA_URL = 'https://raw.githubusercontent.com/KseniaPlasteeva/MLops/refs/heads/main/data/student-mat.csv'
TARGET_COLUMN = 'G3'
CAT_COLUMNS = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
               'nursery', 'higher', 'internet', 'romantic']
NUM_COLUMNS = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
               'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
               'G1', 'G2']

DATA_PATH = '/home/student/airflow/data/student_data.csv'
CLEAR_DATA_PATH = '/home/student/airflow/data/student_clear.csv'
MODEL_PATH = '/home/student/airflow/models/student_model.pkl'

default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 3),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def download_data():
    df = pd.read_csv(DATA_URL, sep=';')
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Загружено строк: {df.shape[0]}, колонок: {df.shape[1]}")
    return df.shape[0]

def preprocess_data():
    df = pd.read_csv(DATA_PATH)
    initial_shape = df.shape[0]
    df = df.dropna()
    if CAT_COLUMNS:
        ordinal = OrdinalEncoder()
        ordinal.fit(df[CAT_COLUMNS])
        ordinal_encoded = ordinal.transform(df[CAT_COLUMNS])
        df_ordinal = pd.DataFrame(ordinal_encoded, columns=CAT_COLUMNS)
        df[CAT_COLUMNS] = df_ordinal[CAT_COLUMNS]
    df.to_csv(CLEAR_DATA_PATH, index=False)
    print(f"Очистка завершена. Было строк: {initial_shape}, стало: {df.shape[0]}")
    return df.shape[0]

def scale_frame(frame):
    df = frame.copy()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scale, Y_scale, power_trans, scaler

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_model():
    df = pd.read_csv(CLEAR_DATA_PATH)
    X, Y, power_trans, scaler = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
        "fit_intercept": [False, True],
    }
    os.makedirs('/home/student/airflow/mlruns', exist_ok=True)
    mlflow.set_tracking_uri('file:///home/student/airflow/mlruns')
    mlflow.set_experiment("student_performance_model")
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        rmse, mae, r2 = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(best, MODEL_PATH)
        print(f"Модель обучена. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
        return {"rmse": rmse, "mae": mae, "r2": r2}

dag = DAG(
    dag_id="student_performance_pipeline",
    default_args=default_args,
    description="ML pipeline for student performance prediction",
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    tags=['ml', 'students'],
)

download_task = PythonOperator(task_id="download_student_data", python_callable=download_data, dag=dag)
preprocess_task = PythonOperator(task_id="preprocess_student_data", python_callable=preprocess_data, dag=dag)
train_task = PythonOperator(task_id="train_student_model", python_callable=train_model, dag=dag)

download_task >> preprocess_task >> train_task

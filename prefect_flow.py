from prefect import flow, task
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


@task
def load_data():
    df = pd.read_csv("reviews_badminton.csv")
    df = df.dropna(subset=["Review text", "Ratings"])
    df["sentiment"] = df["Ratings"].apply(lambda x: 1 if x >= 4 else 0)
    return df


@task
def split_data(df):
    X = df["Review text"]
    y = df["sentiment"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


@task
def train_and_log(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    f1 = f1_score(y_test, preds)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(pipeline, "model")

    return f1


@flow(name="flipkart_sentiment_prefect_flow")
def training_flow():
    mlflow.set_experiment("Flipkart Sentiment Analysis - MLflow")

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    f1 = train_and_log(X_train, X_test, y_train, y_test)

    print("Final F1 Score:", f1)


if __name__ == "__main__":
    training_flow()

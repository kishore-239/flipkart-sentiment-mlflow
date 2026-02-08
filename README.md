# Sentiment Analysis of Real-time Flipkart Product Reviews  
### MLflow Experiment Tracking & Model Management

---

## Project Overview

This project demonstrates how **MLflow** can be integrated into a machine learning workflow to track experiments, compare models, manage artifacts, and register the best-performing model.

The sentiment analysis task is based on real-time Flipkart product reviews. In addition to MLflow, a **minimal Prefect workflow** is included to demonstrate task-level orchestration.

This project focuses on **experiment tracking and model management**, not deployment.

---

## Objective

- Build a sentiment classification model for Flipkart reviews
- Track multiple experiments using MLflow
- Log parameters, metrics, and trained models
- Compare models based on F1-score
- Register the best model in MLflow Model Registry
- Demonstrate Prefect task orchestration

---

## Dataset

The dataset was provided as part of the internship task.

**Product:** YONEX MAVIS 350 Nylon Shuttle  
**Total Reviews:** 8,518  

**Key Columns Used:**
- Review text
- Ratings

**Sentiment Labeling Logic:**
- Rating ≥ 4 → Positive (1)
- Rating ≤ 2 → Negative (0)
- Neutral ratings were excluded

---

## Tech Stack

- Python
- Pandas
- scikit-learn
- MLflow
- Prefect

---

## Project Structure

``` text
flipkart_sentiment_mlflow/
│
├── reviews_badminton.csv
├── train_with_mlflow.ipynb
├── prefect_flow.py
├── requirements.txt
├── README.md
└── screenshots/
├── mlflow_runs.png
├── mlflow_f1_metric_plot.png
├── mlflow_model_registry.png
├── prefect_flow_run.png
└── prefect_task_runs.png

```

---

## MLflow Experiment Tracking

MLflow was used to track and compare multiple experiments.

### Models Trained
- Logistic Regression (C = 1)
- Logistic Regression (C = 10)
- Linear Support Vector Classifier (LinearSVC)

### Logged Information
- **Parameters**
  - Model type
  - Regularization parameter (C)
- **Metrics**
  - F1-score
- **Artifacts**
  - Trained scikit-learn pipeline (TF-IDF + classifier)

Each experiment run was logged under a single MLflow experiment for comparison.

---

## Model Evaluation

- Evaluation Metric: **F1-score**
- All models achieved strong performance
- Best F1-score observed: **~0.96**

Metric comparison plots were generated directly in the MLflow UI.

---

## Model Registry

The best-performing model was registered in the MLflow Model Registry.

- **Registered Model Name:** `flipkart_sentiment_model`
- **Latest Version:** Version 1

This enables versioning and future lifecycle management of the model.

---

## Prefect Workflow (Bonus)

A minimal Prefect workflow was implemented to demonstrate task orchestration.

- A simple flow (`test-flow`) was created
- One task (`hello`) was executed and tracked
- Task execution and completion were visualized in the Prefect UI

This confirms Prefect integration and task-level execution tracking without adding unnecessary complexity.

---

## Screenshots Included

The `screenshots/` directory contains evidence of:

- MLflow experiment runs
- F1-score comparison plot
- MLflow Model Registry with registered model
- Prefect flow execution
- Prefect task run visualization

---

## Conclusion

This project demonstrates how **MLflow** can be effectively used for:

- Experiment tracking
- Model comparison
- Metric visualization
- Model versioning and registration

Additionally, **Prefect** was used to showcase basic workflow orchestration.  
The project emphasizes reproducibility, clarity, and practical MLOps practices.

---

## Author

**Krishna Kishore**  
mail : kishorekrishna623@gmail.com  

---

## Tags

#MachineLearning #MLOps #MLflow #Prefect #SentimentAnalysis #Internship

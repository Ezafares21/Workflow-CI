import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import mlflow

# Memulai run MLflow untuk pelacakan CI secara lokal
with mlflow.start_run(run_name="CI_Automated_Run"):
    
    print("Memuat dataset...")
    # Memastikan path sesuai dengan struktur folder pada repository GitHub
    dataset_path = 'Credit_Scoring_Bank_clean.csv'
    df = pd.read_csv(dataset_path)

    # Pemisahan fitur (X) dan target (y)
    target_column = 'y'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Melatih model...")
    # Pelatihan model Random Forest dengan parameter statis untuk pengujian CI
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    # Prediksi dan perhitungan metrik evaluasi
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Pembuatan dan penyimpanan artefak visual (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    print("Mencatat log metrik dan parameter ke MLflow...")
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    mlflow.sklearn.log_model(rf, "model_rf")
    mlflow.log_artifact(cm_path)
    
    print("Menyimpan model fisik secara lokal...")
    # Menyimpan model ke direktori workspace GitHub untuk validasi artefak
    model_path = os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), "model_lokal")
    
    # Membersihkan direktori model lokal jika sudah ada untuk menghindari konflik penimpaan data
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        
    mlflow.sklearn.save_model(rf, model_path)
    
    print("Proses CI Pipeline selesai dieksekusi.")

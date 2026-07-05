import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import mlflow

# 1. Konfigurasi URI pelacakan MLflow ke server DagsHub
mlflow.set_tracking_uri("https://dagshub.com/Ezafares21/Eksperimen_SML_Reza-Al-Pares.mlflow")

with mlflow.start_run(run_name="CI_Automated_Run"):
    
    # 2. Memuat dataset yang telah diproses
    print("Memuat dataset...")
    df = pd.read_csv('Credit_Scoring_Bank_preprocessing/Credit_Scoring_Bank_clean.csv')

    # 3. Pemisahan fitur (X) dan target (y)
    target_column = 'y'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Pelatihan model Random Forest dengan parameter statis untuk efisiensi CI
    print("Melatih model...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    # 5. Prediksi dan perhitungan metrik evaluasi
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # 6. Pembuatan dan penyimpanan artefak visual (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # 7. Pencatatan (logging) parameter, metrik, dan model utama ke MLflow Tracking
    print("Mencatat log ke server MLflow...")
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    mlflow.sklearn.log_model(rf, "model_rf")
    mlflow.log_artifact(cm_path)
    
    # 8. Penyimpanan model fisik secara lokal untuk pembuatan Docker Image
    print("Menyimpan model fisik untuk Docker build...")
    model_path = os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), "model_lokal")
    mlflow.sklearn.save_model(rf, model_path)
    
    print("Proses CI Pipeline selesai dieksekusi.")
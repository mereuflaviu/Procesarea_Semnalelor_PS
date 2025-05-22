import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Căi către fișierele salvate de preprocess_data.py
FEATURES_PATH = 'features_data.npy'
LABELS_PATH = 'labels_data.npy'

# Căi unde salvăm modelul și scaler-ul
MODEL_PATH = 'rf_model.pkl'
SCALER_PATH = 'scaler.pkl'

def main():
    # 1. Încărcăm datele MFCC + etichete
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)

    print(f"Dimensiuni X: {X.shape}, y: {y.shape}")

    # 2. Împărțim train/test (80% - 20%), menținând distribuția etichetelor cu stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 3. StandardScaler (opțional pentru Random Forest, dar îl păstrăm)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Construim un Random Forest cu parametri rezonabili pentru ~100k instanțe
    rf_model = RandomForestClassifier(
        n_estimators=200,       # creștem nr de arbori pentru stabilitate
        max_depth=15,          # limităm adâncimea - scade timpul și overfitting-ul
        class_weight='balanced',  # penalizează clasele subreprezentate
        n_jobs=-1,             # folosește toate nucleele CPU
        random_state=42
    )

    # Antrenăm modelul
    rf_model.fit(X_train_scaled, y_train)

    # 5. Evaluăm pe test
    y_pred = rf_model.predict(X_test_scaled)

    # Acuratețe globală
    accuracy = (y_pred == y_test).mean()
    print(f"Acuratețe (globală): {accuracy * 100:.2f}%")

    # Raport de clasificare: precision, recall, f1 pentru fiecare clasă
    print("\n=== Raport de clasificare pe fiecare clasă ===")
    print(classification_report(y_test, y_pred, digits=3))

    # 6. Salvăm modelul și scaler-ul
    joblib.dump(rf_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel RF salvat în: {MODEL_PATH}")
    print(f"Scaler salvat în: {SCALER_PATH}")

if __name__ == '__main__':
    main()

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

FEATURES_PATH = 'features_data.npy'
LABELS_PATH = 'labels_data.npy'

MODEL_PATH = 'svm_model.pkl'
SCALER_PATH = 'scaler.pkl'

def main():
    # 1. Încărcăm datele
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    print(f"Dimensiuni X: {X.shape}, y: {y.shape}")

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 3. StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. GridSearchCV pe hiperparametrii SVM (kernel RBF)
    param_grid = {
        'C': [ 1, 10],
        'gamma': ['scale', 0.1],
        'kernel': ['rbf']
    }
    svm = SVC(probability=True)
    grid = GridSearchCV(svm, param_grid, cv=2, verbose=1, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    print(f"Cel mai bun set de hiperparametri: {grid.best_params_}")
    best_model = grid.best_estimator_

    # 5. Evaluare pe setul de test
    test_acc = best_model.score(X_test_scaled, y_test)
    print(f"Acuratețe pe setul de test: {test_acc * 100:.2f}%")

    # 6. Salvăm modelul și scaler-ul
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model salvat în '{MODEL_PATH}'")
    print(f"Scaler salvat în '{SCALER_PATH}'")

if __name__ == '__main__':
    main()

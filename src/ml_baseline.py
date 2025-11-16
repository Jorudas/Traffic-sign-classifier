
# src/ml_baseline.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import load_data_for_ml


def main():
    print("📦 Kraunam duomenis ML modeliui...")
    X_train, X_test, y_train, y_test = load_data_for_ml(limit=8000)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )

    print("🌳 Mokom RandomForest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ RandomForest tikslumas: {acc:.4f}")

    print("\n📊 Klasifikacijos ataskaita:")
    print(classification_report(y_test, y_pred))

    print("\n🧩 Confusion matrix (forma):", confusion_matrix(y_test, y_pred).shape)


if __name__ == "__main__":
    main()
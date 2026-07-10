import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except:
    HAS_LGBM = False


df = pd.read_csv("clean_data.csv")

if "ID" in df.columns:
    df.drop(columns="ID", inplace=True)

if "year" in df.columns:
    if df["year"].nunique() == 1:
        df.drop(columns="year", inplace=True)

leakage_cols = [
    "Interest_rate_spread",
    "Upfront_charges",
    "rate_of_interest"
]

for col in leakage_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

X = df.drop(columns="Status")
y = df["Status"]

cat = X.select_dtypes(include=["object","string"]).columns

X = pd.get_dummies(
    X,
    columns=cat,
    drop_first=True,
    dtype=int
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nSMOTE Sonrası Sınıf Dağılımı")
print(y_train.value_counts())

numeric = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("scale", StandardScaler(), numeric)
], remainder="passthrough")


models = {

    "Logistic Regression":
        Pipeline([
            ("pre", pre),
            ("model",
             LogisticRegression(
                 max_iter=3000,
                 class_weight="balanced",
                 random_state=42
             ))
        ]),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
}

if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

if HAS_LGBM:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )


results = []

plt.figure(figsize=(8,6))

for name, model in models.items():

    print(f"\n{name} modeli eğitiliyor...")

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]

    pred = (prob >= 0.40).astype(int)

    
    acc = accuracy_score(y_test, pred)
    pre_score = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    results.append([
        name,
        acc,
        pre_score,
        rec,
        f1,
        auc
    ])

    print("-"*60)
    print(f"Model      : {name}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {pre_score:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {auc:.4f}")

    RocCurveDisplay.from_predictions(
        y_test,
        prob,
        name=name
    )

plt.title("ROC Eğrisi Karşılaştırması")
plt.grid(True)
plt.tight_layout()
plt.show()


results = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

results = results.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n")
print("="*70)
print("MODEL KARŞILAŞTIRMA SONUÇLARI")
print("="*70)
print(results.round(4))



metrics_to_plot = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1 Score"]
for metric in metrics_to_plot:
    plt.figure(figsize=(10,5))
    plt.bar(results["Model"], results[metric])
    plt.title(f"{metric} Karşılaştırması")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()


best_name = results.iloc[0]["Model"]
print("\nEn Başarılı Model:", best_name)

best_model = models[best_name]
best_model.fit(X_train, y_train)

prob = best_model.predict_proba(X_test)[:,1]
pred = (prob >= 0.40).astype(int)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    display_labels=["Temerrüt Yok", "Temerrüt Var"],
    cmap="Blues",
    values_format="d"
)

plt.title(best_name + " - Karışıklık Matrisi")
plt.tight_layout()
plt.show()



if best_name == "Random Forest":

    importance = pd.DataFrame({
        "Özellik": X.columns,
        "Önem": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Önem",
        ascending=False
    )

    print("\nEn Önemli 20 Özellik")
    print(importance.head(20))

    plt.figure(figsize=(10,8))
    plt.barh(
        importance.head(20)["Özellik"],
        importance.head(20)["Önem"]
    )
    plt.gca().invert_yaxis()
    plt.title("En Önemli 20 Özellik")
    plt.xlabel("Önem Skoru")
    plt.tight_layout()
    plt.show()

print("\nAnaliz tamamlandı.")

import os
import re
import hashlib
import hmac
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
import shap

st.set_page_config(page_title="Cancer Risk Prediction", page_icon="🎗️", layout="wide")

THRESHOLD = 0.30
DEFAULT_EMAIL = "demo@gmail.com"
DEFAULT_PHONE = "9876543210"
DEFAULT_PASSWORD = "Cancer@123"


def secret(name, default):
    try:
        value = st.secrets.get(name)
        return value if value else default
    except Exception:
        return os.getenv(name, default)


def is_valid_identifier(value: str) -> bool:
    email_ok = bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value))
    phone_ok = bool(re.fullmatch(r"\+?[0-9]{10,15}", value))
    return email_ok or phone_ok


def password_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def authenticate(identifier: str, password: str) -> bool:
    configured_password_hash = secret("DEMO_PASSWORD_HASH", password_hash(DEFAULT_PASSWORD))
    configured_email = secret("DEMO_EMAIL", DEFAULT_EMAIL).strip().lower()
    configured_phone = secret("DEMO_PHONE", DEFAULT_PHONE).strip()
    identifier = identifier.strip()
    allowed = identifier.lower() == configured_email or identifier == configured_phone
    return allowed and hmac.compare_digest(password_hash(password), configured_password_hash)


@st.cache_data(show_spinner=False)
def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    # sklearn encoding: 0 = malignant, 1 = benign.
    y_malignant = (y == 0).astype(int)
    return X, y_malignant


@st.cache_resource(show_spinner=False)
def train_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Specificity": specificity,
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        })
        fitted[name] = model

    metrics = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return X, y, X_train, X_test, y_train, y_test, fitted, metrics


def shap_explanation(model_pipeline, X_train, input_df):
    preprocessor = Pipeline(model_pipeline.steps[:-1])
    estimator = model_pipeline.steps[-1][1]
    X_train_processed = preprocessor.fit_transform(X_train)
    input_processed = preprocessor.transform(input_df)
    explainer = shap.LinearExplainer(estimator, X_train_processed)
    values = explainer(input_processed).values[0]
    return values


def login_screen():
    st.title("🎗️ Cancer Risk Prediction System")
    st.caption("Secure demo login")
    st.write("Enter your Gmail address **or** phone number, followed by your password.")
    with st.form("login_form"):
        identifier = st.text_input("Gmail or phone number", placeholder="you@gmail.com or 9876543210")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", use_container_width=True)
    if submitted:
        if not is_valid_identifier(identifier):
            st.error("Please enter a valid Gmail address or phone number.")
        elif authenticate(identifier, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid login details.")
    with st.expander("Demo login details"):
        st.code(f"Gmail: {DEFAULT_EMAIL}\nPhone: {DEFAULT_PHONE}\nPassword: {DEFAULT_PASSWORD}")
    st.info("Demo/screening system only. It is not a medical diagnosis.")


def main_app():
    X, y, X_train, X_test, y_train, y_test, models, metrics = train_models()
    st.title("🎗️ Cancer Risk Prediction Dashboard")
    st.caption("WDBC breast-cancer dataset • screening aid, not a diagnosis")

    with st.sidebar:
        st.header("Account")
        if st.button("Log out", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
        st.divider()
        st.header("Model settings")
        threshold = st.slider("Malignant-risk threshold", 0.10, 0.90, THRESHOLD, 0.05)
        st.caption("Lower thresholds generally increase sensitivity/recall but can also increase false positives.")

    tab1, tab2, tab3 = st.tabs(["Prediction", "Model evaluation", "About project"])

    with tab1:
        st.subheader("Enter clinical measurements")
        st.write("The WDBC dataset contains 30 cell-nucleus measurement features.")
        use_example = st.checkbox("Use an example patient", value=True)
        defaults = X.median() if use_example else pd.Series({c: 0.0 for c in X.columns})

        values = {}
        columns = st.columns(3)
        for i, feature in enumerate(X.columns):
            with columns[i % 3]:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(defaults[feature]),
                    format="%.5f",
                    key=f"feature_{i}",
                )

        if st.button("🔍 Predict cancer risk", type="primary", use_container_width=True):
            input_df = pd.DataFrame([values], columns=X.columns)
            model = models["Logistic Regression"]
            probability = float(model.predict_proba(input_df)[0, 1])
            prediction = int(probability >= threshold)
            label = "Higher malignant-risk screening result" if prediction else "Lower malignant-risk screening result"

            c1, c2, c3 = st.columns(3)
            c1.metric("Malignant-risk probability", f"{probability:.1%}")
            c2.metric("Threshold", f"{threshold:.0%}")
            c3.metric("Result", "Malignant-risk" if prediction else "Benign-risk")
            st.subheader(label)

            if prediction:
                st.warning("The model flags this input above the selected screening threshold. This is not a diagnosis; clinical assessment is required.")
            else:
                st.success("The model flags this input below the selected screening threshold. This does not rule out cancer.")

            try:
                shap_values = shap_explanation(model, X_train, input_df)
                explanation = pd.DataFrame({"Feature": X.columns, "SHAP value": shap_values})
                explanation["Absolute impact"] = explanation["SHAP value"].abs()
                explanation = explanation.sort_values("Absolute impact", ascending=False).head(10)
                st.subheader("Top factors influencing this prediction")
                st.bar_chart(explanation.set_index("Feature")["SHAP value"])
                st.dataframe(explanation, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"SHAP explanation could not be generated: {exc}")

    with tab2:
        st.subheader("Held-out test evaluation")
        st.dataframe(metrics.style.format({
            "Accuracy": "{:.3f}", "Recall": "{:.3f}", "Precision": "{:.3f}",
            "Specificity": "{:.3f}", "ROC-AUC": "{:.3f}"
        }), use_container_width=True, hide_index=True)

        st.subheader("Confusion matrix — Logistic Regression")
        model = models["Logistic Regression"]
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=["Actual benign", "Actual malignant"], columns=["Predicted benign", "Predicted malignant"])
        st.dataframe(cm_df, use_container_width=True)
        st.write(f"ROC-AUC: **{roc_auc_score(y_test, probabilities):.3f}**")

    with tab3:
        st.subheader("Project workflow")
        st.markdown("""
        1. **Login** — user enters a Gmail address or phone number and password.
        2. **Data input** — user enters the 30 WDBC clinical measurement features.
        3. **Preprocessing** — missing values are imputed and features are standardized.
        4. **Prediction** — the Logistic Regression model estimates malignant-risk probability.
        5. **Thresholding** — the selected threshold converts probability into a screening result.
        6. **Explainability** — SHAP shows which features contributed most to the prediction.
        7. **Evaluation** — the app reports recall, specificity, precision, accuracy and ROC-AUC on held-out data.
        """)
        st.warning("This project is a screening/research demonstration. It must not be used as a medical diagnosis or as a substitute for a qualified clinician.")


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    main_app()
else:
    login_screen()

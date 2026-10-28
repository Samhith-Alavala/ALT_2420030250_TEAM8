
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

import plotly.express as px

# ---------------- CONFIGURATION ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_PATH = MODEL_DIR / "fedfit_federated_gentle_69_50.pt"
METRICS_PATH = MODEL_DIR / "federated_metrics_gentle_69_50.json"

ACTIVITIES = [
    "Downstairs", "Jogging", "Sitting",
    "Standing", "Upstairs", "Walking"
]

st.set_page_config(
    page_title="FedFit | Activity Intelligence",
    page_icon="🏃",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #07111f, #101d32);
    color: #edf4ff;
}
[data-testid="stSidebar"] {
    background: #0b1628;
}
h1, h2, h3 { color: #eaf3ff; }
p, label, li { color: #c4d2e5; }
.metric-card {
    background: #14243a;
    border: 1px solid #263d5b;
    padding: 20px;
    border-radius: 14px;
}
.small-note {
    color: #9eb3ce;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = CNNLSTM(num_classes=6)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.asarray(
        checkpoint["normalization_mean"], dtype=np.float32
    ).reshape(1, 1, 3)

    std = np.asarray(
        checkpoint["normalization_std"], dtype=np.float32
    ).reshape(1, 1, 3)

    return model, (mean, std)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        fallback = MODEL_DIR / "federated_metrics.json"
        if not fallback.exists():
            return {}
        path = fallback
    else:
        path = METRICS_PATH

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🏃 FedFit")
    st.caption("Federated Fitness Intelligence")
    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Activity Predictor",
            "Model Analytics",
            "Federated Learning",
            "About Project"
        ]
    )
    st.divider()
    st.caption("KLH University | Research Prototype")
    st.caption("WISDM • CNN-LSTM • FedAvg")


metrics = load_metrics()
model, normalization = load_model()

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.markdown("## FEDFIT / ACTIVITY INTELLIGENCE")
    st.title("Your movement. Understood.")
    st.write(
        "A federated learning prototype for recognizing human "
        "physical activities from wearable accelerometer signals."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activity Classes", "6")
    c2.metric("Sensor Channels", "3")
    c3.metric("Window Length", "5 seconds")
    c4.metric(
        "Selected Model Accuracy",
        f"{metrics.get('best_test_accuracy', metrics.get('final_test_accuracy', 0.695)) * 100:.2f}%"
    )

    st.markdown("### What FedFit does")
    a, b, c = st.columns(3)

    with a:
        st.markdown("#### 🧠 Recognition")
        st.write(
            "A CNN-LSTM model learns spatial patterns and temporal "
            "relationships in accelerometer windows."
        )

    with b:
        st.markdown("#### 🔗 Federated training")
        st.write(
            "Simulated user clients train locally and contribute "
            "model updates through sample-weighted FedAvg."
        )

    with c:
        st.markdown("#### 📊 Performance insights")
        st.write(
            "Explore training rounds, activity categories, and "
            "the limitations of the current model."
        )

    st.info(
        "Research note: the saved score was used during round selection "
        "on the held-out test users. Treat it as a development score, "
        "not an unbiased final test result."
    )

    st.markdown("### Recognized activities")
    cols = st.columns(3)
    icons = ["⬇️", "🏃", "🪑", "🧍", "⬆️", "🚶"]
    for i, activity in enumerate(ACTIVITIES):
        with cols[i % 3]:
            st.markdown(f'<div class="metric-card">{icons[i]} **{activity}**</div>',
                        unsafe_allow_html=True)


# ---------------- PREDICTOR ----------------
elif page == "Activity Predictor":
    st.title("Activity Predictor")
    st.write(
        "Upload a CSV containing exactly 100 consecutive sensor readings "
        "with columns `x`, `y`, and `z`."
    )

    st.warning(
        "Predictions depend on the quality and distribution of the input. "
        "The current model has known weaknesses, particularly for Upstairs."
    )

    uploaded = st.file_uploader("Upload accelerometer CSV", type=["csv"])

    with st.expander("CSV format example"):
        st.code(
            "x,y,z\n0.12,0.98,0.08\n0.15,1.02,0.10\n..."
        )
        st.caption("Your file must contain 100 numeric rows, not counting the header.")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            df.columns = [col.strip().lower() for col in df.columns]

            if not all(col in df.columns for col in ["x", "y", "z"]):
                st.error("CSV must contain x, y, and z columns.")
            elif len(df) != 100:
                st.error(f"Expected 100 rows; received {len(df)}.")
            elif not all(np.issubdtype(df[c].dtype, np.number) for c in ["x", "y", "z"]):
                st.error("Sensor columns must contain numeric values.")
            elif model is None:
                st.error(f"Model checkpoint not found: {MODEL_PATH}")
            else:
                raw = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
                mean, std = normalization
                normalized = (raw.reshape(1, 100, 3) - mean) / std
                tensor = torch.from_numpy(normalized.astype(np.float32))

                with torch.no_grad():
                    logits = model(tensor)
                    probabilities = torch.softmax(logits, dim=1)[0].numpy()

                predicted = int(np.argmax(probabilities))
                st.success(f"Predicted activity: {ACTIVITIES[predicted]}")
                st.metric("Model confidence", f"{probabilities[predicted] * 100:.1f}%")

                chart_df = pd.DataFrame({
                    "Activity": ACTIVITIES,
                    "Probability": probabilities
                }).sort_values("Probability", ascending=True)

                fig = px.bar(
                    chart_df,
                    x="Probability",
                    y="Activity",
                    orientation="h",
                    title="Prediction probabilities",
                    range_x=[0, 1],
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Softmax confidence is not a guarantee that the prediction is correct."
                )
        except Exception as exc:
            st.error(f"Could not process this file: {exc}")


# ---------------- ANALYTICS ----------------
elif page == "Model Analytics":
    st.title("Model Analytics")
    st.write("Explore recorded federated training metrics.")

    rounds = metrics.get("rounds", [])
    if rounds:
        round_df = pd.DataFrame(rounds)

        if "round" in round_df.columns:
            possible = [
                c for c in round_df.columns
                if "accuracy" in c.lower()
            ]
            if possible:
                accuracy_col = (
                    "heldout_test_accuracy"
                    if "heldout_test_accuracy" in possible
                    else possible[0]
                )
                fig = px.line(
                    round_df,
                    x="round",
                    y=accuracy_col,
                    markers=True,
                    title="Accuracy across federated rounds",
                    labels={
                        "round": "Federated round",
                        accuracy_col: "Accuracy"
                    },
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

            if "weighted_local_loss" in round_df.columns:
                fig = px.line(
                    round_df,
                    x="round",
                    y="weighted_local_loss",
                    markers=True,
                    title="Weighted local training loss",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(round_df, use_container_width=True)
    else:
        st.warning("No metrics JSON was found. Check the models folder.")

    st.markdown("### Dataset overview")
    labels_file = DATA_DIR / "label_mapping.csv"
    if labels_file.exists():
        st.dataframe(pd.read_csv(labels_file), use_container_width=True)

    if (DATA_DIR / "y.npy").exists():
        labels = np.load(DATA_DIR / "y.npy")
        counts = pd.Series(labels).value_counts().sort_index()
        label_df = pd.DataFrame({
            "Activity": [ACTIVITIES[int(i)] for i in counts.index],
            "Windows": counts.values
        })
        fig = px.bar(
            label_df,
            x="Activity",
            y="Windows",
            title="Prepared activity windows",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------- FEDERATED LEARNING ----------------
elif page == "Federated Learning":
    st.title("How FedFit learns across users")
    st.write(
        "FedFit simulates multiple user clients on one computer. "
        "Each client trains a copy of the global CNN-LSTM model using "
        "its own user windows."
    )

    steps = [
        ("01", "Initialize", "The server initializes a global CNN-LSTM model."),
        ("02", "Distribute", "The global model is copied to each training client."),
        ("03", "Local training", "Each client updates its model using its local windows."),
        ("04", "Aggregate", "FedAvg combines client parameters weighted by client sample counts."),
        ("05", "Repeat", "The updated global model is sent out for the next round.")
    ]

    for number, title, description in steps:
        st.markdown(
            f'<div class="metric-card"><b>{number} · {title}</b><br>{description}</div>',
            unsafe_allow_html=True
        )
        st.write("")

    st.markdown("### Privacy and scope")
    st.info(
        "This is a simulation of federated learning. Client datasets are "
        "represented separately by user ID, but they remain in the same "
        "local program. The current implementation does not provide "
        "differential privacy, secure aggregation, or a formal privacy guarantee."
    )


# ---------------- ABOUT ----------------
elif page == "About Project":
    st.title("About FedFit")
    st.write(
        "Federated Learning Framework for Personalized Fitness Activity Recognition"
    )

    st.markdown("### Project objectives")
    st.markdown("""
    - Recognize six physical activities from accelerometer sequences.
    - Represent each participant as a federated learning client.
    - Apply sample-count-weighted Federated Averaging (FedAvg).
    - Evaluate generalization to users not included in training.
    - Explore challenges caused by non-IID user activity distributions.
    """)

    st.markdown("### Technology stack")
    st.markdown("""
    - **Dataset:** WISDM smartphone accelerometer data
    - **Deep learning:** PyTorch
    - **Architecture:** CNN-LSTM
    - **Federated method:** FedAvg simulation
    - **Interface:** Streamlit
    - **Analytics:** Pandas and Plotly
    """)

    st.markdown("### Current limitations")
    st.markdown("""
    - Evaluation is based on a specific user-level split.
    - The selected model has weak Upstairs recall.
    - Simulated clients do not represent independent physical devices.
    - Softmax confidence may be poorly calibrated.
    """)

    st.caption("FedFit · Academic research prototype · KLH University")

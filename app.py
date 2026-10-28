
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #142b48 0%, transparent 38%),
        linear-gradient(145deg, #07101d 0%, #0b1424 55%, #101d30 100%);
    color: #edf5ff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(7, 16, 29, 0.75);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1729, #08111f);
    border-right: 1px solid #1e3551;
}

[data-testid="stSidebar"] h1 {
    color: #61e7ed;
    font-weight: 800;
    letter-spacing: -1px;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #9db4cf;
}

h1, h2, h3 {
    color: #f1f7ff !important;
    letter-spacing: -0.6px;
}

p, li, label {
    color: #bdcde0;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #14263c, #101d30);
    border: 1px solid #29415d;
    border-radius: 16px;
    padding: 20px 18px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
}

[data-testid="stMetricLabel"] {
    color: #9eb6d1 !important;
}

[data-testid="stMetricValue"] {
    color: #65e5ed !important;
    font-weight: 800;
}

.stButton > button {
    border: 1px solid #3dd6df;
    border-radius: 10px;
    background: linear-gradient(100deg, #087e94, #13a6b2);
    color: white;
    font-weight: 700;
    padding: 0.65rem 1rem;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border-color: #8bf7f5;
    background: #087e94;
    color: white;
}

[data-testid="stFileUploader"] {
    background: #101e31;
    border: 1px dashed #3b617d;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #243b55;
}

[data-testid="stDataFrame"] {
    border: 1px solid #29415d;
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stVerticalBlock"] > div:has(> .metric-card) {
    margin-bottom: 8px;
}

.metric-card {
    background: linear-gradient(145deg, #152840, #101d30);
    border: 1px solid #2b4562;
    padding: 22px;
    border-radius: 15px;
    color: #eaf4ff;
    min-height: 80px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
}

.small-note {
    color: #9eb3ce;
    font-size: 13px;
}

.fedfit-hero {
    padding: 32px 34px;
    border-radius: 22px;
    background:
        radial-gradient(circle at 90% 10%, rgba(67, 224, 231, 0.18), transparent 35%),
        linear-gradient(115deg, #142b46 0%, #123b50 55%, #10505b 100%);
    border: 1px solid #2b6578;
    margin: 8px 0 26px 0;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.18);
}

.fedfit-eyebrow {
    color: #68e9ef;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.4px;
    margin-bottom: 13px;
}

.fedfit-hero-title {
    color: #f3f8ff;
    font-size: clamp(38px, 5vw, 54px);
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.05;
}

.fedfit-hero-subtitle {
    color: #e2f5ff;
    font-size: 21px;
    font-weight: 600;
    margin-top: 12px;
}

.fedfit-hero-description {
    color: #c0d8e8;
    font-size: 15px;
    line-height: 1.7;
    margin-top: 9px;
    max-width: 760px;
}

#MainMenu, footer {
    visibility: hidden;
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
            "Demo Mode",
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
    st.markdown("""
    <div class="fedfit-hero">
        <div class="fedfit-eyebrow">
            FEDERATED LEARNING &nbsp;•&nbsp; HUMAN ACTIVITY RECOGNITION
        </div>
        <div class="fedfit-hero-title">FedFit</div>
        <div class="fedfit-hero-subtitle">
            Your movement. Understood.
        </div>
        <div class="fedfit-hero-description">
            Personalized fitness activity recognition using wearable sensor
            data and federated learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            st.markdown(
    f'<div class="metric-card">{icons[i]} '
    f'<strong>{activity}</strong></div>',
    unsafe_allow_html=True
)


# ---------------- DEMO MODE ----------------
elif page == "Demo Mode":
    st.markdown("""
    <div class="fedfit-hero">
        <div class="fedfit-eyebrow">PROJECT PRESENTATION MODE</div>
        <div class="fedfit-hero-title">FedFit Demo</div>
        <div class="fedfit-hero-subtitle">
            Federated Learning for Personalized Fitness Activity Recognition
        </div>
        <div class="fedfit-hero-description">
            Recognize physical activities from wearable accelerometer
            signals using a CNN-LSTM model and simulated Federated Averaging.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project at a glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activity Classes", "6")
    c2.metric("Sensor Axes", "3")
    c3.metric("Window Duration", "5 seconds")
    c4.metric(
        "Development Accuracy",
        f"{metrics.get('best_test_accuracy', metrics.get('final_test_accuracy', 0.695)) * 100:.2f}%"
    )

    st.caption(
        "The displayed accuracy is a development result. The held-out test "
        "users were used during round selection, so it is not an unbiased "
        "final evaluation score."
    )

    st.divider()
    st.markdown("### Guided demonstration")

    demo_steps = [
        (
            "01",
            "Activity prediction",
            "Open Activity Predictor and upload a CSV containing 100 "
            "numeric readings with x, y, and z columns."
        ),
        (
            "02",
            "Sensor simulation",
            "Select an activity and simulate a recorded five-second "
            "accelerometer window from the WISDM dataset."
        ),
        (
            "03",
            "Visualize the signals",
            "Explain how the X, Y, and Z accelerometer readings change "
            "over time."
        ),
        (
            "04",
            "Review model probabilities",
            "Compare the probabilities assigned to all six activities. "
            "The predicted class is not guaranteed to be correct."
        ),
        (
            "05",
            "Explore federated learning",
            "Open Federated Learning to explain local client training "
            "and sample-weighted FedAvg aggregation."
        ),
        (
            "06",
            "Review performance",
            "Open Model Analytics to show recorded training-round "
            "metrics and the activity distribution."
        )
    ]

    for number, title, description in demo_steps:
        st.markdown(
            f'<div class="metric-card">'
            f'<strong>{number} · {title}</strong><br><br>'
            f'{description}</div>',
            unsafe_allow_html=True
        )
        st.write("")

    st.divider()
    st.markdown("### Presentation checklist")

   
    checklist_items = [
        "Explain the project objective",
        "Demonstrate an uploaded CSV prediction",
        "Run the sensor data simulator",
        "Show the accelerometer signal graph",
        "Explain the federated learning workflow",
        "Discuss model limitations honestly"
    ]

    if "demo_checklist" not in st.session_state:
        st.session_state.demo_checklist = {
            item: False for item in checklist_items
        }

    for item in checklist_items:
        checked = st.checkbox(
            item,
            value=st.session_state.demo_checklist[item],
            key=f"demo_{item}"
        )
        st.session_state.demo_checklist[item] = checked

    completed = sum(st.session_state.demo_checklist.values())
    st.progress(completed / len(checklist_items))
    st.caption(f"{completed} of {len(checklist_items)} presentation tasks completed")


    st.info(
        "Implementation note: this project simulates federated clients "
        "on one computer. It does not currently implement secure "
        "aggregation or differential privacy."
    )

    st.caption("FedFit · Academic research prototype · KLH University")


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

    # ---------------- SENSOR DATA SIMULATOR ----------------
    st.markdown("### 🏃 Sensor Data Simulator")
    st.caption(
        "Replay a recorded 5-second accelerometer window from WISDM "
        "and see how the existing model classifies it."
    )

    X_FILE = DATA_DIR / "X.npy"
    Y_FILE = DATA_DIR / "y.npy"

    if X_FILE.exists() and Y_FILE.exists():
        simulation_activity = st.selectbox(
            "Choose a sample activity",
            ACTIVITIES,
            key="simulation_activity"
        )

        if st.button("▶ Simulate Sensor Reading", use_container_width=True):
            if model is None or normalization is None:
                st.error("The trained model could not be loaded.")
            else:
                X_data = np.load(X_FILE, mmap_mode="r")
                y_data = np.load(Y_FILE, mmap_mode="r")

                target_label = ACTIVITIES.index(simulation_activity)
                matching_indices = np.flatnonzero(y_data == target_label)

                if len(matching_indices) == 0:
                    st.error("No recorded samples are available for this activity.")
                else:
                    sample_index = int(np.random.choice(matching_indices))
                    raw_signal = np.asarray(
                        X_data[sample_index], dtype=np.float32
                    )

                    mean, std = normalization
                    normalized_signal = (
                        raw_signal.reshape(1, 100, 3) - mean
                    ) / std

                    sensor_tensor = torch.from_numpy(
                        normalized_signal.astype(np.float32)
                    )

                    with torch.no_grad():
                        logits = model(sensor_tensor)
                        probabilities = torch.softmax(logits, dim=1)[0].numpy()

                    predicted_index = int(np.argmax(probabilities))

                   
                    if predicted_index == target_label:
                        st.success(
                            f"Prediction matches the selected activity: "
                            f"{ACTIVITIES[predicted_index]}"
                        )
                    else:
                        st.warning(
                            f"Prediction differs from the selected activity. "
                            f"Selected: {simulation_activity} | "
                            f"Predicted: {ACTIVITIES[predicted_index]}"
                        )


                    col1, col2 = st.columns(2)
                    col1.metric("Selected sample", simulation_activity)
                    col2.metric(
                        "Prediction confidence",
                        f"{probabilities[predicted_index] * 100:.1f}%"
                    )

                    st.caption(
                        f"Recorded sample #{sample_index}. "
                        "The selected activity is the dataset label, "
                        "not a guaranteed model prediction."
                    )

                    signal_df = pd.DataFrame({
                        "Reading": np.arange(1, 101),
                        "X-axis": raw_signal[:, 0],
                        "Y-axis": raw_signal[:, 1],
                        "Z-axis": raw_signal[:, 2]
                    })

                    
                    st.download_button(
                        label="📥 Download Simulated Sensor CSV",
                        data=signal_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"fedfit_{simulation_activity.lower()}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


                    signal_long = signal_df.melt(
                        id_vars="Reading",
                        var_name="Sensor axis",
                        value_name="Acceleration"
                    )

                    signal_fig = px.line(
                        signal_long,
                        x="Reading",
                        y="Acceleration",
                        color="Sensor axis",
                        title="Simulated 5-second accelerometer signal",
                        template="plotly_dark"
                    )
                    st.plotly_chart(signal_fig, use_container_width=True)

                    probability_df = pd.DataFrame({
                        "Activity": ACTIVITIES,
                        "Probability": probabilities
                    }).sort_values("Probability", ascending=True)

                    probability_fig = px.bar(
                        probability_df,
                        x="Probability",
                        y="Activity",
                        orientation="h",
                        range_x=[0, 1],
                        title="Model prediction probabilities",
                        template="plotly_dark"
                    )
                    st.plotly_chart(
                        probability_fig, use_container_width=True
                    )
    else:
        st.info(
            "Simulator unavailable. Make sure data/processed/X.npy "
            "and data/processed/y.npy exist."
        )

    st.divider()

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
                st.markdown(f"""
<div class="fedfit-hero">
    <div class="fedfit-eyebrow">ACTIVITY RECOGNITION RESULT</div>
    <div class="fedfit-hero-title">{ACTIVITIES[predicted]}</div>
    <div class="fedfit-hero-description">
        Predicted physical activity from your uploaded sensor data.
    </div>
</div>
""", unsafe_allow_html=True)    
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
                
                # Download prediction report
                report_df = pd.DataFrame({
                    "Activity": ACTIVITIES,
                    "Probability": probabilities,
                    "Probability_Percent": np.round(probabilities * 100, 2)
                }).sort_values("Probability", ascending=False)

                report_csv = report_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Prediction Report (CSV)",
                    data=report_csv,
                    file_name="fedfit_prediction_report.csv",
                    mime="text/csv",
                    use_container_width=True
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
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            if "weighted_local_loss" in round_df.columns:
                fig = px.line(
                    round_df,
                    x="round",
                    y="weighted_local_loss",
                    markers=True,
                    title="Weighted local training loss",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

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

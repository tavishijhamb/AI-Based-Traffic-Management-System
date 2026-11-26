import os
import tempfile
from datetime import datetime, timedelta
import cv2
import pandas as pd
import streamlit as st
import plotly.express as px
from traffic_core.detector import TrafficDetector
from traffic_core.controller import (
    AdaptiveSignalController,
    density_label
)
from traffic_core.prediction import (
    load_model,
    CongestionForecaster
)
from traffic_core.history import append_observations
from traffic_core.geometry import draw_approaches


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI-Powered Intelligent Traffic Management System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background: linear-gradient(
        135deg,
        #080F1E 0%,
        #101B32 55%,
        #0B1426 100%
    );
    color: #EAF2FF;
}

/* Remove default header background */

header[data-testid="stHeader"] {
    background: rgba(8, 15, 30, 0.85);
}

/* Main content spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Main title */

h1 {
    color: #F1F6FF !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #EAF2FF !important;
    font-weight: 700 !important;
}

p, label, .stCaption {
    color: #A9B8D0;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111D33,
        #0B1426
    );
    border-right: 1px solid #263957;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #67D9FF !important;
}

/* Metric cards */

div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #172944,
        #122139
    );
    border: 1px solid #29476A;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.18);
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: #36C9F5;
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(28, 166, 220, 0.12);
}

div[data-testid="stMetricLabel"] {
    color: #A9BEDB !important;
    font-size: 0.9rem !important;
}

div[data-testid="stMetricValue"] {
    color: #62D8FF !important;
    font-weight: 800 !important;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        90deg,
        #087FCA,
        #13B8D8
    );
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.4rem;
    font-weight: 700;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #0C9FE8,
        #36D5ED
    );
    box-shadow: 0 5px 20px rgba(19, 184, 216, 0.25);
    transform: translateY(-2px);
}

/* Input fields */

.stTextInput input,
.stNumberInput input {
    background-color: #172944 !important;
    color: #EAF2FF !important;
    border: 1px solid #304968 !important;
    border-radius: 8px !important;
}

/* Selectable controls */

.stRadio label,
.stCheckbox label {
    color: #D4E2F5 !important;
}

/* Sliders */

.stSlider [data-testid="stThumbValue"] {
    color: #67D9FF;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background: #13233B;
    border: 1px dashed #36A9D6;
    border-radius: 14px;
    padding: 15px;
}

/* Alert boxes */

div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid #29476A;
}

/* Dividers */

hr {
    border-color: #293D59 !important;
}

/* Video container */

[data-testid="stImage"] {
    border-radius: 14px;
    overflow: hidden;
}

/* Download button */

.stDownloadButton > button {
    background: #172944;
    color: #67D9FF !important;
    border: 1px solid #36A9D6;
    border-radius: 9px;
    font-weight: 600;
}

.stDownloadButton > button:hover {
    background: #203B5D;
    border-color: #67D9FF;
}

/* Progress bar */

.stProgress > div > div > div {
    background: linear-gradient(
        90deg,
        #087FCA,
        #36D5ED
    );
}

/* Plotly chart container */

[data-testid="stPlotlyChart"] {
    background: #13233B;
    padding: 12px;
    border: 1px solid #29476A;
    border-radius: 14px;
}

/* Footer */

footer {
    visibility: hidden;
}

[data-testid="stFileUploader"] {
    background: #12243D !important;
    border: 1px solid #286080 !important;
    border-radius: 14px !important;
    padding: 15px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #172D49 !important;
    border: 1px dashed #36C9F5 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #087FCA !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #13B8D8 !important;
    color: white !important;
}

[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #C5D9F0 !important;
}

/* Uploaded file information */

[data-testid="stFileUploaderFile"] {
    background: #172D49 !important;
    border: 1px solid #286080 !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploaderFile"] span {
    color: #EAF2FF !important;
}

/* Force Streamlit warning box to use dashboard colours */

div[data-testid="stAlert"] {
    background: #172D49 !important;
    background-color: #172D49 !important;
    border: 1px solid #286080 !important;
    border-left: 4px solid #36C9F5 !important;
    border-radius: 12px !important;
}

/* Remove default warning colour from inner elements */

div[data-testid="stAlert"] > div {
    background: transparent !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {
    color: #C5D9F0 !important;
}

/* Warning icon */

div[data-testid="stAlert"] svg {
    color: #36C9F5 !important;
    fill: #36C9F5 !important;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. HEADER
# ==================================================

st.html("""
<div style="background:linear-gradient(110deg,#142B49,#123B59);
border:1px solid #286080;border-radius:18px;
padding:25px 30px;margin-bottom:25px;">

<div style="color:#63D8FF;font-size:12px;
font-weight:800;letter-spacing:3px;margin-bottom:8px;">
INTELLIGENT TRANSPORT SYSTEM
</div>

<div style="color:#F1F6FF;font-size:34px;
font-weight:800;letter-spacing:-1px;">
🚦 AI-Powered Intelligent Traffic Management System
</div>

<div style="color:#B6C9E2;font-size:15px;margin-top:8px;">
Real-time vehicle detection • Adaptive signals •
Emergency priority • Predictive analytics
</div>

</div>
""")

st.warning(
    "Research prototype only. Signal decisions are simulated "
    "and must never be connected to real public-road signals."
)

LANES = ["North", "South", "East", "West"]


# ==================================================
# 4. SIDEBAR CONFIGURATION
# ==================================================

with st.sidebar:

    st.markdown("## ⚙️ Configuration")

    st.caption("Customize traffic monitoring parameters.")

    source = st.radio(
        "Video source",
        ["Upload video", "Webcam"]
    )

    st.divider()

    st.markdown("### Detection Settings")

    confidence = st.slider(
        "YOLO confidence",
        0.1, 0.9, 0.35, 0.05
    )

    vehicle_weights = st.text_input(
        "Vehicle model",
        "yolov8n.pt"
    )

    emergency_weights = st.text_input(
        "Emergency model",
        "models/emergency_best.pt"
    )

    st.divider()

    st.markdown("### Signal Configuration")

    min_green = st.slider(
        "Minimum green (seconds)",
        5, 20, 8
    )

    max_green = st.slider(
        "Maximum green (seconds)",
        20, 90, 45
    )

    st.divider()

    st.markdown("### Processing Settings")

    frame_skip = st.slider(
        "Process every Nth frame",
        1, 10, 3
    )

    st.caption(
        "Higher values improve processing speed but "
        "may miss vehicles or emergency detections."
    )

    max_frames = st.number_input(
        "Maximum video frames",
        min_value=100,
        max_value=100000,
        value=30000,
        step=500
    )

    st.divider()

    st.markdown("### Prediction Settings")

    save_history = st.checkbox(
        "Save traffic observations",
        True
    )

    st.markdown("""
    <div style="
        background:#172944;
        padding:12px;
        border-radius:10px;
        border-left:3px solid #36C9F5;
        margin-top:15px;
    ">
        <span style="color:#67D9FF;font-weight:700;">
            AI SYSTEM
        </span>
        <br>
        <span style="color:#A9BEDB;font-size:12px;">
            Traffic monitoring and predictive analysis
        </span>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# 5. MODEL LOADING
# ==================================================

@st.cache_resource
def get_detector(vw, ew, conf):

    if not os.path.exists(vw):
        raise FileNotFoundError(
            f"Vehicle model not found: {vw}"
        )

    if not os.path.exists(ew):
        raise FileNotFoundError(
            f"Emergency model not found: {ew}"
        )

    return TrafficDetector(vw, ew, conf)


@st.cache_resource
def get_prediction_model():
    return load_model()


# ==================================================
# 6. VIDEO UPLOAD
# ==================================================

uploaded = None

if source == "Upload video":

    st.markdown("### 📁 Video Input")

    uploaded = st.file_uploader(
        "Upload an intersection video",
        type=["mp4", "avi", "mov", "mkv"]
    )


# ==================================================
# 7. DASHBOARD
# ==================================================

st.markdown("## 📊 Traffic Monitoring Dashboard")

start_button = st.button(
    "▶ Start Traffic Monitoring",
    type="primary",
    use_container_width=True
)


# ==================================================
# 8. START MONITORING
# ==================================================

if start_button:

    if source == "Upload video" and uploaded is None:
        st.error("Please upload a video first.")
        st.stop()

    tmp = None

    if source == "Webcam":
        cap = cv2.VideoCapture(0)

    else:

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded.name)[1] or ".mp4"
        )

        tmp.write(uploaded.getvalue())
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)

    if not cap.isOpened():

        st.error("Unable to open video.")

        if tmp is not None:
            os.unlink(tmp.name)

        st.stop()

    # ==============================================
    # LOAD MODELS
    # ==============================================

    try:

        detector = get_detector(
            vehicle_weights,
            emergency_weights,
            confidence
        )

        controller = AdaptiveSignalController(
            min_green,
            max_green
        )

        predictor = get_prediction_model()

        forecaster = (
            CongestionForecaster(predictor)
            if predictor is not None
            else None
        )

    except Exception as e:

        st.error(f"Model loading failed: {e}")

        cap.release()

        if tmp is not None:
            os.unlink(tmp.name)

        st.stop()

    # ==============================================
    # INITIALIZE VARIABLES
    # ==============================================

    last_forecasts = {
        lane: "Collecting history (0/20)"
        for lane in LANES
    }

    if forecaster is None:

        last_forecasts = {
            lane: "Model not trained"
            for lane in LANES
        }

    video_display = st.empty()
    signal_display = st.empty()
    metrics_display = st.empty()
    forecast_display = st.empty()
    statistics_display = st.empty()

    progress = st.progress(0)

    logs = []
    unique_ids = set()
    totals = {}

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_no = 0
    processed_frames = 0

    video_start_time = datetime.now()
    observation_time = video_start_time

    next_sample_time = 15.0

    # ==============================================
    # PREDICTION DISPLAY
    # ==============================================

    forecast_display.subheader(
        "Congestion Prediction - 5 Minutes Ahead"
    )

    forecast_display.info(
        "Waiting for traffic observations..."
    )

    # ==============================================
    # PROCESS VIDEO
    # ==============================================

    try:

        while frame_no < int(max_frames):

            ok, frame = cap.read()

            if not ok:
                break

            frame_no += 1

            if frame_no % frame_skip != 0:
                continue

            processed_frames += 1

            # Vehicle detection

            annotated, counts, vehicles, emergency_lanes = (
                detector.process(frame)
            )

            # Traffic density

            densities = {
                lane: density_label(
                    counts.get(lane, 0)
                )
                for lane in LANES
            }

            # Adaptive signal controller

            decision = controller.decide(
                counts,
                emergency_lanes,
                dt=frame_skip / fps
            )

            # Draw approaches

            annotated = draw_approaches(annotated)

            signal_text = (
                f"SIGNAL: {decision.phase} | "
                f"APPROACH: {decision.active_lane} | "
                f"TIME: {decision.remaining_seconds:.1f}s"
            )

            cv2.putText(
                annotated,
                signal_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 220, 70),
                2
            )

            # Display video

            video_display.image(
                cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                ),
                channels="RGB",
                use_container_width=True
            )

            # Vehicle count cards

            with metrics_display.container():

                cols = st.columns(4)

                for col, lane in zip(cols, LANES):

                    col.metric(
                        lane,
                        f"{counts.get(lane, 0)} vehicles",
                        densities[lane]
                    )

            # Signal status

            signal_display.info(
                f"**Signal phase:** {decision.phase} | "
                f"**Active approach:** {decision.active_lane} | "
                f"**Time remaining:** "
                f"{decision.remaining_seconds:.1f}s | "
                f"**Emergency priority:** "
                f"{'YES' if decision.emergency_priority else 'No'} | "
                f"**Reason:** {decision.reason}"
            )

            # Vehicle statistics

            for vehicle in vehicles:

                if vehicle["id"] >= 0:
                    unique_ids.add(vehicle["id"])

                totals[vehicle["class"]] = (
                    totals.get(vehicle["class"], 0) + 1
                )

            # Collect observations

            elapsed_video_seconds = frame_no / fps

            if elapsed_video_seconds >= next_sample_time:

                observation_time = (
                    video_start_time
                    + timedelta(
                        seconds=elapsed_video_seconds
                    )
                )

                next_sample_time += 15.0

                if save_history:

                    append_observations(
                        "data/traffic_history.csv",
                        counts,
                        densities,
                        observed_at=observation_time
                    )

                # Prediction

                if forecaster is not None:

                    last_forecasts = forecaster.predict(
                        counts,
                        observation_time
                    )

                # Display predictions

                forecast_display.empty()

                forecast_display.subheader(
                    "🔮 Congestion Prediction - 5 Minutes Ahead"
                )

                forecast_cols = forecast_display.columns(4)

                for col, lane in zip(
                    forecast_cols,
                    LANES
                ):

                    col.metric(
                        lane,
                        last_forecasts[lane]
                    )

                forecast_display.caption(
                    f"Last observation: "
                    f"{observation_time.strftime('%H:%M:%S')}"
                )

                # Save logs

                for lane in LANES:

                    logs.append({
                        "timestamp": observation_time,
                        "lane": lane,
                        "vehicle_count": counts.get(lane, 0),
                        "density": densities[lane]
                    })

            # Progress

            if total_frames > 0:

                progress.progress(
                    min(frame_no / total_frames, 1.0)
                )

            elif frame_no % 10 == 0:

                progress.progress(
                    min(frame_no / int(max_frames), 1.0)
                )

            # Statistics

            statistics_display.caption(
                f"Video frames read: {frame_no} | "
                f"Frames processed: {processed_frames} | "
                f"Unique tracker IDs: {len(unique_ids)}"
            )

    except Exception as e:

        st.error(f"Error during video processing: {e}")

    finally:

        cap.release()

        progress.progress(1.0)

        if tmp is not None:

            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # ==============================================
    # FINAL RESULTS
    # ==============================================

    st.divider()

    st.markdown("## 📈 Monitoring Results")

    st.success(
        f"Finished processing {frame_no} video frames."
    )

    a, b = st.columns(2)

    a.metric(
        "Frames actually processed",
        processed_frames
    )

    b.metric(
        "Distinct tracker IDs observed",
        len(unique_ids)
    )

    st.subheader("🚗 Vehicle Detections by Class")

    st.json(totals)

    # Final predictions

    st.subheader(
        "Final Congestion Prediction - 5 Minutes Ahead"
    )

    final_cols = st.columns(4)

    for col, lane in zip(final_cols, LANES):

        col.metric(
            lane,
            last_forecasts[lane]
        )

    # Traffic graph

    if logs:

        df = pd.DataFrame(logs)

        chart = px.line(
            df,
            x="timestamp",
            y="vehicle_count",
            color="lane",
            title="Traffic Density Over Time",
            markers=True,
            color_discrete_sequence=[
                "#36D5ED",
                "#4F8CFF",
                "#A78BFA",
                "#F59E0B"
            ]
        )

        chart.update_layout(
            paper_bgcolor="#13233B",
            plot_bgcolor="#13233B",
            font_color="#EAF2FF",
            title_font_size=20,
            xaxis=dict(
                title="Time",
                gridcolor="#29476A"
            ),
            yaxis=dict(
                title="Vehicle Count",
                gridcolor="#29476A"
            ),
            legend_title="Approach",
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

        # CSV download

        st.download_button(
            "⬇ Download Traffic Monitoring CSV",
            df.to_csv(index=False).encode("utf-8"),
            "traffic_monitoring.csv",
            "text/csv"
        )

    else:

        st.info(
            "No 15-second traffic observations were collected."
        )


# ==================================================
# 9. DEFAULT DASHBOARD
# ==================================================

else:

    st.info(
        "Upload an intersection video and click "
        "'Start Traffic Monitoring' to begin."
    )

    st.markdown("### System Overview")

    a, b, c, d = st.columns(4)

    a.metric("Vehicles Detected", "—")
    b.metric("Traffic Density", "—")
    c.metric("Recommended Signal", "—")
    d.metric("Emergency Alert", "—")

    st.divider()

    st.markdown("Congestion Prediction")

    st.caption(
        "Five-minute congestion predictions will appear "
        "after monitoring begins and sufficient traffic "
        "history has been collected."
    )

    st.html("""
    <div style="background:linear-gradient(110deg,#142B49,#123B59);
    border:1px solid #294F70;border-radius:14px;
    padding:22px;margin-top:15px;">

    <div style="color:#67D9FF;font-size:17px;font-weight:700;">
    Intelligent Traffic Analysis
    </div>

    <p style="color:#B6C9E2;font-size:14px;margin-top:10px;">
    The system analyzes vehicle movement, estimates traffic
    density, prioritizes emergency vehicles and predicts
    upcoming congestion across four intersection approaches.
    </p>

    </div>
    """)

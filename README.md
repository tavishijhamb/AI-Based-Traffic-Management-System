# 🚦 AI-Powered Intelligent Traffic Management System

An AI-based traffic monitoring and management system that uses computer vision to detect and track vehicles, estimate traffic density, simulate adaptive traffic signal allocation, prioritize emergency vehicles, and predict congestion.

---

## 📌 Overview

The project aims to improve intersection traffic management through computer vision and machine learning. It analyzes uploaded traffic videos or webcam feeds to monitor vehicle movement across four intersection approaches: **North, South, East, and West**.

Based on detected vehicle counts and traffic conditions, the system simulates adaptive traffic signal decisions and provides congestion predictions for the upcoming five minutes.

The system integrates vehicle detection, tracking, traffic density estimation, emergency vehicle prioritization, adaptive signal control, and predictive analytics into an interactive Streamlit dashboard.

---

## ✨ Features

- **Vehicle Detection:** Detects vehicles in video feeds using YOLO.
- **Vehicle Tracking:** Tracks detected vehicles and maintains unique tracker IDs.
- **Traffic Density Estimation:** Categorizes traffic conditions for each intersection approach.
- **Adaptive Traffic Signal Control:** Dynamically determines signal allocation based on traffic conditions and vehicle counts.
- **Emergency Vehicle Priority:** Detects emergency vehicles and prioritizes their approaches in simulated signal decisions.
- **Congestion Prediction:** Predicts traffic congestion approximately five minutes ahead using a machine learning model.
- **Real-Time Dashboard:** Displays processed video, vehicle counts, traffic signal status, emergency priority, and congestion forecasts.
- **Traffic Analytics:** Generates traffic monitoring graphs and downloadable CSV reports.
- **Multiple Video Sources:** Supports uploaded videos and webcam input.
- **Configurable Parameters:** Allows adjustment of detection confidence, signal timing, frame processing interval, and processing limits.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Streamlit | Interactive dashboard |
| OpenCV | Video processing and visualization |
| YOLO | Vehicle detection |
| Pandas | Traffic data processing |
| Plotly | Traffic analytics and visualization |
| Scikit-learn | Machine learning |
| NumPy | Numerical operations |

---

## 🏗️ Project Structure

```text
AI-Traffic-Management/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── traffic_core/
│   ├── detector.py
│   ├── controller.py
│   ├── prediction.py
│   ├── history.py
│   └── geometry.py
│
├── models/
│   └── .gitkeep
│
├── data/
│
└── assets/
    └── screenshots/
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Model Setup

The trained model weights are **not included in this repository** because of their file size.

The application requires the following model files:

| Model | Purpose | Location |
|---|---|---|
| `yolov8n.pt` | Vehicle detection | Project root |
| `emergency_best.pt` | Emergency vehicle detection | `models/` directory |

Obtain the required model files separately and place them in the appropriate locations before running the application.

The expected directory structure is:

```text
AI-Traffic-Management/
│
├── app.py
├── yolov8n.pt
│
└── models/
    └── emergency_best.pt
```

The application checks for the required model files during initialization and displays an error if they are missing.

---

## ▶️ Running the Application

After installing the dependencies and placing the required model files, run:

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 🚀 How to Use

1. Launch the application.
2. Select the video source: **Upload Video** or **Webcam**.
3. Configure the YOLO confidence threshold and model paths.
4. Adjust the minimum and maximum green signal durations.
5. Configure the frame processing interval.
6. Upload an intersection video or use the webcam.
7. Click **Start Traffic Monitoring**.
8. Monitor vehicle detections, traffic density, signal decisions, emergency priority, and congestion predictions.
9. View the generated traffic analytics.
10. Download the traffic monitoring data as a CSV file.

---

## 📊 Dashboard Outputs

The dashboard provides the following information:

- Vehicle counts for North, South, East, and West approaches.
- Traffic density classification for each approach.
- Current simulated signal phase.
- Active traffic approach.
- Remaining signal time.
- Emergency vehicle priority status.
- Five-minute congestion forecasts.
- Vehicle detections by class.
- Number of unique tracker IDs observed.
- Traffic variation graphs.
- Downloadable traffic monitoring data.

---

## 🔮 Congestion Prediction

The prediction module uses traffic observations to estimate congestion approximately five minutes into the future.

Traffic observations are collected periodically during video processing and can be stored for machine learning purposes.

Predictions depend on the availability of a trained forecasting model and sufficient traffic history. If the prediction model is unavailable or the required history has not been collected, predictions may not be generated.

---

## 📈 Traffic Analytics

The system records traffic observations for all four intersection approaches, including:

- Timestamp
- Lane/approach
- Vehicle count
- Traffic density

The collected observations are visualized using Plotly and can be exported as a CSV file for further analysis.

---

## ⚠️ Limitations

- Traffic signal decisions are simulated and are not connected to real-world traffic infrastructure.
- Vehicle detection accuracy depends on video quality, lighting, camera positioning, occlusion, and model performance.
- Emergency vehicle detection depends on the trained model and supported detection classes.
- Congestion prediction depends on the quality and availability of training data.
- Processing speed depends on hardware capabilities, video resolution, and frame processing settings.
- The application requires model weights that are not included in the repository.
- The current implementation does not include SUMO traffic simulation integration.

---

## 🔬 Future Scope

- Integration with traffic simulation platforms such as SUMO.
- Evaluation on additional real-world traffic datasets.
- Improved emergency vehicle detection and classification.
- Optimization of traffic signal timing strategies.
- Improved congestion forecasting models.
- Multi-intersection traffic coordination.
- Edge-device deployment.
- Integration with real-time traffic monitoring infrastructure for research purposes.

---

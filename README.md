# 🔍 Crime Pattern Analysis & Prediction System

A comprehensive system for analyzing crime patterns, predicting hotspots, and planning safe routes using Machine Learning and Geospatial Data.

## 🚀 Features

### 📊 Interactive Dashboard
- **Overview**: High-level metrics on crime rates, severity, and trends.
- **Geospatial Analysis**: Interactive maps to visualize crime hotspots and distributions.
- **Temporal Analysis**: Analyze crime patterns by time of day, day of week, and seasonal trends.
- **Data Exploration**: Filter and explore the raw dataset with ease.

### 🛡️ Route Safety API
- **Safe Route Planning**: Flask-based API to calculate the safest path between two points.
- **Risk Assessment**: Real-time risk scoring for different routes based on historical crime data.

### 🧠 Advanced Analytics
- **Machine Learning Models**: Utilizes **LightGBM** and **Scikit-learn** for predictive modeling.
- **Pattern Recognition**: Identifies high-risk areas and emerging crime trends.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) 🎈
- **Backend API**: [Flask](https://flask.palletsprojects.com/) 🌶️
- **Data Processing**: Pandas 🐼, NumPy 🔢
- **Visualization**: Plotly 📈, Folium 🗺️
- **Machine Learning**: Scikit-learn 🤖, LightGBM ⚡

## 📂 Project Structure

```
file_structure
├── app/
│   ├── main_app.py          # Streamlit Dashboard Entry Point
│   └── route_safety_api.py  # Flask API for Safe Routes
├── src/                     # Core Logic & Data Processing
├── data/                    # Datasets
├── notebooks/               # Jupyter Notebooks for Analysis
└── requirements.txt         # Project Dependencies
```

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/latheshkumarsr/2022_BAI786_06.git
   cd 2022_BAI786_06
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

### Launch the Dashboard
Run the Streamlit app to view the interactive dashboard:
```bash
streamlit run app/main_app.py
```

### Start the Safety API
Run the Flask API server:
```bash
python app/route_safety_api.py
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License.

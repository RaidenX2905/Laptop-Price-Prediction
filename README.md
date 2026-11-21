<div align="center">

# Laptop Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](#) [![scikit--learn](https://img.shields.io/badge/scikit--learn-0.88%20R2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](#) [![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#) [![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat-square&logo=pandas&logoColor=white)](#) [![NumPy](https://img.shields.io/badge/NumPy-Numerics-013243?style=flat-square&logo=numpy&logoColor=white)](#) [![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](#)

<p align="center">
  End-to-end Machine Learning pipeline predicting laptop market prices based on hardware specifications with an ensemble Random Forest regression model.
</p>

</div>

---

**Live Demo**: [https://lppbyjsh.streamlit.app/](https://lppbyjsh.streamlit.app/)

---

## Pipeline Architecture

1. **Data Ingestion**: Processes multi-brand catalog specifications (`laptops.csv`).
2. **Preprocessing**: Median imputation for missing ratings and categorical feature encoding (`LabelEncoder`).
3. **Model Selection**: Benchmarked against Linear Regression and Decision Trees; Random Forest Regressor selected achieving an **R^2 score of ~0.88**.
4. **Interactive Deployment**: Clean Streamlit dashboard for real-time spec configuration and instant price forecasting.

---

## Quick Start

### Installation

```bash
git clone https://github.com/RaidenX2905/Laptop-Price-Prediction.git
cd Laptop-Price-Prediction
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Repository Structure

```text
├── data/                  # Dataset with raw laptop specifications
├── notebooks/             # Exploratory analysis and training notebook
├── app.py                 # Streamlit deployment entry point
└── requirements.txt       # Python dependencies
```

---

## License

MIT License

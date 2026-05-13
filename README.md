# 🎓 Student Grade Predictor

A machine learning application that predicts student final grades based on demographic and academic factors using a Random Forest model with explainable AI (SHAP).

---

## 📋 Project Overview

This project demonstrates a complete ML pipeline with:
- **Data Processing**: Encoding categorical variables
- **Model Training**: Random Forest Regressor with scikit-learn
- **Model Deployment**: Interactive Streamlit web application
- **Model Explainability**: SHAP (SHapley Additive exPlanations) for interpretability
- **Feature Importance**: Visual analysis of feature contributions

The application predicts a student's final grade (G3) on a scale of 0-20 based on:
- **Demographics**: School, sex, age, address, family size
- **Family Info**: Parental education, occupation, status
- **Academic History**: Previous grades (G1, G2), failures, study time
- **Lifestyle**: Absences, alcohol consumption, romantic relationships

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bhawanachapagain/StudentMLProject.git
   cd StudentMLProject
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup

Place your dataset in the `data/` folder:
```bash
data/
└── student-por.csv
```

The dataset should include student records with columns like `school`, `sex`, `age`, `G1`, `G2`, `G3`, etc.

---

## 🛠️ Usage

### 1. Train the Model
```bash
python train_model.py
```
This will:
- Load the student data
- Preprocess categorical variables
- Train a Random Forest model
- Save the model to `models/student_grade_model.pkl`

### 2. Run the Streamlit App
```bash
streamlit run app.py
```
Then open your browser and navigate to `http://localhost:8501`

### 3. Make Predictions
- Fill in the student information form
- Click **"🎯 Predict Grade"** button
- View:
  - Predicted final grade
  - Top factors influencing the prediction (SHAP)
  - Feature importance rankings

---

## 📁 Project Structure

```
StudentMLProject/
├── app.py                          # Streamlit web app
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── data/
│   └── student-por.csv            # Student dataset (Portuguese course)
├── models/
│   └── student_grade_model.pkl     # Trained model (generated after training)
└── README.md                       # This file
```

---

## 🔬 Technical Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | scikit-learn |
| **Model** | Random Forest Regressor |
| **Data Processing** | pandas, numpy |
| **Explainability** | SHAP |
| **Visualization** | matplotlib, streamlit |
| **Web App** | Streamlit |
| **Model Serialization** | joblib |

---

## 📊 Model Details

### Preprocessing
- **OneHotEncoder**: Transforms categorical variables (school, sex, job types, etc.)
- **ColumnTransformer**: Applies encoding to selected columns while preserving numeric features

### Model
- **Algorithm**: Random Forest Regressor
- **Random State**: 42 (for reproducibility)
- **Output**: Grade prediction (0-20)

### Explainability
- **SHAP TreeExplainer**: Explains individual predictions
- **Feature Importance**: Shows which features matter most across all predictions

---

## 📈 Features

✅ **Interactive UI**: User-friendly Streamlit interface  
✅ **Real-time Predictions**: Instant grade predictions  
✅ **Model Explainability**: Understand why the model predicted a specific grade  
✅ **Feature Analysis**: Identify key factors influencing grades  
✅ **Reproducible**: Fixed random seed for consistent results  
✅ **Scalable**: Can be deployed to Streamlit Cloud or other platforms  

---

## 💡 Key Insights

The application provides:
1. **Individual Prediction Explanation**: Why did the model predict this specific grade for this student?
2. **Global Feature Importance**: Which student characteristics matter most in general?
3. **Visual Feedback**: Bar charts showing SHAP values and feature importances

---

## 🔄 Prediction Workflow

```
Student Input
    ↓
Data Validation & Filling Defaults
    ↓
One-Hot Encoding (Preprocessing)
    ↓
Random Forest Prediction
    ↓
Grade Clipping (0-20 range)
    ↓
SHAP Explanation Generation
    ↓
Visualization & Display
```

---

## 📦 Dependencies

Core packages:
- `streamlit` - Web application framework
- `scikit-learn` - Machine learning
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `joblib` - Model persistence
- `shap` - Model explainability
- `matplotlib` - Visualization

Full list in `requirements.txt`

---

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repository and branch
4. Deploy!

### Docker (Local)
```bash
docker build -t student-predictor .
docker run -p 8501:8501 student-predictor
```

---

## 🔮 Future Enhancements

- [ ] Cross-validation metrics dashboard
- [ ] Model comparison (XGBoost, LightGBM)
- [ ] Batch prediction upload (CSV)
- [ ] Model performance metrics display
- [ ] Data drift monitoring
- [ ] User authentication & logging
- [ ] API endpoints (FastAPI)

---

## 📚 Data Source

The project uses student performance data with attributes like:
- `G1`: First period grade
- `G2`: Second period grade
- `G3`: Final grade (target variable)
- Student demographics and lifestyle factors

Typical dataset: ~400-600 student records

---

## ⚙️ How SHAP Works

**SHAP (SHapley Additive exPlanations)** provides:
- **Local Explanations**: Why this specific prediction?
- **Shapley Values**: Contribution of each feature to the prediction
- **Positive Values**: Push grade up
- **Negative Values**: Push grade down

Example: "High absences SHAP -2.5" means absences lower the predicted grade by 2.5 points.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Bhawana Chapagain**  
GitHub: [@bhawanachapagain](https://github.com/bhawanachapagain)

---

## 📞 Support

For issues, questions, or suggestions:
- Open an [issue](https://github.com/bhawanachapagain/StudentMLProject/issues)
- Check existing issues first

---

## 🎯 Use Cases

- **Educational Analytics**: Understand factors affecting student performance
- **Early Intervention**: Identify at-risk students
- **Research**: Study relationships between student characteristics and grades
- **Portfolio Project**: Demonstrate ML, data science, and web development skills

---

**⭐ If you found this helpful, please star the repository!**

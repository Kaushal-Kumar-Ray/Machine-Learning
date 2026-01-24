# 💰 Gold Price Prediction using Machine Learning & Streamlit

This project predicts the **gold price (GLD)** using machine learning based on key **macroeconomic and market indicators**.  
It also includes a **Streamlit web dashboard** where users can input values and get instant gold price predictions.

---

## 📌 Project Overview

Gold prices are influenced by multiple economic factors such as:
- Stock market performance
- Oil prices
- Silver prices
- Currency exchange rates

In this project, we:
- Trained a **Random Forest Regression** model on historical market data
- Evaluated it using proper **time-series splitting**
- Deployed the trained model using **Streamlit** for real-time predictions

---

## 🧠 Machine Learning Model

- **Model Used:** Random Forest Regressor  
- **Target Variable:** `GLD` (Gold ETF price)  
- **Features Used:**
  - `SPX` – S&P 500 Index
  - `USO` – Oil Price ETF
  - `SLV` – Silver Price ETF
  - `EUR/USD` – Currency Exchange Rate

---

## 📂 Project Structure

gold-price-predictor/
│
├── app.py # Streamlit dashboard
├── gold_price_model.pkl # Trained ML model
├── scaler.pkl # Feature scaler
├── gld_price_data.csv # Dataset
├── GPP.ipynb
└── README.md # Project documentation



---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
git clone "github file link"
cd gold-price-predictor

2️⃣ Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt
▶️ Running the Streamlit App
⚠️ Do NOT use python app.py
✅ Correct command:
streamlit run app.py  (  !! "ensure env is activated")
After running the command:
A browser window will open automatically
If not, visit: http://localhost:8501


🧪 How the App Works
User enters:

                        get the data from 
S&P 500 Index value (SPX)  ]----Index points                 4200 – 5200
Oil Price (USO)            ]----USO ETF price (USD)          40 – 90
Silver Price (SLV)         ]----Unit: SLV ETF price (USD)    15 – 35
EUR / USD Exchange Rate    ]----Exchange rate                1.05 – 1.20
<<Gold share today value us>  check online

Inputs are scaled using the same scaler used during training
The trained ML model predicts the gold price
Result is displayed instantly

📊 Model Evaluation Metrics
R² Score: Measures how well the model explains gold price movement
MAE: Average prediction error
RMSE: Penalizes large errors
The model achieves high accuracy due to strong correlation between gold and the selected features.

🚀 Technologies Used
Python
Pandas & NumPy
Scikit-learn
Matplotlib & Seaborn
Streamlit
Joblib

📈 Future Improvements
Add lag features for time-series learning
Use XGBoost or LSTM models
Integrate live market data APIs
Deploy app on Streamlit Cloud or Hugging Face Spaces
Add interactive charts to the dashboard

💡 Disclaimer
This project is for educational purposes only.
Predictions are based on historical patterns and should not be used for financial trading decisions.

👨‍💻 Author
Built by [Kaushal Kumar Ray]
Machine Learning & Data Science Enthusiast 🚀
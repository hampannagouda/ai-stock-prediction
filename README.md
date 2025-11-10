# 🧠 AI Stock Prediction System — Next Day Close Forecast (LSTM)

A deep learning-based stock market prediction system that uses **LSTM (Long Short-Term Memory)** networks to predict the **next day’s closing price** for any listed stock.  
Built with Python, TensorFlow, and Streamlit — this project combines data science, AI, and web deployment skills.

---

## 📊 Features
✅ Real-time data fetching using `yfinance`  
✅ Technical indicators (SMA, EMA, RSI, MACD, etc.)  
✅ Time-series prediction using **LSTM neural network**  
✅ Interactive **Streamlit web dashboard**  
✅ Visual comparison of actual vs predicted prices  
✅ Modular code structure for easy enhancement  

---

## 🧰 Tech Stack
| Component | Technology |
|------------|-------------|
| Language | Python 3.11 |
| ML Framework | TensorFlow / Keras |
| Visualization | Matplotlib, Seaborn |
| Data Source | Yahoo Finance API (`yfinance`) |
| Web App | Streamlit |
| Environment | VS Code + Virtualenv |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/hampannagouda/ai-stock-prediction.git
cd ai-stock-prediction

### 2️⃣ Create a Virtual Environment
python -m venv venv

Activate it:

Windows (Git Bash):

source venv/Scripts/activate


Command Prompt:

venv\Scripts\activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

🧠 Train the Model

python train_lstm.py

This script:

Downloads stock data

Adds technical indicators

Trains an LSTM model

Saves model and scalers to /models folder

Output example:

MAE: 3.12, RMSE: 4.89
✅ Model and scalers saved successfully.

Launch the Streamlit App
streamlit run app.py

Open your browser at http://localhost:8501

You’ll see an interactive dashboard where you can:

Enter any stock ticker (AAPL, TSLA, INFY.NS)

Get predicted next-day closing price

View line charts of past and predicted trends

### 🧩 Project Structure
ai-stock-prediction/
│
├── data_pipeline.py        # Data download, preprocessing, feature creation
├── train_lstm.py           # Model training & evaluation
├── app.py                  # Streamlit web UI
├── requirements.txt
├── README.md
└── models/
    ├── lstm_model.h5
    ├── scaler_x.save
    └── scaler_y.save

### 🚀 Future Enhancements

Add news sentiment analysis (NLP + Finance data)

Implement Transformer-based models

Integrate real-time stock streaming (WebSocket)

Deploy on Streamlit Cloud / Render / AWS

### 🧑‍💻 Author

Hampannagouda
B.Tech in Computer Science & Engineering, Dayanand Sagar University
Passionate about AI, ML, and full-stack development.

### ⭐ If you like this project, give it a star on GitHub!


---

## ✅ Next Steps for You

1. Create the file `README.md` in your repo → paste the content above.  
2. Commit & push to GitHub:
   ```bash
   git add .
   git commit -m "Added README.md and project setup completed"
   git push origin main


Run python train_lstm.py to train the model once.

Run streamlit run app.py to test your web app.

Thank you.
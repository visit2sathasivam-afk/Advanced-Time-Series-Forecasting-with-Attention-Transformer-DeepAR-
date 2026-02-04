📈 Advanced Time Series Forecasting with Attention (TensorFlow)
-----------------------------------------------------------------

This project implements an advanced multivariate time series forecasting system using a Transformer-based attention model in TensorFlow/Keras. The goal is to forecast future values from multiple correlated time series and compare the performance of an attention-based model against a strong baseline (LSTM).

The project demonstrates how self-attention captures long-range temporal dependencies better than traditional recurrent architectures.

🚀 Project Objectives
----------------------
Generate a complex multivariate time series dataset with trend and seasonality.

Preprocess data using normalization and sliding windows.

Build a Transformer (self-attention) forecasting model.

Build a baseline LSTM model for comparison.

Train both models using TensorFlow.

Evaluate performance using RMSE, MAE, and WAPE.

Visualize and analyze attention weights.

Document architectural choices and results.

🗂 Project Structure
-----------------------
ts_attention_project/
│
├── data_generator.py        # Generate synthetic multivariate data
├── preprocess.py           # Scaling and windowing
├── transformer_model.py    # Transformer with self-attention
├── lstm_baseline.py        # LSTM baseline model
├── train_transformer.py    # Train transformer model
├── train_lstm.py           # Train LSTM baseline
├── evaluate.py             # Metrics and comparison
├── visualize_attention.py  # Attention weight visualization
├── main.py                 # End-to-end pipeline
├── report.md               # Final project report
└── README.md               # Project documentation


(You may also run everything in a single notebook if preferred.)

⚙️ Requirements
-----------------------
Install dependencies before running:

pip install tensorflow numpy pandas scikit-learn matplotlib

📊 Dataset Generation
-----------------------
The dataset is synthetically generated with:

Minimum 3 correlated series

Trend component

Seasonality component

Gaussian noise

More than 1000 time steps

Example shape:

(time_steps, features) → (1500, 3)


This simulates realistic scenarios such as energy demand, stock prices, or sensor signals.

🔄 Preprocessing
-------------------------
Steps:

Normalize data using MinMaxScaler.

Apply sliding window technique.

Convert time series into supervised learning format:

Past 30 steps → Predict next 10 steps


Resulting tensors:

X → (samples, past_steps, features)
Y → (samples, future_steps, features)

🧠 Transformer Architecture
----------------------------
The core model uses:

Input embedding layer

Positional-aware representation

Multi-Head Self-Attention

Feed-forward network

Layer normalization & dropout

Dense output projection

This enables the model to focus on important historical time points dynamically.

🧪 Baseline Model
----------------------------
A standard LSTM network is implemented as a benchmark.
Both Transformer and LSTM are trained on the same dataset for a fair comparison.

🏋️ Training
----------------------------
Optimizer: Adam

Loss: Mean Squared Error

Batch size: 32

Epochs: 30

Validation split: 0.2

📏 Evaluation Metrics
---------------------------
The models are evaluated using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

WAPE (Weighted Absolute Percentage Error)

These metrics measure forecasting accuracy and robustness.

🔍 Attention Analysis
----------------------------
The Transformer exposes attention weights which are visualized as heatmaps.
This helps interpret:

Which past timesteps influence predictions.

How seasonality and trends are captured.

How long-range dependencies are learned.

▶️ How to Run
----------------------------
Run the full pipeline:

python main.py


Or train models separately:

python train_transformer.py
python train_lstm.py
python evaluate.py
python visualize_attention.py

📄 Report
----------------------------
The final report (report.md) includes:

Data generation strategy

Preprocessing steps

Model architecture

Hyperparameters

Performance comparison

Attention visualization analysis

Conclusion

✅ Key Outcomes
------------------------------
Transformer captures temporal patterns more effectively than LSTM.

Attention improves long-range dependency modeling.

Provides interpretable forecasting via attention weights.

🛠 Technologies Used
------------------------------
Python

TensorFlow / Keras

NumPy

Scikit-learn

Matplotlib

📌 Author
-----------------------------

Sathasivam Murugesan

Advanced Time Series Forecasting Project
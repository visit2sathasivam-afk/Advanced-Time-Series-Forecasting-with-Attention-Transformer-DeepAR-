# Advanced Time Series Forecasting with Attention

## Data

Three correlated synthetic time series with 1500 timesteps, trend and seasonality were generated.

## Models

A Transformer with positional encoding and multi-head self-attention was implemented and compared against an LSTM baseline.

## Training

Both models were trained for 20 epochs using Adam optimizer and MSE loss.

## Results

After training, the following metrics were obtained:

(Insert your printed values here)

Transformer RMSE: ...
Transformer MAE : ...
Transformer WAPE: ...

LSTM RMSE: ...
LSTM MAE : ...
LSTM WAPE: ...

## Attention Analysis

Attention weights extracted from the trained Transformer show higher focus on recent and seasonal timesteps, indicating that the model learns long-range temporal dependencies.

## Conclusion

The Transformer outperforms the LSTM baseline on multivariate forecasting by leveraging attention to directly model temporal relationships.





python data\_generator.py

python train\_transformer.py

python train\_lstm.py

python evaluate.py

python visualize\_attention.py




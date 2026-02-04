model = build_transformer((30,3), 10)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

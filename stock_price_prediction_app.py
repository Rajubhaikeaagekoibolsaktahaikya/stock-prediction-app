# ... same imports and st setup as before ...

if st.button("Predict Stock Price"):
    with st.spinner("Fetching data and predicting..."):
        # Download historical data
        df = yf.download(ticker, start="2018-01-01", end="2024-12-31")

        if df.empty:
            st.error("❌ No data found for the entered ticker. Please try a different symbol.")
        else:
            df['Tomorrow'] = df['Close'].shift(-1)
            df.dropna(inplace=True)

            X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            y = df['Tomorrow']

            # Check if sufficient data is available
            if len(X) < 2 or len(y) < 2:
                st.error("❌ Not enough data after processing. Try a different stock or time range.")
            else:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluation
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Display results
                st.success(f"✅ Mean Squared Error (MSE): {mse:.2f}")
                st.success(f"✅ R² Score: {r2:.2f}")

                # Plotting
                st.subheader("📊 Actual vs Predicted Closing Prices")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(y_test.values, label='Actual Price', color='blue')
                ax.plot(y_pred, label='Predicted Price', color='green', linestyle='--')
                ax.set_xlabel("Test Data Points")
                ax.set_ylabel("Price (INR)")
                ax.set_title(f"Stock Price Prediction for {ticker}")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

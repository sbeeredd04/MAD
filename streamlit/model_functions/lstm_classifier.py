from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

def reshape_data_for_lstm(X, timesteps=3):
    """Reshape data into 3D format required for LSTM: (samples, timesteps, features)"""
    samples = X.shape[0] - timesteps + 1
    features = X.shape[1]
    X_reshaped = np.zeros((samples, timesteps, features))
    
    for i in range(samples):
        X_reshaped[i] = X[i:i+timesteps]
    
    return X_reshaped

def train_lstm(X_train, y_train, X_test, y_test, X_val=None, y_val=None, **params):
    """
    Train an LSTM model and return predictions and model details.
    """
    # Extract parameters
    timesteps = params.get('timesteps', 3)
    units = params.get('units', 50)
    n_layers = params.get('layers', 1)
    dropout = params.get('dropout', 0.2)
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', 32)
    
    # Scale features first
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to scaled training data
    sm = SMOTE(random_state=42, sampling_strategy=0.50)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)
    
    # Reshape SMOTE-enhanced data for LSTM
    X_train_reshaped = reshape_data_for_lstm(X_train_sm, timesteps)
    X_test_reshaped = reshape_data_for_lstm(X_test_scaled, timesteps)
    
    # Adjust target variables
    y_train_adjusted = y_train_sm[timesteps-1:]
    y_test_adjusted = y_test[timesteps-1:]
    
    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_reshaped = reshape_data_for_lstm(X_val_scaled, timesteps)
        y_val_adjusted = y_val[timesteps-1:]
        validation_data = (X_val_reshaped, y_val_adjusted)
    
    # Build and train model (same as before)
    model = Sequential()
    
    # Add LSTM layers
    for i in range(n_layers):
        if i == 0:
            model.add(LSTM(units, 
                          return_sequences=(n_layers > 1),
                          input_shape=(timesteps, X_train.shape[1])))
        elif i == n_layers - 1:
            model.add(LSTM(units))
        else:
            model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train_reshaped, y_train_adjusted,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2 if validation_data is None else 0.0,
        validation_data=validation_data,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Generate predictions
    test_pred_proba = model.predict(X_test_reshaped, verbose=0)
    test_pred = (test_pred_proba > 0.5).astype(int)
    
    # Generate validation predictions if provided
    val_pred = None
    val_pred_proba = None
    if validation_data is not None:
        val_pred_proba = model.predict(X_val_reshaped, verbose=0)
        val_pred = (val_pred_proba > 0.5).astype(int)
    
    # Feature importance through permutation
    importances = np.zeros(X_train.shape[1])
    baseline_score = model.evaluate(X_test_reshaped, y_test_adjusted, verbose=0)[1]
    
    for i in range(X_train.shape[1]):
        X_test_permuted = X_test_reshaped.copy()
        X_test_permuted[:, :, i] = np.random.permutation(X_test_permuted[:, :, i])
        score = model.evaluate(X_test_permuted, y_test_adjusted, verbose=0)[1]
        importances[i] = baseline_score - score
    
    feat_importances = pd.Series(importances, index=X_train.columns)
    feat_importances.sort_values(ascending=False, inplace=True)
    
    # Return adjusted y_test along with predictions for proper metric calculation
    return {
        'model': model,
        'test_predictions': test_pred.flatten(),
        'test_predictions_proba': test_pred_proba.flatten(),
        'val_predictions': val_pred.flatten() if val_pred is not None else None,
        'val_predictions_proba': val_pred_proba.flatten() if val_pred_proba is not None else None,
        'cv_scores': np.array([history.history['val_accuracy'][-1]]),
        'feature_importances': feat_importances,
        'training_history': history.history,
        'y_test_adjusted': y_test_adjusted
    }
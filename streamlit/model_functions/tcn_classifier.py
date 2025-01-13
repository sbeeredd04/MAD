from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def reshape_data_for_tcn(X, sequence_length):
    """Reshape data into 3D format required for TCN: (samples, timesteps, features)"""
    samples = X.shape[0] - sequence_length + 1
    features = X.shape[1]
    X_reshaped = np.zeros((samples, sequence_length, features))
    
    for i in range(samples):
        X_reshaped[i] = X[i:i+sequence_length]
    
    return X_reshaped

def create_tcn_layer(input_shape, nb_filters, kernel_size, nb_stacks, dilations, dropout_rate):
    """Create a custom TCN-like architecture using Conv1D layers"""
    x = input_layer = Input(shape=input_shape)
    
    for s in range(nb_stacks):
        for d in dilations:
            # Dilated causal convolution
            conv = Conv1D(
                filters=nb_filters,
                kernel_size=kernel_size,
                dilation_rate=d,
                padding='causal',
                activation='relu'
            )(x)
            # Add dropout
            x = Dropout(dropout_rate)(conv)
    
    # Add global pooling to reduce sequence dimension
    x = GlobalAveragePooling1D()(x)
    
    return input_layer, x

def train_tcn(X_train, y_train, X_test, y_test, X_val=None, y_val=None, **params):
    """
    Train a TCN model and return predictions and model details.
    """
    # Extract parameters with defaults
    sequence_length = params.get('sequence_length', 10)
    nb_filters = params.get('nb_filters', 64)
    kernel_size = params.get('kernel_size', 3)
    nb_stacks = params.get('nb_stacks', 1)
    dilations = params.get('dilations', [1, 2, 4, 8])
    dropout_rate = params.get('dropout_rate', 0.2)
    learning_rate = params.get('learning_rate', 0.001)
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', 32)
    
# Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to scaled training data
    sm = SMOTE(random_state=42, sampling_strategy=0.50)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)
    
    # Reshape SMOTE-enhanced data for TCN
    X_train_reshaped = reshape_data_for_tcn(X_train_sm, sequence_length)
    X_test_reshaped = reshape_data_for_tcn(X_test_scaled, sequence_length)
    
    # Adjust target variables for SMOTE-enhanced data
    y_train_adjusted = y_train_sm[sequence_length-1:]
    y_test_adjusted = y_test[sequence_length-1:]
    
    # Prepare validation data if provided
    validation_data = None
    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_reshaped = reshape_data_for_tcn(X_val_scaled, sequence_length)
        y_val_adjusted = y_val[sequence_length-1:]
        validation_data = (X_val_reshaped, y_val_adjusted)
    
    # Build custom TCN model
    input_shape = (sequence_length, X_train.shape[1])
    input_layer, tcn_output = create_tcn_layer(
        input_shape, 
        nb_filters, 
        kernel_size, 
        nb_stacks, 
        dilations, 
        dropout_rate
    )
    
    # Add dense layers for classification
    x = Dense(32, activation='relu')(tcn_output)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
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
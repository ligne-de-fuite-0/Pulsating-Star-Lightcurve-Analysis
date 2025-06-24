# -*- coding: utf-8 -*-
"""
GRU Autoencoder for lightcurve anomaly detection.

This module implements a GRU-based autoencoder for detecting anomalies
in variable star lightcurves.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
from lightkurve import LightCurve


class GRUAutoencoder:
    """GRU-based autoencoder for lightcurve anomaly detection."""
    
    def __init__(self, encoding_dim: int = 16, 
                 activation: str = 'relu',
                 optimizer: str = 'adam',
                 regularization: Optional[str] = None,
                 lambda_reg: float = 0.001):
        """
        Initialize GRU autoencoder.
        
        Parameters:
        -----------
        encoding_dim : int
            Dimension of the encoding layer
        activation : str
            Activation function
        optimizer : str
            Optimizer name
        regularization : str, optional
            Regularization type ('l1', 'l2', or None)
        lambda_reg : float
            Regularization strength
        """
        self.encoding_dim = encoding_dim
        self.activation = activation
        self.optimizer = optimizer
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.scaler = MinMaxScaler()
        self.input_dim = None
        
    def _build_model(self, input_dim: int) -> None:
        """Build the GRU autoencoder model."""
        self.input_dim = input_dim
        timesteps = input_dim
        features = 1
        
        # Encoder
        inputs = Input(shape=(timesteps, features), name='encoder_input')
        
        if self.regularization == 'l1':
            encoded = GRU(self.encoding_dim, activation=self.activation,
                         activity_regularizer=regularizers.l1(self.lambda_reg),
                         name='encoder_gru')(inputs)
        elif self.regularization == 'l2':
            encoded = GRU(self.encoding_dim, activation=self.activation,
                         activity_regularizer=regularizers.l2(self.lambda_reg),
                         name='encoder_gru')(inputs)
        else:
            encoded = GRU(self.encoding_dim, activation=self.activation, 
                         name='encoder_gru')(inputs)
        
        self.encoder = Model(inputs, encoded, name='encoder')
        
        # Decoder
        latent_inputs = Input(shape=(self.encoding_dim,), name='decoder_input')
        repeated = RepeatVector(timesteps)(latent_inputs)
        decoded = GRU(64, activation=self.activation, return_sequences=True, 
                     name='decoder_gru')(repeated)
        decoded_outputs = TimeDistributed(Dense(features), 
                                        name='decoder_output')(decoded)
        
        self.decoder = Model(latent_inputs, decoded_outputs, name='decoder')
        
        # Full autoencoder
        autoencoder_outputs = self.decoder(self.encoder(inputs))
        self.autoencoder = Model(inputs, autoencoder_outputs, name='gru_autoencoder')
        
        # Compile
        if self.optimizer == 'adam':
            opt = Adam()
        else:
            opt = self.optimizer
            
        self.autoencoder.compile(optimizer=opt, loss='mse')
        
    def fit(self, lightcurves: List[LightCurve], 
            epochs: int = 100, 
            batch_size: int = 32,
            validation_split: float = 0.2,
            verbose: int = 1) -> Dict:
        """
        Train the autoencoder on lightcurve data.
        
        Parameters:
        -----------
        lightcurves : list of LightCurve
            Training lightcurves
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data for validation
        verbose : int
            Verbosity level
            
        Returns:
        --------
        dict
            Training history
        """
        # Prepare data
        X_train = self._prepare_data(lightcurves)
        
        # Build model if not exists
        if self.autoencoder is None:
            self._build_model(X_train.shape[1])
            
        # Train
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            verbose=verbose
        )
        
        return history.history
        
    def predict(self, lightcurves: List[LightCurve]) -> np.ndarray:
        """
        Predict reconstruction errors for lightcurves.
        
        Parameters:
        -----------
        lightcurves : list of LightCurve
            Input lightcurves
            
        Returns:
        --------
        np.ndarray
            Reconstruction errors (MSE)
        """
        if self.autoencoder is None:
            raise ValueError("Model must be fitted before prediction")
            
        X = self._prepare_data(lightcurves, fit_scaler=False)
        reconstructed = self.autoencoder.predict(X)
        
        # Calculate MSE for each lightcurve
        mse_errors = np.mean((X - reconstructed) ** 2, axis=(1, 2))
        return mse_errors
        
    def _prepare_data(self, lightcurves: List[LightCurve], 
                     fit_scaler: bool = True) -> np.ndarray:
        """Prepare lightcurve data for training/prediction."""
        # Extract flux values
        flux_data = []
        for lc in lightcurves:
            flux = lc.flux.value
            flux_data.append(flux)
            
        # Convert to array and pad/truncate to same length
        max_len = max(len(flux) for flux in flux_data)
        
        X = np.zeros((len(flux_data), max_len))
        for i, flux in enumerate(flux_data):
            if len(flux) >= max_len:
                X[i] = flux[:max_len]
            else:
                X[i, :len(flux)] = flux
                
        # Normalize
        if fit_scaler:
            X_reshaped = X.reshape(-1, 1)
            self.scaler.fit(X_reshaped)
            X_scaled = self.scaler.transform(X_reshaped).reshape(X.shape)
        else:
            X_reshaped = X.reshape(-1, 1)
            X_scaled = self.scaler.transform(X_reshaped).reshape(X.shape)
            
        # Reshape for GRU: (samples, timesteps, features)
        X_final = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        return X_final
        
    def evaluate_encoding_dimensions(self, lightcurves: List[LightCurve],
                                   encoding_dims: List[int],
                                   epochs: int = 50) -> Dict[int, float]:
        """
        Evaluate reconstruction error for different encoding dimensions.
        
        Parameters:
        -----------
        lightcurves : list of LightCurve
            Input lightcurves
        encoding_dims : list of int
            Encoding dimensions to test
        epochs : int
            Training epochs per dimension
            
        Returns:
        --------
        dict
            Mapping of encoding dimension to reconstruction error
        """
        results = {}
        
        for dim in encoding_dims:
            print(f"Testing encoding dimension: {dim}")
            
            # Create new model with this dimension
            model = GRUAutoencoder(
                encoding_dim=dim,
                activation=self.activation,
                optimizer=self.optimizer,
                regularization=self.regularization,
                lambda_reg=self.lambda_reg
            )
            
            # Train
            model.fit(lightcurves, epochs=epochs, verbose=0)
            
            # Evaluate
            errors = model.predict(lightcurves)
            mean_error = np.mean(errors)
            results[dim] = mean_error
            
            print(f"Mean reconstruction error: {mean_error:.6f}")
            
        return results
        
    def plot_training_history(self, history: Dict) -> None:
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def summary(self) -> None:
        """Print model summary."""
        if self.autoencoder is not None:
            self.autoencoder.summary()
        else:
            print("Model not built yet. Call fit() first.")
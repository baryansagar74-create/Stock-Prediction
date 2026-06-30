from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.optimizers import Adam
from app.utils import logger

def build_ann_model(input_shape: tuple, learning_rate: float = 0.001) -> Sequential:
    """
    Builds and compiles the Multi-layer Feedforward Neural Network.
    
    Architecture:
    - Input Layer
    - Flatten (since sequence output shape is (look_back, features))
    - Dense(128), ReLU, Dropout
    - Dense(64), ReLU, Dropout
    - Dense(32), ReLU
    - Dense(1)
    """
    try:
        model = Sequential([
            Input(shape=input_shape),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info(f"ANN Model built successfully with input shape {input_shape}.")
        return model
    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise

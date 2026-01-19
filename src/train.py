import os
import pickle
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

def main():
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Loading configuration...")
    with open(os.path.join(DATA_DIR, "data.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Feature Mapping
    scorecard_features_map = {
        'school_name': 'school.name',
        'state': 'school.state',
        'control': 'school.ownership',
        'tuition_in_state': 'cost.tuition.in_state',
        'sat_avg': 'admissions.sat_scores.average.overall',
        'pell_grant_rate': 'aid.pell_grant_rate',
        'faculty_salary': 'school.faculty_salary',
        'unitid': 'id'
    }

    use_cols_scorecard = {}
    for alias, yaml_key in scorecard_features_map.items():
        if yaml_key in config['dictionary']:
            raw_col = config['dictionary'][yaml_key]['source']
            use_cols_scorecard[raw_col] = alias

    print("Loading and merging data...")
    # Load Data
    df_sc = pd.read_csv(
        os.path.join(DATA_DIR, "MERGED2023_24_PP.csv"), 
        usecols=use_cols_scorecard.keys(), 
        na_values=config['null_value']
    )
    df_sc.rename(columns=use_cols_scorecard, inplace=True)

    df_ipeds = pd.read_csv(os.path.join(DATA_DIR, "ADM2024.csv"))
    df_ipeds = df_ipeds[['UNITID', 'ADMSSN', 'ENRLT']]

    # Merge
    df_final = pd.merge(df_sc, df_ipeds, left_on='unitid', right_on='UNITID', how='inner')

    # Calculate Target
    df_final['YIELD'] = df_final['ENRLT'] / df_final['ADMSSN']

    # Clean
    df_final = df_final[df_final['ADMSSN'] > 0]
    df_final = df_final[df_final['YIELD'] <= 1.0]
    df_final = df_final.dropna(subset=['YIELD'])
    
    print(f"Data ready: {df_final.shape}")

    # Features
    numeric_features = ['tuition_in_state', 'sat_avg', 'pell_grant_rate', 'faculty_salary']
    categorical_features = ['state', 'control']

    X = df_final[numeric_features + categorical_features]
    y = df_final['YIELD']

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit Preprocessor
    print("Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to dense for TensorFlow
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    # Build Model
    def build_model(input_shape):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    print("Training model...")
    model = build_model(X_train_processed.shape[1])
    
    # Train
    model.fit(
        X_train_processed, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )

    # Evaluate
    loss, mae = model.evaluate(X_test_processed, y_test, verbose=0)
    print(f"Test MAE: {mae:.4f}")

    # Save
    print("Saving artifacts...")
    # Save preprocessor
    with open(os.path.join(MODELS_DIR, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Save Model
    # Keras models cannot be cleanly pickled standardly. We save as .keras
    model_path = os.path.join(MODELS_DIR, 'yield_model.keras')
    model.save(model_path)
    
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {os.path.join(MODELS_DIR, 'preprocessor.pkl')}")

if __name__ == "__main__":
    main()

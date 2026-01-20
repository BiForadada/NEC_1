import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_and_save(file_path: str, output_path: str):
    """
    Loads the cleaned data, applies Standard Scaling to all numerical features
    and One-Hot Encoding to categorical features, then saves the resulting 
    processed data to a new CSV file.

    Args:
        file_path (str): Path to the cleaned CSV file.
        output_path (str): Path to save the final processed CSV file.
    """
    print("--- Starting Data Preprocessing ---")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file path '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    # Split the columns of the target prediction and the rest
    TARGET_COL = 'Life expectancy'
    if TARGET_COL not in df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found.")
        return
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Columns with strings as values
    CATEGORICAL_COLS = ['Country', 'Status']
    
    # A standard scaling will be applied to all numerical columns
    ALL_NUMERICAL_COLS = [
        'Infant deaths', 'Under-five deaths', 'Measles', 
        'HIV/AIDS', 'Alcohol', 'Total expenditure',
        'Year', 'Adult Mortality', 'BMI', 'Polio', 'Diphtheria',
        'thinness 1-19 years', 'thinness 5-9 years'
    ]
    
    # Filter columns to ensure they exist
    all_numerical_cols = [col for col in ALL_NUMERICAL_COLS if col in X.columns]
    categorical_cols = [col for col in CATEGORICAL_COLS if col in X.columns]

    
    # Standard Scaling for all numerical features
    scaling_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # One-Hot Encoding for categorical features
    onehot_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_scale', scaling_pipeline, all_numerical_cols),
            ('cat', onehot_pipeline, categorical_cols)
        ],
        remainder='passthrough'
    )

    # Apply the transformations to the entire dataset
    print("Applying ColumnTransformer to training and scaling Life Expectancy...")
    
    # Fit and transform the feature columns
    X_processed = preprocessor.fit_transform(X)
    
    # Fit and transform target column 'Life expectancy'
    target_scaler = StandardScaler()
    y_processed = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Extract final feature names and clean them
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split('__')[-1] for name in feature_names]
    
    # Create the final processed dataFrame
    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    df_processed['TARGET_Life_expectancy_SCALED'] = y_processed

    # Print results
    print(f"Features: {len(feature_names)} columns created.")
    print(f"Final processed shape: {df_processed.shape}")
    df_processed.to_csv(output_path, index=False)
    print(f"--- Preprocessing Complete. Data saved to: {output_path} ---")


if __name__ == "__main__":
    CLEANED_CSV_PATH = 'life_expectancy_data_cleaned.csv'
    PROCESSED_CSV_PATH = 'life_expectancy_data_processed.csv'

    preprocess_and_save(CLEANED_CSV_PATH, PROCESSED_CSV_PATH)
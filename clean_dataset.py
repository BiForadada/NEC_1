import pandas as pd

def clean_life_expectancy_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Life Expectancy dataset from the given path, applies a specific
    set of filtering, extrapolation, and column deletion rules based on prior
    missing value analysis, and returns the cleaned DataFrame.

    Args:
        file_path (str): The path to the original CSV file.

    Returns:
        pd.DataFrame: The DataFrame after applying all cleaning rules.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file path '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return pd.DataFrame()

    # Drop the following columns because of the number of values missing in the dataset
    print("-> Dropping columns (GDP, Population, Income composition of resources, Schooling, Hepatitis B)...")
    columns_to_drop = [
        'GDP', 'Population', 'Income composition of resources',
        'Schooling', 'Hepatitis B', 'Percentage expenditure'
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)


    # Delete the rows the miss the values for 'Life expectancy ' or 'Adult Mortality'
    print("-> Deleting rows where 'Life expectancy ' or 'Adult Mortality' is missing...")
    df.dropna(subset=['Life expectancy', 'Adult Mortality'], inplace=True)


    # Extrapolate the values of 2014 to 2015 to the countries that do not have that value in place yet
    # This is done for the countries of 'Alcohol' and 'Total expenditure'
    year_col = 'Year'
    alcohol_col = 'Alcohol'
    total_exp_col = 'Total expenditure'

    # Extrapolation of the values of alcohol
    if alcohol_col in df.columns and year_col in df.columns:
        print(f"-> Extrapolating values of 'Alcohol' (2015 missing -> 2014 value)...")
        
        missing_alcohol_countries = df[df[alcohol_col].isnull()]['Country'].unique()
        imputed_count = 0
        for country in missing_alcohol_countries:
            # Identify countries that only miss alcohol data in 2015
            is_2015_missing = df[(df['Country'] == country) & (df[year_col] == 2015) & (df[alcohol_col].isnull())]
            has_2014_value = df[(df['Country'] == country) & (df[year_col] == 2014)][alcohol_col]

            if not is_2015_missing.empty and not has_2014_value.empty:
                # Get the 2014 value
                val_2014 = has_2014_value.iloc[0]
                # Assign the 2014 value to the 2015 missing slot
                df.loc[is_2015_missing.index, alcohol_col] = val_2014
                imputed_count += 1

        print(f"   -> Extrapolated {imputed_count} rows using 2014 data.")

        # Delete the remaing rows with missing alcohol data
        print("   -> Deleting remaining rows with missing 'Alcohol' data...")
        df.dropna(subset=[alcohol_col], inplace=True)
    else:
        print("Skipping Alcohol cleaning: 'Alcohol' or 'Year' column not found.")

    # Extrapolation of the values of total expenditure
    if total_exp_col in df.columns and year_col in df.columns:
        print(f"-> Extrapolating values of 'Total expenditure' (2015 missing -> 2014 value)...")

        missing_total_exp_countries = df[df[total_exp_col].isnull()]['Country'].unique()
        imputed_count = 0
        for country in missing_total_exp_countries:
            # Identify countries that only miss total expenditure data in 2015
            is_2015_missing = df[(df['Country'] == country) & (df[year_col] == 2015) & (df[total_exp_col].isnull())]
            has_2014_value = df[(df['Country'] == country) & (df[year_col] == 2014)][total_exp_col]

            if not is_2015_missing.empty and not has_2014_value.empty:
                val_2014 = has_2014_value.iloc[0]
                df.loc[is_2015_missing.index, total_exp_col] = val_2014
                imputed_count += 1

        print(f"   -> Extrapolated {imputed_count} rows using 2014 data.")

        # Delete any remaining NaNs
        print("   -> Deleting remaining rows with missing 'Total expenditure' data...")
        df.dropna(subset=[total_exp_col], inplace=True)
    else:
        print("Skipping Total expenditure cleaning: 'Total expenditure' or 'Year' column not found.")


    # Delete the rows that miss BMI as there are not a representative amount of the dataset (Monaco and San Marino)
    print("-> Deleting rows with missing 'BMI '...")
    df.dropna(subset=['BMI'], inplace=True)

    # Delete the rows with missing Polio & Diphtheria values (early Montenegro and Timor Leste)
    print("-> Deleting rows with missing 'Polio' or 'Diphtheria '...")
    df.dropna(subset=['Polio', 'Diphtheria'], inplace=True)
    
    # Delete rows with missing thinness 1-19 years and thinness 5-9 years missing values
    print("-> Deleting rows with missing 'thinness' values...")
    df.dropna(subset=['thinness 1-19 years', 'thinness 5-9 years'], inplace=True)


    print(f"\n--- Cleaning Summary ---")
    print(f"Original rows (before cleaning): {pd.read_csv(file_path).shape[0]}")
    print(f"Final rows remaining: {df.shape[0]}")
    print(f"Final columns remaining: {df.shape[1]}")
    return df


# --- Execution Example ---
if __name__ == "__main__":
    INPUT_CSV_PATH = 'life_expectancy_original.csv'
    OUTPUT_CSV_PATH = 'life_expectancy_data_cleaned.csv'

    # Run the cleaning function
    cleaned_df = clean_life_expectancy_data(INPUT_CSV_PATH)

    if not cleaned_df.empty:
        # Save the cleaned DataFrame to a new CSV file
        cleaned_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSuccessfully saved the cleaned data to: {OUTPUT_CSV_PATH}")
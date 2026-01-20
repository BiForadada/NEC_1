# Data Cleaning and Preprocessing Report: Life Expectancy Dataset

## Initial Missing Value Assessment
The following table provides the complete count of missing values per column as identified in the original dataset:

| Column                          | Missing Values | Column                          | Missing Values |
| :---                            | :---           | :---                            | :---           |
| **Country** | 0    | **Under-five deaths** | 0    |
| **Year** | 0    | **Polio** | 19   |
| **Status** | 0    | **Total expenditure** | 226  |
| **Life expectancy** | 10   | **Diphtheria** | 19   |
| **Adult Mortality** | 10   | **HIV/AIDS** | 0    |
| **Infant deaths** | 0    | **GDP** | 448  |
| **Alcohol** | 194  | **Population** | 652  |
| **Percentage expenditure** | 0    | **thinness 1-19 years** | 34   |
| **Hepatitis B** | 553  | **thinness 5-9 years** | 34   |
| **Measles** | 0    | **Income composition** | 167  |
| **BMI** | 34   | **Schooling** | 163  |

---

## Column Removal Strategy
Specific columns were dropped entirely based on high missingness or data reliability concerns:

* **GDP & Population:** Both developed and developing countries lacked this data; removing the affected rows would have severely reduced the dataset size.
* **Hepatitis B:** This column represented 1/6 of the total dataset, making row deletion non-viable.
* **Percentage Expenditure:** Dropped because 557 rows contained 0.0 values, including for significant countries throughout multiple years.
* **Income Composition & Schooling:** These features were removed due to widespread missing instances across various country types.

---

## Targeted Row Deletion
Where data gaps were minimal or concentrated in specific unreliable records, the rows were removed.

### Critical Target Data
* **Life Expectancy & Adult Mortality:** Deleted the 10 rows missing these values because they are essential for prediction and Life Expectancy is the target variable.

### Data Quality & Consistency
* **BMI & Thinness:** Rows for South Sudan and Sudan were deleted as they lacked data across several columns.
* **Polio & Diphtheria:** Records for South Sudan, early Montenegro, and Timor Leste were removed.
* **Incomplete Records:** Data for the Democratic People's Republic of Korea and Somalia were removed due to inaccuracies and extensive missing features.

---

## Temporal Extrapolation
To preserve 2015 data, some data was extrapolated from 2014:

* **Alcohol & Total Expenditure:** If 2015 data was missing but 2014 data existed for a country, the 2014 value was assigned to 2015.
* **Cleanup:** After the extrapolation, any remaining rows with missing values were deleted.

---

## Feature Transformation Summary


### Numerical Feature Scaling
All numerical features are standardized using **StandardScaler**. This ensures that every feature has a mean of 0 and a standard deviation of 1.
* **Scaled Features**: `Infant deaths`, `Under-five deaths`, `Measles`, `HIV/AIDS`, `Alcohol`, `Total expenditure`, `Year`, `Adult Mortality`, `BMI`, `Polio`, `Diphtheria`, `thinness 1-19 years`, and `thinness 5-9 years`.

### Categorical Feature Encoding
* **One-Hot Encoding**: Both the **Country** and **Status** columns are transformed using One-Hot Encoding.

### Target Variable Scaling
* **Target Scaling**: The target variable, **Life expectancy**, is also scaled using a **StandardScaler**.
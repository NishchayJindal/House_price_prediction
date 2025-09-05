# Bangalore House Price Prediction

This is a data science project that aims to predict house prices in Bangalore, India. The project uses machine learning regression techniques to create a model that estimates property prices based on various features like location, size (BHK), total square feet, and number of bathrooms.

The entire workflow, from data cleaning and feature engineering to model building and evaluation, is documented in the Jupyter Notebook.

## 🚀 Project Overview

The project follows a standard data science pipeline:

1.  **Data Loading & Exploration:** Loading the dataset and understanding its structure.
2.  **Data Cleaning:** Handling missing values, and dropping features that are not relevant for the model.
3.  **Feature Engineering:** Creating new features from existing ones to improve model performance (e.g., calculating price per square foot).
4.  **Dimensionality Reduction:** Reducing the number of unique location categories to make the model more manageable.
5.  **Outlier Removal:** Applying business logic and statistical methods to remove anomalous data points.
6.  **Model Building:** Training and testing several regression models.
7.  **Model Evaluation:** Using K-Fold cross-validation and GridSearchCV to find the best-performing model and its optimal parameters.
8.  **Model Export:** Saving the final trained model and a column information file for use in a prediction application.

## 📊 Dataset

The dataset used for this project is sourced from Kaggle and contains information about house listings in Bangalore.

-   **Source:** [Bengaluru House Price Data on Kaggle](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data)

## 🛠️ Technologies Used

-   **Python**
-   **Pandas** for data manipulation and analysis.
-   **NumPy** for numerical operations.
-   **Scikit-learn** for machine learning models and evaluation.
-   **Matplotlib** for data visualization.
-   **Jupyter Notebook** for interactive development.

## 📋 Project Workflow

### 1. Data Cleaning and Preprocessing
-   Dropped columns that were not essential for price prediction (`area_type`, `society`, `balcony`, `availability`).
-   Handled null values by dropping rows with missing data.

### 2. Feature Engineering
-   Created a **`bhk`** column by extracting the numerical value from the `size` column (e.g., '2 BHK' becomes 2).
-   Cleaned the **`total_sqft`** column by converting ranges (e.g., "2100-2850") into a single average value and removing non-numeric entries.
-   Created a **`price_per_sqft`** feature, which was crucial for outlier detection.

### 3. Dimensionality Reduction
-   The `location` column had over 1200 unique categories. Locations with fewer than 10 data points were grouped into an "other" category to reduce the number of dimensions.

### 4. Outlier Removal
Several techniques were used to remove outliers and improve data quality:
-   **Business Logic 1:** Removed properties where the square feet per bedroom was less than 300, as this is typically unrealistic.
-   **Statistical Outliers:** Used the mean and one standard deviation to filter out extreme values of `price_per_sqft` for each location.
-   **Business Logic 2:** Removed properties where the number of bathrooms was greater than the number of bedrooms plus 2.
-   **Price Anomaly:** Removed instances where a smaller BHK apartment in the same location was priced higher than a larger one.

### 5. Model Training and Evaluation
-   **One-Hot Encoding:** Converted the categorical `location` column into numerical data using `pd.get_dummies()`.
-   **Model Selection:** Tested three different regression algorithms:
    1.  Linear Regression
    2.  Lasso Regression
    3.  Decision Tree Regressor
-   **Evaluation:** Used **K-Fold cross-validation** (`ShuffleSplit`) and **GridSearchCV** to find the best model. **Linear Regression** was found to be the best-performing model with a cross-validation score of approximately **85%**.

## 📦 Project Files

-   `banglore_home_prices_final.ipynb`: The main Jupyter notebook containing all the code for data analysis, feature engineering, and model training.
-   `banglore_home_prices_model.pickle`: The exported, trained linear regression model.
-   `columns.json`: A JSON file containing the list of columns (including the one-hot encoded locations) needed for making predictions.

## ⚙️ How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install the required libraries.** It is recommended to create a virtual environment first.
    ```bash
    pip install numpy pandas matplotlib scikit-learn jupyter
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook banglore_home_prices_final.ipynb
    ```
4.  After running the notebook, the `banglore_home_prices_model.pickle` and `columns.json` files will be generated. These can be used to build a web application for real-time predictions.

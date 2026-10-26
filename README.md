# House Prices Prediction

End-to-end Machine Learning regression project developed using the Kaggle House Prices dataset.

The objective of this project is to predict residential property sale prices while building a complete and reproducible Machine Learning workflow, from exploratory data analysis to model optimization and final Kaggle submission.

## Project Overview

This project covers the following stages:

1. Exploratory Data Analysis
2. Missing-value analysis
3. Feature engineering
4. Data preprocessing
5. Regression model comparison
6. Cross-validation
7. Overfitting analysis
8. Hyperparameter optimization
9. Final model training
10. Kaggle submission generation

The target variable, `SalePrice`, was transformed using:

```python
np.log1p(SalePrice)
```

This transformation reduces the positive skewness of the target distribution and allows the models to focus more on proportional prediction errors.

The main evaluation metric is Root Mean Squared Error calculated over the logarithmic target.

## Repository Structure

```text
house-prices-machine-learning/
│
├── data/
│   └── raw/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_model_optimization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── preprocessing.py
│
├── models/
│   ├── model_comparison.csv
│   ├── optimization_comparison.csv
│   └── best_model.joblib
│
├── submissions/
│   └── submission_optimized.csv
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Exploratory Data Analysis

The exploratory analysis focused on:

- Understanding the dimensions and structure of the dataset
- Analyzing the distribution of `SalePrice`
- Detecting missing values
- Separating numerical and categorical variables
- Studying correlations between numerical features and `SalePrice`
- Analyzing categorical features
- Identifying potential outliers
- Investigating relationships between property characteristics and sale prices

The original `SalePrice` distribution showed a positive skew with a long right tail. A logarithmic transformation was therefore applied to obtain a more symmetric target distribution.

Some of the numerical variables most strongly correlated with `SalePrice` were:

- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `GarageArea`
- `TotalBsmtSF`
- `1stFlrSF`
- `FullBath`
- `YearBuilt`
- `YearRemodAdd`

Correlation was used as an exploratory tool rather than as the only criterion for selecting features.

## Missing-Value Strategy

Missing values were treated according to their meaning.

For categorical variables where a missing value represented the absence of a property feature, the missing value was replaced with `"None"`.

Examples include:

- No garage
- No basement
- No fireplace
- No pool
- No fence
- No alley access

For numerical variables where a missing value represented the absence of a feature, the value was replaced with `0`.

Remaining unknown numerical values were imputed using the median inside the Scikit-learn pipelines. This ensured that imputation statistics were learned only from the training data in each cross-validation fold and prevented data leakage.

Remaining categorical missing values were imputed using a separate `"Missing"` category.

## Feature Engineering

Several new features were created from the original data:

- `TotalSF`: total basement, first-floor, and second-floor area
- `HouseAge`: age of the property when it was sold
- `RemodAge`: years since the latest remodeling
- `TotalBathrooms`: combined number of full and half bathrooms
- `HasGarage`: indicates whether the property has a garage
- `HasBasement`: indicates whether the property has a basement
- `HasFireplace`: indicates whether the property has a fireplace

Reusable feature-engineering code is stored in:

```text
src/features.py
```

The dataset-specific missing-value treatment is stored in:

```text
src/preprocessing.py
```

## Preprocessing Pipelines

The project uses Scikit-learn pipelines to create a reproducible preprocessing and modeling workflow.

### Numerical Features

Numerical variables are processed using:

- Median imputation
- Standard scaling for linear models

### Categorical Features

Categorical variables are processed using:

- Constant-value imputation
- One-Hot Encoding
- Ignoring previously unseen categories during prediction

Tree-based models do not use StandardScaler because their decision rules do not depend on the scale of numerical variables.

## Models Evaluated

The following regression models were compared:

- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor

All models were evaluated using the same five-fold cross-validation strategy.

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The primary model-selection metric was mean cross-validation log-RMSE.

## Initial Model Comparison

| Model | Train RMSE | CV RMSE | CV RMSE Standard Deviation | CV R² | Overfitting Gap |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.05972 | **0.12729** | 0.01557 | 0.89335 | 0.06757 |
| ElasticNet | 0.10517 | 0.14026 | 0.03202 | 0.86729 | 0.03510 |
| Lasso | 0.11260 | 0.14027 | 0.03343 | 0.86699 | 0.02767 |
| Ridge | 0.10708 | 0.14297 | 0.02917 | 0.86312 | 0.03589 |
| Random Forest | 0.05349 | 0.14361 | 0.01484 | 0.86444 | 0.09013 |

Gradient Boosting achieved the best initial cross-validation performance.

Random Forest achieved a very low training error but had the largest train-validation gap, indicating stronger overfitting.

Gradient Boosting also showed some overfitting, but it obtained the lowest cross-validation RMSE and was selected as the main optimization candidate.

ElasticNet was retained as the strongest regularized linear model.

## Hyperparameter Optimization

### ElasticNet

ElasticNet was optimized using `GridSearchCV`.

The search evaluated different combinations of:

- `alpha`
- `l1_ratio`

Best parameters:

```python
{
    "alpha": 0.001,
    "l1_ratio": 0.7
}
```

Optimized ElasticNet result:

```text
CV RMSE: 0.13987
```

The optimization produced a small improvement over the original ElasticNet model.

### Gradient Boosting

Gradient Boosting was optimized using `RandomizedSearchCV`.

The search included:

- Number of estimators
- Learning rate
- Maximum tree depth
- Minimum samples required to split a node
- Minimum samples per leaf
- Subsampling ratio
- Number of features considered during tree construction

Best parameters:

```python
{
    "n_estimators": 700,
    "learning_rate": 0.025,
    "max_depth": 3,
    "min_samples_split": 10,
    "min_samples_leaf": 12,
    "subsample": 0.9,
    "max_features": "sqrt"
}
```

## Final Results

| Model | Train RMSE | CV RMSE | Overfitting Gap |
|---|---:|---:|---:|
| **Optimized Gradient Boosting** | 0.08280 | **0.12191** | **0.03911** |
| Original Gradient Boosting | 0.05972 | 0.12729 | 0.06757 |
| Optimized ElasticNet | 0.10838 | 0.13987 | 0.03148 |
| Original ElasticNet | 0.10517 | 0.14026 | 0.03510 |

The optimized Gradient Boosting model achieved:

- Mean training RMSE: **0.08280**
- Mean cross-validation RMSE: **0.12191**
- Cross-validation RMSE standard deviation: **0.01490**
- Mean cross-validation MAE: **0.08325**
- Mean cross-validation R²: **0.90222**
- Train-validation RMSE gap: **0.03911**

The optimization improved the cross-validation RMSE from `0.12729` to `0.12191`.

It also reduced the train-validation gap from `0.06757` to `0.03911`, indicating better generalization and less overfitting.

Compared with the initial Ridge baseline, the optimized Gradient Boosting model reduced the cross-validation RMSE by approximately 14.7%.

## Final Model

The final selected model was:

```text
GradientBoostingRegressor
```

The final pipeline contains:

1. Dataset-specific missing-value treatment
2. Numerical median imputation
3. Categorical missing-value imputation
4. One-Hot Encoding
5. Feature engineering
6. Optimized Gradient Boosting model

The final model was trained using the complete training dataset.

Predictions for the Kaggle test dataset were transformed back to the original price scale using:

```python
np.expm1(predictions_log)
```

The final predictions were saved to:

```text
submissions/submission_optimized.csv
```

## Kaggle Result

The final optimized Gradient Boosting model achieved:

- **Local 5-Fold CV RMSE:** 0.12191
- **Kaggle Score:** 0.13582

The Kaggle score is slightly higher than the local cross-validation
estimate, indicating a moderate generalization gap between the local
validation folds and the unseen Kaggle test data.

This result highlights the importance of evaluating models on unseen
data rather than relying exclusively on training or local validation
performance.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Joblib
- Git
- GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/house-prices-machine-learning.git
```

Enter the project directory:

```bash
cd house-prices-machine-learning
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in the following order:

```text
01_eda.ipynb
02_preprocessing.ipynb
03_modeling.ipynb
04_model_optimization.ipynb
```

## Key Learnings

This project demonstrates practical knowledge of:

- Exploratory data analysis
- Regression problems
- Missing-value interpretation
- Numerical and categorical preprocessing
- Feature engineering
- Logarithmic target transformations
- Scikit-learn pipelines
- One-Hot Encoding
- Feature scaling
- Cross-validation
- Model comparison
- Hyperparameter optimization
- Overfitting detection
- Model regularization
- Model persistence
- Kaggle submission generation
- Reproducible Machine Learning workflows

## Future Improvements

Potential next steps include:

- Evaluating XGBoost, LightGBM, or CatBoost
- Studying permutation feature importance
- Performing more detailed outlier analysis
- Testing additional engineered features
- Comparing local cross-validation performance with the Kaggle score
- Creating an ensemble of the strongest models
- Adding automated tests for preprocessing and feature-engineering functions

## Author

**Jose Rodrigo Carrillo Soult**

Financial Operations professional developing skills in Data Analytics, Machine Learning, Python, SQL, Power BI, and data-driven process improvement.

- GitHub: `https://github.com/Jose-Rodrigo-Carrillo`
- LinkedIn: `www.linkedin.com/in/rodrigocarrillosoult/`

## Acknowledgements

The dataset is provided through the Kaggle House Prices competition:

**House Prices: Advanced Regression Techniques**

The project was developed for educational and portfolio purposes.
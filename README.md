# California Housing Valuation Engine

A machine learning pipeline that predicts median house values across California districts using demographic and spatial data. Goes beyond baseline OLS by implementing L2-Regularized Ridge Regression, spatial feature engineering, and targeted data filtration.

## Highlights

### Data Preprocessing & Outlier Removal

The raw scikit-learn dataset caps all properties valued over $500,000 at `5.0`. These records were filtered out, leaving a clean training set of 19,648 records.

### Feature Engineering

* **Density ratios:** `Rooms_per_Household` and `Bedrooms_per_Room` isolate dwelling quality from raw geographic size.
* **Log scaling:** `log1p` transformations on right-skewed variables like Median Income stabilize variance and improve convergence.

### Regularized Modeling

Ridge Regression (L2 penalty) addresses multicollinearity between spatial coordinates (Latitude vs. Longitude), preventing disproportionate weights on correlated geographic features.

### Visual Diagnostics

Hexbin density maps and coefficient charts replace scatter plots to handle overplotting at scale and surface model behavior clearly.

## Technologies Used

* **Language:** Python 3
* **Libraries:** Scikit-Learn, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Modeling:** Ridge Regression (L2 Penalty)

## Technologies

| Role          | Stack                               |
| ------------- | ----------------------------------- |
| Language      | Python 3                            |
| Data          | Pandas, NumPy                       |
| Modeling      | Scikit-Learn, Ridge Regression (L2) |
| Visualization | Matplotlib, Seaborn                 |

## Files

| File                   | Description                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `aarav_lr_house.ipynb` | End-to-end pipeline: EDA, feature engineering, model training, and visual diagnostics |
| `ridge_model.pkl`      | Pre-trained Ridge Regression model (serialized)                                       |
| `feature_scaler.pkl`   | StandardScaler state for normalizing new input data                                   |
| `Report.pdf`           | LaTeX project report covering mathematical architecture and diagnostics               |

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/VEDANT1727100/Linear_Regression.git

# 2. Navigate to the project directory
cd Linear_Regression

# 3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 4. Launch the notebook
jupyter notebook aarav_lr_house.ipynb
```

## Project Workflow

1. Load the California Housing dataset from Scikit-Learn.
2. Remove capped target values (`MedHouseVal >= 5.0`) to reduce target distortion.
3. Engineer density and quality-based housing features.
4. Apply logarithmic transformations to skewed variables.
5. Standardize numerical features using `StandardScaler`.
6. Train a Ridge Regression model with L2 regularization.
7. Evaluate model performance using RMSE and R² metrics.
8. Visualize feature importance and prediction behavior using diagnostic plots.

## Model Outputs

The project generates:

* A trained Ridge Regression model (`ridge_model.pkl`)
* A fitted feature scaler (`feature_scaler.pkl`)
* Visual diagnostics and exploratory analysis plots
* A detailed PDF report documenting methodology and results

## Author

**Vedant Agrawal**

Machine Learning & Data Science Project

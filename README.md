# California Housing Valuation Engine

A machine learning pipeline that predicts median house values across California districts using demographic and spatial data. Goes beyond baseline OLS by implementing L2-Regularized Ridge Regression, spatial feature engineering, and targeted data filtration.

## Highlights

**Data Preprocessing & Outlier Removal**
The raw scikit-learn dataset caps all properties valued over $500,000 at `5.0`. These records were filtered out, leaving a clean training set of 19,648 records.

**Feature Engineering**
- *Density ratios:* `Rooms_per_Household` and `Bedrooms_per_Room` isolate dwelling quality from raw geographic size.
- *Log scaling:* `log1p` transformations on right-skewed variables like Median Income stabilize variance and improve convergence.

**Regularized Modeling**
Ridge Regression (L2 penalty) addresses multicollinearity between spatial coordinates (Latitude vs. Longitude), preventing disproportionate weights on correlated geographic features.

**Visual Diagnostics**
Hexbin density maps and coefficient charts replace scatter plots to handle overplotting at scale and surface model behavior clearly.

## Files

| File | Description |
|------|-------------|
| `aarav_lr_house.ipynb` | End-to-end pipeline: EDA, model training, and visual diagnostics |
| `ridge_model.pkl` | Pre-trained Ridge Regression model (serialized) |
| `feature_scaler.pkl` | StandardScaler state for normalizing new input data |
| `Report.pdf` | LaTeX project report covering math architecture and diagnostics |

## Setup

```bash
# 1. Clone the repo
git clone <your-repository-url>
cd <your-repository-directory>

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Launch the notebook
jupyter notebook aarav_lr_house.ipynb
```

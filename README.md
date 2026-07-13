# CarDekho Used Car Price Prediction

An end-to-end machine learning project that predicts the resale (selling) price of a used
car based on attributes such as its showroom price, age, mileage, fuel type, and ownership
history. Built as an applied follow-up to the *Cardeko Car Price Prediction Model* walkthrough,
this project implements the full pipeline — data cleaning, feature engineering, exploratory
analysis, model training/comparison, hyperparameter tuning, and evaluation — on a real dataset.

## 1. Introduction

Pricing a used car fairly is hard: value depends on a mix of depreciation, mileage, condition,
and market demand. This project builds a regression model that learns these relationships from
historical listings, so a seller/dealer/buyer can get an instant, data-driven price estimate
instead of relying on guesswork.

## 2. Dataset

- **Source:** [CarDekho used car listings dataset](https://github.com/SahilTKgpian/Cardekho_dataset)
  (originally published on Kaggle as *"Vehicle dataset from CarDekho"*), a widely used public
  dataset of 301 real used-car listings from cardekho.com.
- **File:** `data/car_data.csv`
- **Raw features (9 columns):**

| Column | Description |
|---|---|
| `Car_Name` | Model name of the car |
| `Year` | Year the car was purchased new |
| `Selling_Price` | **Target** — price the car is being resold for (Lakhs INR) |
| `Present_Price` | Current showroom (ex-price) of the same model, if bought new today |
| `Kms_Driven` | Total kilometers driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Seller_Type` | Dealer / Individual |
| `Transmission` | Manual / Automatic |
| `Owner` | Number of previous owners |

### Preprocessing applied

1. **Duplicate removal** — 2 exact duplicate rows dropped.
2. **Feature engineering** — `Year` converted to `Car_Age` (years since manufacture, relative
   to the newest car in the dataset), which is more directly predictive than a raw calendar year.
3. **High-cardinality column dropped** — `Car_Name` has 98 unique values across only 301 rows;
   one-hot/label encoding it would cause severe sparsity and overfitting for a dataset this
   size, so it was dropped in favor of the more general `Present_Price` (which already captures
   most of the "which model/brand" signal via showroom price).
4. **Outlier capping** — `Present_Price` and `Kms_Driven` each have a long right tail (a handful
   of premium or very high-mileage cars). Instead of deleting these rows, values above the 99th
   percentile were capped, preserving sample size while limiting the influence of extreme points.
5. **Categorical encoding** — `Fuel_Type`, `Seller_Type`, and `Transmission` label-encoded for
   the model; encoders are saved so the same mapping is reused at inference time.
6. **Train/test split** — 80/20 split, `random_state=42` for reproducibility.

## 3. Models Implemented

Five regressors were trained and compared on identical train/test splits:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best baseline model was then refined with `GridSearchCV` (5-fold CV) over its key
hyperparameters. Evaluation used **RMSE**, **MAE**, and **R²** on the held-out test set.

## 4. Results

| Model | RMSE (Lakhs) | MAE (Lakhs) | R² |
|---|---|---|---|
| Linear Regression | 2.59 | 1.56 | 0.741 |
| Ridge | 2.57 | 1.55 | 0.743 |
| **Lasso (best)** | **2.53** | **1.53** | **0.752** |
| Random Forest | 4.31 | 1.70 | 0.279 |
| Gradient Boosting | 3.55 | 1.41 | 0.512 |

*(Full numbers are written to `outputs/metrics.json` on every run.)*

**Lasso Regression** was selected as the final model. `GridSearchCV` confirmed the untuned
regularization strength was already near-optimal for this dataset.

### Key Findings

- **`Present_Price` is by far the strongest predictor** of resale value (Pearson correlation
  ≈ 0.88 with `Selling_Price`) — see `outputs/figures/correlation_heatmap.png` — followed by
  `Car_Age` and `Kms_Driven`.
- **Automatic-transmission cars resell for noticeably more** than manual cars at a similar age,
  visible in `outputs/figures/categorical_boxplots.png`.
- **Linear models outperformed tree ensembles here**, which was initially counter-intuitive.
  Investigating a large individual error revealed why: rare combinations, like a car with a
  high showroom price but very high mileage, an old age, and several previous owners, are
  under-represented in only ~240 training rows. Tree-based models split on such combinations
  greedily and generalize poorly to the few examples of that combination, while linear models'
  additive structure (a global weighted sum of features) extrapolates much more gracefully with
  limited data. This is a good illustration of why simpler models often win on small,
  well-structured tabular datasets.
- The dataset is right-skewed (`outputs/figures/target_distribution.png`): most cars sell for
  under 10 lakhs, with a small number of premium vehicles pulling the tail out to 33–35 lakhs.

### Visual Outputs (`outputs/figures/`)

- `target_distribution.png` – distribution of selling prices
- `correlation_heatmap.png` – numeric feature correlations
- `price_vs_age.png`, `price_vs_present_price.png` – key scatter relationships
- `categorical_boxplots.png` – price by fuel type / seller type / transmission
- `model_comparison.png` – R² and RMSE across all five models
- `predicted_vs_actual.png` – final model's predictions vs. ground truth
- `feature_importance.png` – produced automatically if the final model supports it
  (tree-based models only; not generated for the winning Lasso model since it uses
  coefficients instead — see the printed coefficients in the training log)

## 5. Model Improvements & Future Enhancements

- **More data** would likely let tree ensembles (Random Forest / Gradient Boosting) catch up to
  or beat the linear model, since their main weakness here was sparse coverage of rare feature
  combinations, not a fundamentally worse fit.
- **Re-introduce `Car_Name`** using target encoding or grouping into brand/segment (e.g.
  luxury vs. economy) instead of dropping it outright, to recover brand-level pricing signal.
- **Log-transform the target** (`Selling_Price`) to reduce the influence of the long right tail
  and better satisfy linear regression's assumptions.
- **Add external features** such as engine size, seating capacity, or regional demand, which
  are known to matter but aren't present in this particular dataset.
- **Deploy as a small web app** (e.g. Flask/FastAPI + the saved `car_price_model.pkl`) so users
  can get a live price estimate through a form instead of running a script.

## 6. Project Structure

```
car_price_project/
├── data/
│   └── car_data.csv                # raw dataset
├── src/
│   ├── car_price_prediction.py     # full pipeline: clean -> EDA -> train -> evaluate -> save
│   └── predict.py                  # load saved model and predict a price for one car
├── outputs/
│   ├── figures/                    # all EDA & evaluation plots (PNG)
│   ├── model/                      # saved model, encoders, feature list (joblib .pkl)
│   └── metrics.json                # RMSE / MAE / R² for every model tried
├── requirements.txt
└── README.md
```

## 7. Setup & Usage

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd car_price_project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (cleans data, runs EDA, trains & evaluates all models,
#    saves the best one)
python src/car_price_prediction.py

# 5. Predict a price for a new car using the saved model
python src/predict.py
```

To predict on your own car, edit the `sample_car` dictionary at the bottom of
`src/predict.py`, or import `predict_price()` from that module into your own script:

```python
from src.predict import predict_price

price = predict_price({
    "Present_Price": 9.5,
    "Kms_Driven": 35000,
    "Fuel_Type": "Petrol",
    "Seller_Type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0,
    "Car_Age": 5,
})
print(price)  # predicted selling price, in Lakhs INR
```

## 8. License / Attribution

Dataset originally sourced from CarDekho.com listings, redistributed for educational use via
the public GitHub mirror linked above. Code in this repository is provided for educational
purposes as part of a car price prediction learning project.

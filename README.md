# Predicting Marketing Campaign Response with XGBoost

Which customers will accept the next offer? Building an XGBoost classifier with tidymodels to predict marketing campaign response from customer demographics and purchase behaviour.

## Problem

Marketing teams waste budget targeting customers who won't convert. The goal: build a model that predicts whether a customer will accept a campaign offer (`Response = 1`) based on their profile — age, income, purchase history, channel preferences, and past campaign behaviour. The challenge: only 15% of customers accepted, so the model needs to handle class imbalance.

## Approach

1. **Explore** — visualise missing data (`naniar`), check correlations, understand distributions
2. **Engineer features** — parse dates, calculate tenure and age, clean Income from formatted strings to numeric, count children
3. **Split** — 80/20 with stratification to preserve the 15% positive class
4. **Train** — XGBoost via tidymodels with default parameters
5. **Evaluate** — confusion matrix, accuracy, precision, recall, F1, AUC, log loss

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.889 |
| Precision | 0.917 |
| Recall | 0.955 |
| F1 Score | 0.936 |
| AUC | 0.920 |
| Log Loss | 0.245 |

**Confusion Matrix:**

|  | Actual 0 | Actual 1 |
|--|---------|---------|
| Predicted 0 | 365 (TN) | 33 (FN) |
| Predicted 1 | 17 (FP) | 34 (TP) |

The model catches 95.5% of actual responders (high recall) with 91.7% precision — meaning when it flags a customer as likely to convert, it's right 9 times out of 10.

## Key Findings

- **Class imbalance** — 85% non-responders vs 15% responders. The model handles this well without resampling, thanks to XGBoost's built-in handling
- **Income was formatted as strings** (`"$84,835.00 "`) — required cleaning before modelling
- **Feature engineering mattered** — tenure, age, and child count all contributed to model performance
- **Past campaign acceptance** is a strong predictor — customers who accepted previous offers are more likely to accept again

## Setup

```bash
git clone https://github.com/wsamuelw/predict-marketing-response-with-xgboost.git
cd predict-marketing-response-with-xgboost
```

```r
install.packages(c("tidymodels", "rpart.plot", "vip", "naniar", "stringr", "lubridate", "corrplot"))
source("r code.R")
```

## Data

**Marketing Analytics** — [Kaggle](https://www.kaggle.com/jackdaoud/marketing-data/version/1). 2,240 customers with 28 features.

| Feature Group | Examples |
|--------------|---------|
| Demographics | Year of birth, education, marital status, income, country |
| Household | Kids at home, teens at home |
| Spending | Amount spent on wines, fruits, meat, fish, sweets, gold |
| Purchases | Web, catalog, store, deal purchases, web visits |
| Campaigns | AcceptedCmp1–5, Response (target), Complain |
| Engagement | Days since last purchase (recency), customer tenure |

## Feature Engineering

| Feature | How |
|---------|-----|
| `age` | Current year − Year_Birth |
| `Tenure` | Days since customer registered |
| `num_children` | Kidhome + Teenhome |
| `Income` | Strip `$`, `,`, `.00`; convert to numeric; replace NA with 0 |
| `Response` | Convert to factor for tidymodels |

## Tech Stack

- **tidymodels** — unified modelling framework (split, fit, evaluate)
- **xgboost** — gradient boosting engine
- **vip** — variable importance plots
- **naniar** — missing data visualisation
- **corrplot** — correlation matrix
- **lubridate / stringr** — date and string manipulation

## References

- [Marketing Data on Kaggle](https://www.kaggle.com/jackdaoud/marketing-data/version/1)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [tidymodels](https://www.tidymodels.org/)

## License

MIT

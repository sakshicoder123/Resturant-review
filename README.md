# 🍽️ Restaurant Rating Prediction and Customer Preference Analysis

## 📋 Project Overview

A comprehensive machine learning project that predicts restaurant aggregate ratings and analyzes customer cuisine preferences using a real-world restaurant dataset containing ~9,551 records across multiple countries.

**Built for**: College Project | GitHub Portfolio | Resume Project | Machine Learning Showcase

---

## 🎯 Problem Statement

Restaurants receive ratings from customers, but what factors actually drive these ratings? This project aims to:

1. **Predict** a restaurant's aggregate rating based on its characteristics (cost, location, services, cuisine type, etc.)
2. **Analyze** customer preferences by studying the relationship between cuisine types, votes, and ratings
3. **Compare** multiple ML algorithms to find the best-performing model
4. **Identify** which restaurant features contribute most to higher ratings

---

## 📊 Dataset Description

| Property | Value |
|----------|-------|
| **Source** | Zomato Restaurant Dataset |
| **Total Records** | ~9,551 restaurants |
| **Total Features** | 21 columns |
| **Countries** | Multiple (India, Philippines, UAE, etc.) |
| **Target Variable** | `Aggregate rating` (0–5 scale) |

### Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| Restaurant ID | int | Unique restaurant identifier |
| Restaurant Name | str | Name of the restaurant |
| Country Code | int | Country identifier |
| City | str | City where restaurant is located |
| Address | str | Full address |
| Locality | str | Area/locality name |
| Locality Verbose | str | Detailed locality |
| Longitude | float | Geographic longitude |
| Latitude | float | Geographic latitude |
| Cuisines | str | Cuisine types (comma-separated) |
| Average Cost for two | int | Average cost for two people |
| Currency | str | Local currency |
| Has Table booking | str | Yes/No |
| Has Online delivery | str | Yes/No |
| Is delivering now | str | Yes/No |
| Switch to order menu | str | Yes/No |
| Price range | int | 1–4 scale |
| Aggregate rating | float | Overall rating (0–5) — **TARGET** |
| Rating color | str | Color code for rating |
| Rating text | str | Rating category text |
| Votes | int | Number of customer votes |

---

## 🔧 Data Preprocessing

### Steps Performed

1. **Dataset Inspection**: Shape, dtypes, missing values, duplicates
2. **Missing Value Analysis**: Identified and handled missing `Cuisines` entries
3. **Duplicate Removal**: Detected and removed duplicate records
4. **Data Leakage Prevention**: Removed `Rating color` and `Rating text` (derived from target)
5. **Irrelevant Column Removal**: Dropped `Restaurant ID`, `Restaurant Name`, `Address`, `Locality`, `Locality Verbose`, `Currency`, `Switch to order menu`, `Country Code`
6. **Feature Engineering**:
   - `Num_Cuisines`: Count of cuisines per restaurant
   - `Primary_Cuisine`: First listed cuisine (top 20 + Other)
   - Binary encoding for `Has Table booking`, `Has Online delivery`, `Is delivering now`
   - City cardinality reduction (top 30 + Other)
7. **Preprocessing Pipeline**: Scikit-learn `ColumnTransformer` with:
   - `StandardScaler` for numerical features
   - `OneHotEncoder` (drop='first') for categorical features

### Features Used for Prediction

**Numerical (9)**:
- Average Cost for two
- Longitude, Latitude
- Votes
- Price range
- Num_Cuisines
- Has Table booking (encoded)
- Has Online delivery (encoded)
- Is delivering now (encoded)

**Categorical (2)**:
- City (top 30 + Other)
- Primary_Cuisine (top 20 + Other)

---

## 📈 Exploratory Data Analysis

Since visualization libraries (Matplotlib, Seaborn, Plotly) are NOT used, all analysis is done through:

- **Statistical summaries** using Pandas `.describe()`, `.groupby()`, `.corr()`
- **Tabular analysis** with formatted output tables
- **Distribution analysis** using value counts and binning

### Key Relationships Analyzed

| Analysis | Finding |
|----------|---------|
| Votes vs Rating | Restaurants with more votes tend to have higher ratings |
| Price Range vs Rating | Higher price range → generally higher ratings |
| Table Booking vs Rating | Restaurants with table booking have higher average ratings |
| Online Delivery vs Rating | Slight variation in ratings based on delivery availability |
| City vs Rating | Significant variation across cities |
| Cuisine vs Rating | Certain cuisine types correlate with higher ratings |

---

## 🤖 Machine Learning Models

### Algorithms Trained

| # | Model | Type | Library |
|---|-------|------|---------|
| 1 | Linear Regression | Linear | Scikit-learn |
| 2 | Decision Tree Regressor | Tree-based | Scikit-learn |
| 3 | Random Forest Regressor | Ensemble | Scikit-learn |
| 4 | Neural Network | Deep Learning | TensorFlow/Keras |

### Model Evaluation Metrics

| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| **Linear Regression** | 0.9829 | 1.4249 | 1.1937 | 0.3740 |
| **Decision Tree** | 0.2724 | 0.1768 | 0.4205 | 0.9223 |
| **Random Forest (Base)** | 0.1941 | 0.0872 | 0.2952 | 0.9617 |
| **Random Forest (Optimized)** | **0.1928** | **0.0867** | **0.2944** | **0.9619** |

> **Key Finding**: The **Optimized Random Forest** achieves the highest accuracy with an **R² score of ~0.962** and an average error of less than **0.20 rating points (MAE 0.1928)**.

### Metric Definitions

- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual ratings. Lower is better.
- **MSE** (Mean Squared Error): Average squared difference. Penalizes large errors. Lower is better.
- **RMSE** (Root Mean Squared Error): Square root of MSE. In same units as target. Lower is better.
- **R² Score**: Proportion of variance explained by the model. 1.0 = perfect. Higher is better.

---

## 🏆 Model Optimization

The best-performing model (Random Forest) is optimized using:

- **RandomizedSearchCV** (30 parameter iterations across 5 folds)
- **5-fold Cross-Validation** (Best CV R²: 0.9591)
- Best Hyperparameters:
  - `n_estimators`: 300
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1
  - `max_features`: None
  - `max_depth`: None

---

## 📊 Feature Importance

The top features contributing to rating prediction from the best Random Forest model:

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | **Votes** | 94.75% | Customer engagement & review volume are the strongest rating indicators |
| 2 | **Longitude** | 1.81% | Geographic position / regional dining quality clusters |
| 3 | **Latitude** | 1.27% | Regional dining cluster variation |
| 4 | **Average Cost for two** | 0.65% | Pricing & premium tier influence |
| 5 | **Num_Cuisines** | 0.26% | Menu breadth and cuisine specialization |

---

## 🍕 Cuisine & Customer Preference Analysis

### Analysis Performed

1. **Cuisine Processing**: Multi-cuisine fields split and exploded for per-cuisine analysis
2. **Popular Cuisines**: Top 10 cuisines ranked by total customer votes
3. **Highest-Rated Cuisines**: Cuisines with highest average rating (minimum 20 restaurants for reliability)
4. **Preference Comparison**: Side-by-side comparison of most popular vs highest-rated cuisines

### Key Findings

- **Popularity ≠ Highest Rating**: The most-voted cuisines are not necessarily the highest-rated
- **Niche cuisines** often maintain higher average ratings due to specialized focus
- **Votes correlate with exposure**, not directly with quality
- All findings are **correlations**, not causal claims

---

## 🧠 Neural Network (Optional)

A TensorFlow/Keras regression network is built with:

- Input layer matching preprocessed feature dimensions
- Multiple Dense layers (128 → 64 → 32 → 16 → 1)
- ReLU activations + linear output
- Adam optimizer, MSE loss
- 100 epochs with 20% validation split

If the neural network underperforms traditional ML models, the traditional model is selected as final.

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow keras
```

### Run the Interactive Web Application

```bash
# Start the local web dashboard (opens in browser at http://localhost:8000)
python server.py
```

### Run the Machine Learning Pipeline CLI

```bash
# Execute the full training and analysis pipeline
python src/restaurant_analysis.py
```

### Expected Output

The CLI script and Web App will provide:
1. Dataset inspection results
2. Exploratory data analysis tables
3. Preprocessing summary
4. Model training progress
5. Model comparison table
6. Optimization results
7. Feature importance rankings
8. Cuisine analysis tables
9. Final model summary
10. Live interactive rating predictions

---

## 📁 Project Structure

```
restaurant-rating-analysis/
│
├── web/
│   ├── index.html                  # Modern web dashboard interface
│   ├── styles.css                  # Custom glassmorphism design system
│   └── app.js                      # Client ML engine, charts & interactivity
│
├── dataset/
│   └── restaurant.csv              # Restaurant dataset (9,551 records)
│
├── src/
│   └── restaurant_analysis.py      # Complete analysis & ML pipeline
│
├── notebooks/
│   └── restaurant_analysis.ipynb   # Jupyter notebook version
│
├── models/
│   ├── pipeline_config.csv         # Pipeline configuration (auto-generated)
│   └── final_model.keras           # Keras model (if NN is best)
│
├── server.py                       # Lightweight local web server launcher
├── requirements.txt                # Dependencies (5 libraries only)
└── README.md                       # Documentation & Project Guide
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Pandas** | Data loading, preprocessing, analysis |
| **NumPy** | Numerical operations, array manipulation |
| **Scikit-learn** | ML models, preprocessing, evaluation, optimization |
| **TensorFlow** | Neural network backend |
| **Keras** | Neural network model building and training |

> ⚠️ **No other libraries are used.** No Matplotlib, Seaborn, Plotly, SciPy, XGBoost, LightGBM, or Joblib.

---

## 🔮 Future Improvements

1. **Sentiment Analysis**: Incorporate text reviews for NLP-based rating prediction
2. **Geospatial Clustering**: Use location data for regional rating pattern analysis
3. **Time-Series Analysis**: Track rating trends over time if temporal data is available
4. **Recommendation System**: Build a collaborative filtering system based on cuisine preferences
5. **Web Dashboard**: Create a Flask/Django frontend for interactive predictions
6. **More Ensemble Models**: Test Gradient Boosting, XGBoost (outside strict library constraints)
7. **Feature Store**: Build a feature engineering pipeline for real-time predictions
8. **A/B Testing Framework**: Test impact of restaurant changes on predicted ratings

---

## 🎤 Viva Questions and Answers

### 1. What is the objective of this project?
**Answer**: To predict a restaurant's aggregate rating using machine learning and analyze the relationship between cuisine types, customer votes, and ratings.

### 2. Why did you remove 'Rating text' and 'Rating color' from features?
**Answer**: These columns are directly derived from the target variable (Aggregate rating). Including them would cause **data leakage** — the model would learn the answer from the input, giving unrealistically high accuracy that wouldn't generalize to new data.

### 3. Why did you use Random Forest as a primary candidate?
**Answer**: Random Forest is well-suited for tabular regression because it handles nonlinear relationships, mixed feature types, and is robust against overfitting. However, we didn't assume it would be the best — we compared it against Linear Regression, Decision Tree, and a Neural Network.

### 4. What is R² Score and what does it mean?
**Answer**: R² (coefficient of determination) measures the proportion of variance in the target variable explained by the model. R² = 1.0 means perfect prediction, R² = 0 means the model is no better than predicting the mean. Higher is better.

### 5. How did you handle restaurants with multiple cuisines?
**Answer**: The Cuisines column contains comma-separated values (e.g., "French, Japanese, Desserts"). We extracted the primary cuisine (first listed) for feature encoding, created a `Num_Cuisines` count feature, and for cuisine analysis, we exploded multi-cuisine entries into separate rows.

### 6. What preprocessing did you apply?
**Answer**: StandardScaler for numerical features (zero mean, unit variance), OneHotEncoder with drop='first' for categorical features (City, Primary Cuisine), binary encoding for Yes/No columns, and we used ColumnTransformer to combine these in a pipeline.

### 7. Why is the minimum restaurant threshold important in cuisine analysis?
**Answer**: Without a minimum threshold, a cuisine with only 1-2 restaurants could appear as the "highest rated" if those restaurants happen to have high ratings. A minimum of 20 restaurants ensures statistical reliability and prevents misleading conclusions.

### 8. What is the difference between MAE and RMSE?
**Answer**: MAE (Mean Absolute Error) treats all errors equally — it's the average of absolute differences. RMSE (Root Mean Squared Error) squares errors before averaging, so it penalizes large errors more heavily. If a model makes a few large mistakes, RMSE will be significantly higher than MAE.

### 9. Did the Neural Network outperform traditional ML models?
**Answer**: This depends on the actual results. Neural networks often underperform on smaller tabular datasets because they require large amounts of data and are more prone to overfitting. Tree-based models typically excel on structured/tabular data.

### 10. How would you deploy this model in production?
**Answer**: I would create a Flask/FastAPI REST API that accepts restaurant features as JSON input, applies the same preprocessing pipeline, and returns the predicted rating. The model could be containerized with Docker and deployed to a cloud service (AWS, GCP, or Azure).

---

## 📜 License

This project is created for educational and analytical purposes.

---

## 👤 Author

Restaurant Analytics Team

---

*Built with ❤️ using only Pandas, NumPy, Scikit-learn, TensorFlow, and Keras*

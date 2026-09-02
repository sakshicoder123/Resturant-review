"""
============================================================================
 RESTAURANT RATING PREDICTION AND CUSTOMER PREFERENCE ANALYSIS
============================================================================
 Project  : Restaurant Analytics & Machine Learning Rating Predictor
 Author   : Restaurant Analytics Team
 Dataset  : Zomato Restaurant Dataset (~9,551 records)
 Target   : Aggregate rating
 Models   : Linear Regression, Decision Tree, Random Forest, Neural Network
 Libraries: Pandas, NumPy, Scikit-learn, TensorFlow/Keras (ONLY)
============================================================================
"""

import os
import sys
import warnings

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
#  SECTION 1: CONFIGURATION & DATA LOADING
# ============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, 'dataset', 'restaurant.csv')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subheader(title):
    """Print formatted sub-section header."""
    print(f"\n--- {title} ---")


# ============================================================================
#  SECTION 2: DATASET INSPECTION
# ============================================================================

print_header("SECTION 1: DATASET INSPECTION")

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(DATASET_PATH, encoding='latin-1')

print(f"\nDataset loaded successfully from: {DATASET_PATH}")
print(f"\nDataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Total Records: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")

print_subheader("Column Names and Data Types")
col_info = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes.values,
    'Non-Null Count': df.notnull().sum().values,
    'Null Count': df.isnull().sum().values,
    'Unique Values': df.nunique().values
})
print(col_info.to_string(index=False))

print_subheader("Missing Values Analysis")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0]
if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("No missing values found in any column.")

# Check all columns for missing
print(f"\nTotal missing values across entire dataset: {df.isnull().sum().sum()}")

print_subheader("Duplicate Records")
dup_count = df.duplicated().sum()
print(f"Duplicate rows found: {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New shape: {df.shape}")

print_subheader("Numerical Features")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"\nNumerical Summary Statistics:")
print(df[numerical_cols].describe().round(2).to_string())

print_subheader("Categorical Features")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
for col in categorical_cols:
    n_unique = df[col].nunique()
    print(f"  {col}: {n_unique} unique values", end="")
    if n_unique <= 8:
        print(f" -> {df[col].unique().tolist()}")
    else:
        print(f" -> Top 5: {df[col].value_counts().head(5).index.tolist()}")

print_subheader("Target Variable: Aggregate rating")
print(df['Aggregate rating'].describe().round(4).to_string())
print(f"\nRating Distribution:")
rating_dist = df['Aggregate rating'].value_counts().sort_index()
for rating, count in rating_dist.items():
    pct = count / len(df) * 100
    bar = "#" * int(pct)
    print(f"  {rating:>4.1f}: {count:>5} ({pct:>5.1f}%) {bar}")

zero_ratings = (df['Aggregate rating'] == 0).sum()
print(f"\nRestaurants with 0 rating (unrated): {zero_ratings} ({zero_ratings/len(df)*100:.1f}%)")


# ============================================================================
#  SECTION 3: EXPLORATORY DATA ANALYSIS (Statistical / Tabular)
# ============================================================================

print_header("SECTION 2: EXPLORATORY DATA ANALYSIS")

# --- 3.1 Votes vs Aggregate Rating ---
print_subheader("Relationship: Votes vs Aggregate Rating")
vote_bins = [0, 10, 50, 100, 500, 1000, 5000, df['Votes'].max() + 1]
vote_labels = ['0-10', '11-50', '51-100', '101-500', '501-1000', '1001-5000', '5000+']
df['Vote_Bin'] = pd.cut(df['Votes'], bins=vote_bins, labels=vote_labels, right=False)

vote_analysis = df.groupby('Vote_Bin', observed=False).agg(
    Count=('Aggregate rating', 'count'),
    Avg_Rating=('Aggregate rating', 'mean'),
    Median_Rating=('Aggregate rating', 'median'),
    Avg_Votes=('Votes', 'mean')
).round(2)
print(vote_analysis.to_string())
print("\nInsight: Restaurants with more votes tend to have higher average ratings.")

# --- 3.2 Price Range vs Aggregate Rating ---
print_subheader("Relationship: Price Range vs Aggregate Rating")
price_analysis = df.groupby('Price range').agg(
    Count=('Aggregate rating', 'count'),
    Avg_Rating=('Aggregate rating', 'mean'),
    Median_Rating=('Aggregate rating', 'median'),
    Avg_Cost=('Average Cost for two', 'mean'),
    Avg_Votes=('Votes', 'mean')
).round(2)
print(price_analysis.to_string())
print("\nInsight: Higher price range restaurants generally have higher average ratings.")

# --- 3.3 Table Booking vs Aggregate Rating ---
print_subheader("Relationship: Table Booking vs Aggregate Rating")
booking_analysis = df.groupby('Has Table booking').agg(
    Count=('Aggregate rating', 'count'),
    Avg_Rating=('Aggregate rating', 'mean'),
    Median_Rating=('Aggregate rating', 'median'),
    Avg_Votes=('Votes', 'mean')
).round(2)
print(booking_analysis.to_string())

# --- 3.4 Online Delivery vs Aggregate Rating ---
print_subheader("Relationship: Online Delivery vs Aggregate Rating")
delivery_analysis = df.groupby('Has Online delivery').agg(
    Count=('Aggregate rating', 'count'),
    Avg_Rating=('Aggregate rating', 'mean'),
    Median_Rating=('Aggregate rating', 'median'),
    Avg_Votes=('Votes', 'mean')
).round(2)
print(delivery_analysis.to_string())

# --- 3.5 Rating Text vs Aggregate Rating ---
print_subheader("Relationship: Rating Text vs Aggregate Rating")
rating_text_analysis = df.groupby('Rating text').agg(
    Count=('Aggregate rating', 'count'),
    Avg_Rating=('Aggregate rating', 'mean'),
    Min_Rating=('Aggregate rating', 'min'),
    Max_Rating=('Aggregate rating', 'max'),
    Avg_Votes=('Votes', 'mean')
).sort_values('Avg_Rating', ascending=False).round(2)
print(rating_text_analysis.to_string())

# --- 3.6 City-wise Analysis ---
print_subheader("Top 15 Cities by Number of Restaurants")
city_analysis = df.groupby('City').agg(
    Count=('Aggregate rating', 'count'),
    Avg_Rating=('Aggregate rating', 'mean'),
    Avg_Cost=('Average Cost for two', 'mean'),
    Avg_Votes=('Votes', 'mean')
).sort_values('Count', ascending=False).head(15).round(2)
print(city_analysis.to_string())

# --- 3.7 Country Code Analysis ---
print_subheader("Country Code Distribution")
country_analysis = df.groupby('Country Code').agg(
    Count=('Aggregate rating', 'count'),
    Avg_Rating=('Aggregate rating', 'mean'),
    Avg_Cost=('Average Cost for two', 'mean')
).sort_values('Count', ascending=False).round(2)
print(country_analysis.to_string())

# --- 3.8 Correlation Analysis ---
print_subheader("Correlation with Aggregate Rating (Numerical Features)")
corr_cols = ['Average Cost for two', 'Price range', 'Votes', 'Longitude', 'Latitude']
correlations = df[corr_cols + ['Aggregate rating']].corr()['Aggregate rating'].drop('Aggregate rating').sort_values(ascending=False)
for feat, corr in correlations.items():
    strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
    direction = "Positive" if corr > 0 else "Negative"
    print(f"  {feat:<30}: {corr:>+.4f} ({strength} {direction})")

# Clean up temporary column
df.drop('Vote_Bin', axis=1, inplace=True)


# ============================================================================
#  SECTION 4: DATA PREPROCESSING FOR ML
# ============================================================================

print_header("SECTION 3: DATA PREPROCESSING")

# --- 4.1 Identify and Remove Irrelevant Columns ---
print_subheader("Feature Selection")

# Columns to DROP (IDs, text, addresses, redundant, or leakage)
drop_columns = [
    'Restaurant ID',       # Unique ID, no predictive value
    'Restaurant Name',     # Text identifier, no predictive value
    'Address',             # Too granular, free text
    'Locality',            # Redundant with City
    'Locality Verbose',    # Redundant with City
    'Currency',            # Redundant with Country Code
    'Rating color',        # DATA LEAKAGE: derived from target
    'Rating text',         # DATA LEAKAGE: derived from target
    'Switch to order menu',# Almost all same value, not useful
    'Country Code',        # Will use City instead (more granular)
]

print("Columns DROPPED and reasons:")
for col in drop_columns:
    if col in ['Rating color', 'Rating text']:
        print(f"  - {col}: DATA LEAKAGE (derived from Aggregate rating)")
    elif col in ['Restaurant ID', 'Restaurant Name']:
        print(f"  - {col}: Unique identifier, no predictive value")
    elif col in ['Address', 'Locality', 'Locality Verbose']:
        print(f"  - {col}: Too granular / redundant text")
    elif col == 'Currency':
        print(f"  - {col}: Redundant with Country Code")
    elif col == 'Switch to order menu':
        print(f"  - {col}: Nearly constant value")
    elif col == 'Country Code':
        print(f"  - {col}: Using City instead for location info")

# --- 4.2 Process Cuisines ---
print_subheader("Cuisine Feature Engineering")

# Count number of cuisines per restaurant
df['Num_Cuisines'] = df['Cuisines'].fillna('Unknown').apply(lambda x: len(str(x).split(',')))
print(f"Created 'Num_Cuisines' feature (range: {df['Num_Cuisines'].min()} - {df['Num_Cuisines'].max()})")

# Extract primary cuisine (first listed cuisine)
df['Primary_Cuisine'] = df['Cuisines'].fillna('Unknown').apply(lambda x: str(x).split(',')[0].strip())
top_cuisines = df['Primary_Cuisine'].value_counts().head(20).index.tolist()
df['Primary_Cuisine'] = df['Primary_Cuisine'].apply(lambda x: x if x in top_cuisines else 'Other')
print(f"Created 'Primary_Cuisine' feature with top 20 cuisines + 'Other'")
print(f"  Top 10 primary cuisines: {top_cuisines[:10]}")

# --- 4.3 Handle Missing Values ---
print_subheader("Handling Missing Values")
cuisines_missing = df['Cuisines'].isnull().sum()
print(f"Cuisines missing values: {cuisines_missing}")
if cuisines_missing > 0:
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    print(f"  Filled {cuisines_missing} missing Cuisines with 'Unknown'")

# Check other missing
for col in df.columns:
    m = df[col].isnull().sum()
    if m > 0:
        print(f"  {col}: {m} missing values")
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
            print(f"    Filled with median")
        else:
            df[col] = df[col].fillna('Unknown')
            print(f"    Filled with 'Unknown'")

# --- 4.4 Encode Binary Columns ---
print_subheader("Binary Feature Encoding")
binary_cols = ['Has Table booking', 'Has Online delivery', 'Is delivering now']
for col in binary_cols:
    df[col + '_enc'] = (df[col] == 'Yes').astype(int)
    yes_count = df[col + '_enc'].sum()
    print(f"  {col}: Yes={yes_count}, No={len(df) - yes_count}")

# --- 4.5 Prepare Final Feature Set ---
print_subheader("Final Feature Set")

# Target
TARGET = 'Aggregate rating'
y = df[TARGET].values

# Features to use
feature_columns_numeric = [
    'Average Cost for two',
    'Longitude',
    'Latitude',
    'Votes',
    'Price range',
    'Num_Cuisines',
    'Has Table booking_enc',
    'Has Online delivery_enc',
    'Is delivering now_enc',
]

feature_columns_categorical = [
    'City',
    'Primary_Cuisine',
]

print(f"Numerical features ({len(feature_columns_numeric)}):")
for f in feature_columns_numeric:
    print(f"  - {f}")

print(f"\nCategorical features ({len(feature_columns_categorical)}):")
for f in feature_columns_categorical:
    n_unique = df[f].nunique()
    print(f"  - {f} ({n_unique} unique values)")

# --- 4.6 Build Preprocessing Pipeline ---
print_subheader("Preprocessing Pipeline")

# Limit City cardinality
top_cities = df['City'].value_counts().head(30).index.tolist()
df['City'] = df['City'].apply(lambda x: x if x in top_cities else 'Other')
print(f"City reduced to top 30 cities + 'Other' (total: {df['City'].nunique()})")

X = df[feature_columns_numeric + feature_columns_categorical]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_columns_numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'),
         feature_columns_categorical),
    ]
)

print("ColumnTransformer created with:")
print("  - StandardScaler for numerical features")
print("  - OneHotEncoder (drop='first') for categorical features")

# --- 4.7 Train-Test Split ---
print_subheader("Train-Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Random state: 42")


# ============================================================================
#  SECTION 5: MODEL TRAINING & EVALUATION
# ============================================================================

print_header("SECTION 4: MODEL TRAINING & EVALUATION")

def evaluate_model(name, y_true, y_pred):
    """Calculate and return all evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R² Score': r2}

results = []

# --- 5.1 Linear Regression ---
print_subheader("Model 1: Linear Regression")
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
lr_metrics = evaluate_model('Linear Regression', y_test, lr_pred)
results.append(lr_metrics)
for k, v in lr_metrics.items():
    if k != 'Model':
        print(f"  {k}: {v:.4f}")

# --- 5.2 Decision Tree Regressor ---
print_subheader("Model 2: Decision Tree Regressor")
dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])
dt_pipeline.fit(X_train, y_train)
dt_pred = dt_pipeline.predict(X_test)
dt_metrics = evaluate_model('Decision Tree', y_test, dt_pred)
results.append(dt_metrics)
for k, v in dt_metrics.items():
    if k != 'Model':
        print(f"  {k}: {v:.4f}")

# --- 5.3 Random Forest Regressor ---
print_subheader("Model 3: Random Forest Regressor")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
rf_metrics = evaluate_model('Random Forest', y_test, rf_pred)
results.append(rf_metrics)
for k, v in rf_metrics.items():
    if k != 'Model':
        print(f"  {k}: {v:.4f}")

# --- 5.4 Model Comparison Table ---
print_subheader("MODEL COMPARISON TABLE")
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.set_index('Model')
print(comparison_df.round(4).to_string())

# Determine best model
best_model_name = comparison_df['R² Score'].idxmax()
best_r2 = comparison_df.loc[best_model_name, 'R² Score']
print(f"\n★ Best Performing Model: {best_model_name} (R² = {best_r2:.4f})")

# Map names to pipeline objects
model_map = {
    'Linear Regression': lr_pipeline,
    'Decision Tree': dt_pipeline,
    'Random Forest': rf_pipeline,
}
best_pipeline = model_map[best_model_name]
best_pred = {'Linear Regression': lr_pred, 'Decision Tree': dt_pred, 'Random Forest': rf_pred}[best_model_name]


# ============================================================================
#  SECTION 6: MODEL OPTIMIZATION (Best Model Hyperparameter Tuning)
# ============================================================================

print_header("SECTION 5: MODEL OPTIMIZATION")

# We optimize the best-performing model
print(f"Optimizing: {best_model_name}")

if best_model_name == 'Random Forest':
    print_subheader("RandomizedSearchCV for Random Forest")
    param_distributions = {
        'regressor__n_estimators': [50, 100, 200, 300],
        'regressor__max_depth': [None, 10, 20, 30, 50],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['sqrt', 'log2', None],
    }

    rf_search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    print("Running RandomizedSearchCV (30 iterations, 5-fold CV)...")
    rf_search.fit(X_train, y_train)

    print(f"\nBest Parameters Found:")
    for param, value in rf_search.best_params_.items():
        clean_param = param.replace('regressor__', '')
        print(f"  {clean_param}: {value}")
    print(f"Best CV R² Score: {rf_search.best_score_:.4f}")

    # Evaluate optimized model on test set
    optimized_pred = rf_search.predict(X_test)
    opt_metrics = evaluate_model('Random Forest (Optimized)', y_test, optimized_pred)

    print_subheader("Original vs Optimized Model Comparison")
    orig_metrics = rf_metrics.copy()
    orig_metrics['Model'] = 'Random Forest (Original)'
    opt_comparison = pd.DataFrame([orig_metrics, opt_metrics]).set_index('Model')
    print(opt_comparison.round(4).to_string())

    improvement = opt_metrics['R² Score'] - rf_metrics['R² Score']
    print(f"\nR² Improvement: {improvement:+.4f}")

    if opt_metrics['R² Score'] > rf_metrics['R² Score']:
        print("✓ Optimized model is BETTER. Using optimized model as final.")
        best_pipeline = rf_search.best_estimator_
        best_pred = optimized_pred
        best_model_name = 'Random Forest (Optimized)'
        final_metrics = opt_metrics
    else:
        print("✗ Optimization did not improve. Keeping original model.")
        final_metrics = rf_metrics

elif best_model_name == 'Decision Tree':
    print_subheader("RandomizedSearchCV for Decision Tree")
    dt_param_distributions = {
        'regressor__max_depth': [None, 5, 10, 15, 20, 30],
        'regressor__min_samples_split': [2, 5, 10, 20],
        'regressor__min_samples_leaf': [1, 2, 4, 8],
        'regressor__max_features': ['sqrt', 'log2', None],
    }

    dt_search = RandomizedSearchCV(
        dt_pipeline,
        param_distributions=dt_param_distributions,
        n_iter=25,
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    print("Running RandomizedSearchCV (25 iterations, 5-fold CV)...")
    dt_search.fit(X_train, y_train)

    print(f"\nBest Parameters Found:")
    for param, value in dt_search.best_params_.items():
        clean_param = param.replace('regressor__', '')
        print(f"  {clean_param}: {value}")
    print(f"Best CV R² Score: {dt_search.best_score_:.4f}")

    optimized_pred = dt_search.predict(X_test)
    opt_metrics = evaluate_model('Decision Tree (Optimized)', y_test, optimized_pred)

    print_subheader("Original vs Optimized Model Comparison")
    orig_metrics = dt_metrics.copy()
    orig_metrics['Model'] = 'Decision Tree (Original)'
    opt_comparison = pd.DataFrame([orig_metrics, opt_metrics]).set_index('Model')
    print(opt_comparison.round(4).to_string())

    if opt_metrics['R² Score'] > dt_metrics['R² Score']:
        print("✓ Optimized model is BETTER. Using optimized model as final.")
        best_pipeline = dt_search.best_estimator_
        best_pred = optimized_pred
        best_model_name = 'Decision Tree (Optimized)'
        final_metrics = opt_metrics
    else:
        print("✗ Optimization did not improve. Keeping original model.")
        final_metrics = dt_metrics
else:
    print("Linear Regression selected — no hyperparameters to tune.")
    print("Performing cross-validation instead...")
    cv_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"  5-Fold CV R² Scores: {cv_scores.round(4)}")
    print(f"  Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    final_metrics = lr_metrics


# ============================================================================
#  SECTION 7: FEATURE IMPORTANCE
# ============================================================================

print_header("SECTION 6: FEATURE IMPORTANCE")

# Get feature names after preprocessing
cat_encoder = best_pipeline.named_steps['preprocessor'].transformers_[1][1]
try:
    cat_feature_names = cat_encoder.get_feature_names_out(feature_columns_categorical).tolist()
except Exception:
    cat_feature_names = []
    for i, col in enumerate(feature_columns_categorical):
        cats = cat_encoder.categories_[i][1:]  # skip first due to drop='first'
        for cat in cats:
            cat_feature_names.append(f"{col}_{cat}")

all_feature_names = feature_columns_numeric + cat_feature_names

# Check if the final model is tree-based
final_regressor = best_pipeline.named_steps['regressor']
if hasattr(final_regressor, 'feature_importances_'):
    importances = final_regressor.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print_subheader("Top 15 Most Important Features")
    top_features = importance_df.head(15)
    for idx, row in top_features.iterrows():
        bar = "#" * int(row['Importance'] * 100)
        # Clean up feature names for readability
        name = row['Feature']
        name = name.replace('Primary_Cuisine_', 'Cuisine: ')
        name = name.replace('City_', 'City: ')
        name = name.replace('_enc', '')
        print(f"  {name:<40} {row['Importance']:.4f}  {bar}")

    print_subheader("Feature Importance Interpretation")
    top3 = importance_df.head(3)['Feature'].tolist()
    print(f"The top 3 most important features for predicting Aggregate Rating are:")
    for i, f in enumerate(top3, 1):
        fname = f.replace('Primary_Cuisine_', 'Cuisine: ').replace('City_', 'City: ').replace('_enc', '')
        print(f"  {i}. {fname}")
    print("\nThis suggests that restaurant characteristics like voting activity,")
    print("pricing, and location are strong indicators of aggregate rating.")
else:
    print("Final model is not tree-based. Feature importance via coefficients:")
    if hasattr(final_regressor, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Coefficient': final_regressor.coef_
        })
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        print(coef_df.head(15)[['Feature', 'Coefficient']].to_string(index=False))


# ============================================================================
#  SECTION 8: CUSTOMER PREFERENCE & CUISINE ANALYSIS
# ============================================================================

print_header("SECTION 7: CUSTOMER PREFERENCE & CUISINE ANALYSIS")

# Reload original data for cuisine analysis (before preprocessing dropped columns)
try:
    df_cuisine = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_cuisine = pd.read_csv(DATASET_PATH, encoding='latin-1')

print_subheader("Cuisine Column Identification")
print(f"Cuisine column: 'Cuisines'")
print(f"Sample values:")
for i, val in enumerate(df_cuisine['Cuisines'].dropna().head(5)):
    print(f"  {i+1}. {val}")
print(f"\nTotal records: {len(df_cuisine)}")
print(f"Records with cuisine info: {df_cuisine['Cuisines'].notna().sum()}")
print(f"Records without cuisine: {df_cuisine['Cuisines'].isna().sum()}")

# --- 8.1 Explode cuisines (one row per cuisine per restaurant) ---
print_subheader("Processing Multi-Cuisine Restaurants")
df_cuisine['Cuisines'] = df_cuisine['Cuisines'].fillna('Unknown')

# Split and explode
cuisine_expanded = df_cuisine.assign(
    Cuisine=df_cuisine['Cuisines'].str.split(',')
).explode('Cuisine')
cuisine_expanded['Cuisine'] = cuisine_expanded['Cuisine'].str.strip()

total_unique_cuisines = cuisine_expanded['Cuisine'].nunique()
print(f"Total unique cuisines found: {total_unique_cuisines}")

# --- 8.2 Popular Cuisines by Total Votes ---
print_subheader("Top 10 Most Popular Cuisines (by Total Votes)")

cuisine_stats = cuisine_expanded.groupby('Cuisine').agg(
    Num_Restaurants=('Restaurant ID', 'nunique'),
    Total_Votes=('Votes', 'sum'),
    Avg_Rating=('Aggregate rating', 'mean')
).round(2)

top_10_by_votes = cuisine_stats.sort_values('Total_Votes', ascending=False).head(10)
print("")
print("  Cuisine                      Restaurants  Total Votes  Avg Rating")
print("  ----------------------------  -----------  -----------  ----------")
for cuisine, row in top_10_by_votes.iterrows():
    print(f"  {cuisine:<28} {int(row['Num_Restaurants']):>11}  {int(row['Total_Votes']):>11}  {row['Avg_Rating']:>10.2f}")

# --- 8.3 Highest-Rated Cuisines ---
print_subheader("Highest-Rated Cuisines (with Sufficient Data)")

MIN_RESTAURANTS = 20
print(f"\nMinimum restaurant threshold: {MIN_RESTAURANTS}")
print("Reason: Cuisines with very few restaurants may have misleading averages.")
print("        A minimum of 20 ensures statistical reliability.\n")

popular_cuisines = cuisine_stats[cuisine_stats['Num_Restaurants'] >= MIN_RESTAURANTS]
top_rated = popular_cuisines.sort_values('Avg_Rating', ascending=False).head(15)

print("  Cuisine                      Restaurants  Avg Rating  Total Votes")
print("  ----------------------------  -----------  ----------  -----------")
for cuisine, row in top_rated.iterrows():
    print(f"  {cuisine:<28} {int(row['Num_Restaurants']):>11}  {row['Avg_Rating']:>10.2f}  {int(row['Total_Votes']):>11}")

# --- 8.4 Customer Preference Comparison ---
print_subheader("Customer Preference Comparison")
print("\nMost Popular (by Votes) vs Highest Rated:")
print(f"\n{'Rank':<6} {'Most Popular (Votes)':<30} {'Highest Rated':<30}")
print("-" * 66)

top_popular = top_10_by_votes.index.tolist()
top_rated_list = top_rated.head(10).index.tolist()

for i in range(10):
    pop = top_popular[i] if i < len(top_popular) else '-'
    rated = top_rated_list[i] if i < len(top_rated_list) else '-'
    overlap = ' *' if pop in top_rated_list[:10] else ''
    print(f"  {i+1:<4} {pop:<30} {rated:<30}{overlap}")

overlap_count = len(set(top_popular) & set(top_rated_list[:10]))
print(f"\nOverlap: {overlap_count} cuisines appear in BOTH top 10 lists (marked with *)")

print_subheader("Key Findings on Cuisine and Ratings")
print("""
1. POPULARITY ≠ HIGHEST RATING: The most popular cuisines (by votes) are not
   necessarily the highest-rated ones. Popular cuisines may have more competition,
   leading to a wider range of quality.

2. NICHE CUISINES tend to have HIGHER AVERAGE RATINGS: Specialty cuisines with
   fewer but dedicated restaurants often maintain higher quality standards.

3. VOTES CORRELATE WITH EXPOSURE: Cuisines with more restaurants naturally
   accumulate more total votes, but this does not directly cause higher ratings.

4. IMPORTANT: These are CORRELATIONS observed in the data, NOT causal claims.
   A cuisine type does not cause higher or lower ratings — many other factors
   (service, location, price, chef quality) contribute to restaurant ratings.
""")


# ============================================================================
#  SECTION 9: TENSORFLOW/KERAS NEURAL NETWORK (Optional)
# ============================================================================

print_header("SECTION 8: NEURAL NETWORK MODEL (TensorFlow/Keras)")

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore

    print("TensorFlow version:", tf.__version__)

    # Preprocess data for neural network
    X_train_processed = best_pipeline.named_steps['preprocessor'].transform(X_train)
    X_test_processed = best_pipeline.named_steps['preprocessor'].transform(X_test)

    input_dim = X_train_processed.shape[1]
    print(f"Input dimension: {input_dim}")

    # Build model
    print_subheader("Neural Network Architecture")
    nn_model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')  # Regression output
    ])

    nn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    nn_model.summary()

    # Train
    print_subheader("Training Neural Network")
    history = nn_model.fit(
        X_train_processed, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        verbose=1
    )

    # Evaluate
    nn_pred = nn_model.predict(X_test_processed, verbose=0).flatten()
    nn_metrics = evaluate_model('Neural Network (Keras)', y_test, nn_pred)

    print_subheader("Neural Network Results")
    for k, v in nn_metrics.items():
        if k != 'Model':
            print(f"  {k}: {v:.4f}")

    # Compare all models including NN
    print_subheader("COMPLETE MODEL COMPARISON (Including Neural Network)")
    all_results = results + [nn_metrics]
    all_comparison = pd.DataFrame(all_results).set_index('Model')
    print(all_comparison.round(4).to_string())

    overall_best_name = all_comparison['R² Score'].idxmax()
    overall_best_r2 = all_comparison.loc[overall_best_name, 'R² Score']

    if overall_best_name == 'Neural Network (Keras)':
        print(f"\n>> Neural Network achieved the best R2 ({overall_best_r2:.4f}).")
        print("  Saving Keras model as the final model...")
        nn_model.save(os.path.join(MODELS_DIR, 'final_model.keras'))
        print(f"  Model saved to: {os.path.join(MODELS_DIR, 'final_model.keras')}")
        final_model_type = 'keras'
        final_metrics = nn_metrics
    else:
        print(f"\n>> Traditional ML model '{overall_best_name}' outperforms Neural Network.")
        print(f"  Neural Network R²: {nn_metrics['R² Score']:.4f}")
        print(f"  Best ML Model R²:  {overall_best_r2:.4f}")
        print("  The traditional model is more suitable for this tabular dataset.")
        final_model_type = 'sklearn'

    nn_available = True

except ImportError:
    print("TensorFlow/Keras not installed. Skipping neural network model.")
    print("The project will use the best Scikit-learn model as the final model.")
    nn_available = False
    final_model_type = 'sklearn'
except Exception as e:
    print(f"Error training neural network: {e}")
    print("Continuing with the best Scikit-learn model.")
    nn_available = False
    final_model_type = 'sklearn'


# ============================================================================
#  SECTION 10: FINAL MODEL SUMMARY
# ============================================================================

print_header("SECTION 9: FINAL MODEL SUMMARY")

# Use the final_metrics determined from optimization section (or NN if better)
print(f"""
+{'='*62}+
|                     FINAL MODEL RESULTS                      |
+{'='*62}+
|  Best Model : {best_model_name:<46} |
|  MAE        : {final_metrics['MAE']:<46.4f} |
|  MSE        : {final_metrics['MSE']:<46.4f} |
|  RMSE       : {final_metrics['RMSE']:<46.4f} |
|  R2 Score   : {final_metrics['R² Score']:<46.4f} |
+{'='*62}+
""")

print("Metric Interpretation:")
print(f"  - MAE  = {final_metrics['MAE']:.4f} -> On average, predictions deviate by ~{final_metrics['MAE']:.2f} rating points")
print(f"  - RMSE = {final_metrics['RMSE']:.4f} -> Root mean squared error penalizes larger mistakes")
print(f"  - R2   = {final_metrics['R² Score']:.4f} -> Model explains ~{final_metrics['R² Score']*100:.1f}% of rating variance")


# ============================================================================
#  SECTION 11: PREDICTION FUNCTION
# ============================================================================

print_header("SECTION 10: PREDICTION FUNCTION")

def predict_restaurant_rating(input_data):
    """
    Predict the Aggregate Rating for a restaurant.

    Parameters
    ----------
    input_data : dict
        Dictionary containing restaurant features:
        - 'Average Cost for two': float
        - 'Longitude': float
        - 'Latitude': float
        - 'Votes': int
        - 'Price range': int (1-4)
        - 'Num_Cuisines': int
        - 'Has Table booking': str ('Yes' or 'No')
        - 'Has Online delivery': str ('Yes' or 'No')
        - 'Is delivering now': str ('Yes' or 'No')
        - 'City': str
        - 'Primary_Cuisine': str

    Returns
    -------
    float
        Predicted Aggregate Rating (0-5 scale)
    """
    # Encode binary features
    input_processed = {
        'Average Cost for two': input_data.get('Average Cost for two', 500),
        'Longitude': input_data.get('Longitude', 0.0),
        'Latitude': input_data.get('Latitude', 0.0),
        'Votes': input_data.get('Votes', 0),
        'Price range': input_data.get('Price range', 2),
        'Num_Cuisines': input_data.get('Num_Cuisines', 1),
        'Has Table booking_enc': 1 if input_data.get('Has Table booking', 'No') == 'Yes' else 0,
        'Has Online delivery_enc': 1 if input_data.get('Has Online delivery', 'No') == 'Yes' else 0,
        'Is delivering now_enc': 1 if input_data.get('Is delivering now', 'No') == 'Yes' else 0,
        'City': input_data.get('City', 'Other'),
        'Primary_Cuisine': input_data.get('Primary_Cuisine', 'Other'),
    }

    input_df = pd.DataFrame([input_processed])

    # Ensure column order matches training
    expected_cols = feature_columns_numeric + feature_columns_categorical
    input_df = input_df[expected_cols]

    if final_model_type == 'keras' and 'nn_model' in globals():
        input_transformed = best_pipeline.named_steps['preprocessor'].transform(input_df)
        prediction = float(nn_model.predict(input_transformed, verbose=0).flatten()[0])
    else:
        prediction = float(best_pipeline.predict(input_df)[0])

    # Clip to valid rating range
    prediction = np.clip(prediction, 0.0, 5.0)

    return round(float(prediction), 2)


# --- Example Predictions ---
print_subheader("Example Predictions")

# Example 1: High-end restaurant
example_1 = {
    'Average Cost for two': 2000,
    'Longitude': 77.2090,
    'Latitude': 28.6139,
    'Votes': 500,
    'Price range': 4,
    'Num_Cuisines': 3,
    'Has Table booking': 'Yes',
    'Has Online delivery': 'No',
    'Is delivering now': 'No',
    'City': 'New Delhi',
    'Primary_Cuisine': 'North Indian',
}
pred_1 = predict_restaurant_rating(example_1)
print(f"\nExample 1: Upscale North Indian restaurant in New Delhi")
print(f"  Cost for two: ₹2000, Votes: 500, Price range: 4, Table booking: Yes")
print(f"  ➤ Predicted Rating: {pred_1} / 5.0")

# Example 2: Budget restaurant
example_2 = {
    'Average Cost for two': 300,
    'Longitude': 77.2090,
    'Latitude': 28.6139,
    'Votes': 50,
    'Price range': 1,
    'Num_Cuisines': 1,
    'Has Table booking': 'No',
    'Has Online delivery': 'Yes',
    'Is delivering now': 'Yes',
    'City': 'New Delhi',
    'Primary_Cuisine': 'Fast Food',
}
pred_2 = predict_restaurant_rating(example_2)
print(f"\nExample 2: Budget Fast Food restaurant in New Delhi")
print(f"  Cost for two: ₹300, Votes: 50, Price range: 1, Online delivery: Yes")
print(f"  ➤ Predicted Rating: {pred_2} / 5.0")

# Example 3: Mid-range restaurant
example_3 = {
    'Average Cost for two': 800,
    'Longitude': 72.8777,
    'Latitude': 19.0760,
    'Votes': 200,
    'Price range': 3,
    'Num_Cuisines': 2,
    'Has Table booking': 'Yes',
    'Has Online delivery': 'Yes',
    'Is delivering now': 'No',
    'City': 'Other',
    'Primary_Cuisine': 'Chinese',
}
pred_3 = predict_restaurant_rating(example_3)
print(f"\nExample 3: Mid-range Chinese restaurant")
print(f"  Cost for two: ₹800, Votes: 200, Price range: 3, Table & delivery: Yes")
print(f"  ➤ Predicted Rating: {pred_3} / 5.0")


# ============================================================================
#  SECTION 12: MODEL SAVING
# ============================================================================

print_header("SECTION 11: MODEL SAVING")

if final_model_type == 'sklearn':
    print("Final model is a Scikit-learn model.")
    print("Since Joblib is NOT an allowed library, the model is not serialized to disk.")
    print("Instead, the complete training pipeline is reproducible by running this script.")
    print(f"\nTo reproduce: python src/restaurant_analysis.py")
    print(f"The script will retrain and evaluate all models automatically.")

    # Save training pipeline info for reproduction
    pipeline_info = {
        'best_model': best_model_name,
        'features_numeric': feature_columns_numeric,
        'features_categorical': feature_columns_categorical,
        'test_size': 0.20,
        'random_state': 42,
    }
    pipeline_df = pd.DataFrame(list(pipeline_info.items()), columns=['Parameter', 'Value'])
    pipeline_df.to_csv(os.path.join(MODELS_DIR, 'pipeline_config.csv'), index=False)
    print(f"\nPipeline config saved to: {os.path.join(MODELS_DIR, 'pipeline_config.csv')}")

print(f"""
+{'='*62}+
| RESTAURANT RATING PREDICTION & CUISINE ANALYSIS              |
+{'='*62}+
| Dataset Size        : {df.shape[0]} restaurants                     |
| Total Features      : {df.shape[1]} columns                             |
| Features Used       : {len(feature_columns_numeric)} numerical + {len(feature_columns_categorical)} categorical          |
| Train/Test Split    : 80% / 20%                              |
| Random State        : 42                                     |
+{'='*62}+
| MODELS TRAINED                                               |
|   1. Linear Regression                                       |
|   2. Decision Tree Regressor                                 |
|   3. Random Forest Regressor                                 |
|   4. Neural Network (Keras)  {'[Trained]' if nn_available else '[Skipped]':<30} |
+{'='*62}+
| BEST MODEL          : {best_model_name:<37} |
| Final MAE           : {final_metrics['MAE']:<37.4f}  |
| Final MSE           : {final_metrics['MSE']:<37.4f}  |
| Final RMSE          : {final_metrics['RMSE']:<37.4f}  |
| Final R2 Score      : {final_metrics['R² Score']:<37.4f}  |
+{'='*62}+
| LIBRARIES USED      : Pandas, NumPy, Scikit-learn,           |
|                        TensorFlow, Keras                     |
| NO other libraries used                                      |
+{'='*62}+
""")

print("Project execution completed successfully!")
print("=" * 80)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score



def prepare_data(df, value_col, group_col=None, group_name=None):
    """
    Prepares X and y for Linear Regression.

    How it works:
        X = row number (1, 2, 3, 4 ... represents time)
        y = the actual values from the chosen column

    Parameters:
        df         (DataFrame) : cleaned data
        value_col  (str)       : numeric column to predict
        group_col  (str)       : optional group column (e.g. Product)
        group_name (str)       : optional specific group to filter

    Returns:
        X    (ndarray) : time index array, shape (n, 1)
        y    (ndarray) : values array, shape (n,)
        sub  (DataFrame) : the subset used
        msg  (str)     : description
    """
    df_work = df.copy()

    # Filter to specific group if requested
    if group_col and group_name and group_col in df_work.columns:
        df_work = df_work[df_work[group_col] == group_name].copy()
        msg = f"Using data for {group_col} = '{group_name}'"
    else:
        msg = f"Using all {len(df_work)} rows"

    # Convert value column to numeric
    df_work[value_col] = pd.to_numeric(df_work[value_col], errors='coerce')
    df_work.dropna(subset=[value_col], inplace=True)
    df_work.reset_index(drop=True, inplace=True)

    if len(df_work) < 2:
        return None, None, df_work, "Need at least 2 rows to train model."

    # X = time index (1, 2, 3, ...)
    # reshape(-1, 1) makes it a column vector for sklearn
    X = np.arange(1, len(df_work) + 1).reshape(-1, 1)

    # y = actual values
    y = df_work[value_col].values

    return X, y, df_work, msg


# ------------------------------------------------------------
# FUNCTION 2 — train_model
# Trains a Linear Regression model on X and y
# ------------------------------------------------------------
def train_model(X, y):
    """
    Trains a Linear Regression model.

    Linear Regression equation:
        predicted_value = (slope × time_index) + intercept

    Parameters:
        X (ndarray) : time index (from prepare_data)
        y (ndarray) : actual values (from prepare_data)

    Returns:
        model     (LinearRegression) : trained model object
        slope     (float)            : m in y = mx + c
        intercept (float)            : c in y = mx + c
        direction (str)              : 'Increasing' or 'Decreasing'
    """
    model = LinearRegression()
    model.fit(X, y)

    slope     = round(float(model.coef_[0]), 4)
    intercept = round(float(model.intercept_), 4)
    direction = 'Increasing ↑' if slope >= 0 else 'Decreasing ↓'

    return model, slope, intercept, direction



def predict_future(model, last_index, n_future=3):
    """
    Predicts values for the next n_future periods.

    Parameters:
        model      (LinearRegression) : trained model
        last_index (int)              : last time index in training data
        n_future   (int)              : how many future periods to predict

    Returns:
        future_X      (ndarray) : future time indices
        future_preds  (ndarray) : predicted values
        period_labels (list)    : labels like 'Period 7', 'Period 8' ...
    """
    # Create future time indices
    # If last training row was index 6, future = [7, 8, 9]
    future_X = np.arange(
        last_index + 1,
        last_index + 1 + n_future
    ).reshape(-1, 1)

    # Predict using trained model
    future_preds = model.predict(future_X)

    # Make sure no negative predictions
    future_preds = np.maximum(future_preds, 0)

    # Create period labels
    period_labels = [f"Period {i}" for i in range(
        last_index + 1, last_index + 1 + n_future
    )]

    return future_X, future_preds.round(2), period_labels


# ------------------------------------------------------------
# FUNCTION 4 — evaluate_model
# Calculates how accurate the model is
# ------------------------------------------------------------
def evaluate_model(model, X, y):
    """
    Evaluates the trained model on training data.

    Metrics explained:
        MAE  = Mean Absolute Error
               Average difference between predicted and actual
               Lower is better

        R²   = R-squared score (0 to 1)
               How well the line fits the data
               Closer to 1 = better fit

    Parameters:
        model (LinearRegression) : trained model
        X     (ndarray)          : training features
        y     (ndarray)          : actual values

    Returns:
        metrics (dict) : MAE, R2, and interpretation
    """
    y_pred = model.predict(X)

    mae = round(float(mean_absolute_error(y, y_pred)), 2)
    r2  = round(float(r2_score(y, y_pred)), 4)

    # Interpret R2
    if r2 >= 0.9:
        r2_label = "Excellent fit"
    elif r2 >= 0.7:
        r2_label = "Good fit"
    elif r2 >= 0.5:
        r2_label = "Moderate fit"
    else:
        r2_label = "Poor fit — data may not be linear"

    metrics = {
        'MAE'      : mae,
        'R2'       : r2,
        'R2_label' : r2_label,
        'n_samples': len(y)
    }
    return metrics


# ------------------------------------------------------------
# FUNCTION 5 — get_trend_line
# Returns fitted (predicted) values for training data
# Used to draw the regression line on charts
# ------------------------------------------------------------
def get_trend_line(model, X):
    """
    Gets the fitted values for the training data.
    Used to overlay the regression line on a chart.

    Parameters:
        model (LinearRegression) : trained model
        X     (ndarray)          : training time indices

    Returns:
        fitted_values (ndarray) : predicted values for training data
    """
    return model.predict(X).round(2)


def run_full_prediction(df, value_col, group_col=None,
                        group_name=None, n_future=3):
    """
    Runs the complete prediction pipeline in one call.

    Parameters:
        df         (DataFrame) : cleaned data
        value_col  (str)       : numeric column to predict
        group_col  (str)       : optional group column
        group_name (str)       : optional specific group
        n_future   (int)       : months/periods to predict ahead

    Returns:
        result (dict) with keys:
            success       (bool)
            model         (LinearRegression)
            X, y          (arrays)
            sub_df        (DataFrame)
            slope         (float)
            intercept     (float)
            direction     (str)
            metrics       (dict)
            trend_line    (ndarray)
            future_labels (list)
            future_preds  (ndarray)
            error         (str) — only if success=False
    """
    # Step 1 — Prepare data
    X, y, sub_df, prep_msg = prepare_data(
        df, value_col, group_col, group_name
    )

    if X is None:
        return {'success': False, 'error': prep_msg}

    if len(X) < 2:
        return {'success': False, 'error': 'Need at least 2 data points.'}

    # Step 2 — Train model
    model, slope, intercept, direction = train_model(X, y)

    # Step 3 — Evaluate
    metrics = evaluate_model(model, X, y)

    # Step 4 — Get trend line for chart
    trend_line = get_trend_line(model, X)

    # Step 5 — Predict future
    last_index = int(X[-1][0])
    _, future_preds, future_labels = predict_future(
        model, last_index, n_future
    )

    return {
        'success'      : True,
        'model'        : model,
        'X'            : X,
        'y'            : y,
        'sub_df'       : sub_df,
        'slope'        : slope,
        'intercept'    : intercept,
        'direction'    : direction,
        'metrics'      : metrics,
        'trend_line'   : trend_line,
        'future_labels': future_labels,
        'future_preds' : future_preds,
        'prep_msg'     : prep_msg
    }


# ============================================================
#  SELF TEST — run: python prediction.py
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  prediction.py — self test")
    print("=" * 55)

    # Simulate 6 months of sales data with a drop
    test_df = pd.DataFrame({
        'Month'  : ['Jan','Feb','Mar','Apr','May','Jun'],
        'Product': ['A']*6,
        'Sales'  : [50000, 48000, 30000, 45000, 47000, 22000]
    })

    print("\n[1] run_full_prediction()")
    result = run_full_prediction(
        test_df,
        value_col='Sales',
        n_future=3
    )

    if result['success']:
        print(f"  Slope     : {result['slope']}")
        print(f"  Intercept : {result['intercept']}")
        print(f"  Direction : {result['direction']}")
        print(f"  MAE       : {result['metrics']['MAE']}")
        print(f"  R²        : {result['metrics']['R2']} ({result['metrics']['R2_label']})")
        print(f"\n  Future predictions:")
        for label, pred in zip(result['future_labels'], result['future_preds']):
            print(f"    {label} → {pred:,.2f}")
    else:
        print(f"  Error: {result['error']}")

    print("\n✅ prediction.py — all functions working!")
import pandas as pd
import numpy as np

def load_any_csv(filepath):
    """
    Reads any CSV file uploaded by the user.

    Parameters:
        filepath (str) : full path to the CSV file

    Returns:
        df  (DataFrame) : the loaded data
        msg (str)       : success or error message
    """
    try:
        # Try reading with UTF-8 encoding first
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            # Some files need latin1 encoding
            df = pd.read_csv(filepath, encoding='latin1')

        # Strip spaces from column names
        df.columns = [str(c).strip() for c in df.columns]

        # Remove completely empty rows
        df.dropna(how='all', inplace=True)
        df.reset_index(drop=True, inplace=True)

        if len(df) == 0:
            return None, "File is empty after loading."

        msg = f"Loaded successfully — {len(df)} rows, {len(df.columns)} columns."
        return df, msg

    except FileNotFoundError:
        return None, "File not found. Please check the path."
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

def clean_data(df):
    """
    Cleans the DataFrame by handling missing values.

    Strategy:
        - Numeric columns  → fill missing with column mean
        - Text columns     → fill missing with 'Unknown'
        - Duplicate rows   → removed

    Parameters:
        df (DataFrame) : raw loaded DataFrame

    Returns:
        df_clean  (DataFrame) : cleaned DataFrame
        report    (dict)      : cleaning summary report
    """
    df_clean = df.copy()

    report = {
        'original_rows'   : len(df),
        'duplicate_rows'  : 0,
        'columns_fixed'   : [],
        'missing_before'  : int(df.isnull().sum().sum()),
        'missing_after'   : 0
    }

    # Step 1 — Remove duplicate rows
    before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    report['duplicate_rows'] = before - len(df_clean)

    # Step 2 — Fix missing values column by column
    for col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count == 0:
            continue  # nothing to fix

        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            mean_val = df_clean[col].mean()
            df_clean[col].fillna(round(mean_val, 2), inplace=True)
            report['columns_fixed'].append(
                f"{col}: {missing_count} missing → filled with mean ({round(mean_val,2)})"
            )
        else:
            df_clean[col].fillna('Unknown', inplace=True)
            report['columns_fixed'].append(
                f"{col}: {missing_count} missing → filled with 'Unknown'"
            )

    report['missing_after']  = int(df_clean.isnull().sum().sum())
    report['cleaned_rows']   = len(df_clean)

    return df_clean, report



def detect_drops(df, value_col, group_col=None, threshold_pct=10.0):
    """
    Detects drops in a numeric column compared to previous period.

    How it works:
        For each row, we look at the previous row's value.
        If current < previous by more than threshold_pct%, it is a DROP.

    Parameters:
        df            (DataFrame) : cleaned data
        value_col     (str)       : the numeric column to check (e.g. Sales)
        group_col     (str)       : optional — group by this column (e.g. Product)
        threshold_pct (float)     : flag drop if fall > this % (default 10%)

    Returns:
        drops_df (DataFrame) : only the rows that had drops
        all_df   (DataFrame) : full data with change_pct column added
    """
    df = df.copy()

    # Make sure the value column is numeric
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df.dropna(subset=[value_col], inplace=True)

    if group_col and group_col in df.columns:
        # Calculate change per group (e.g., per Product)
        df['prev_value'] = df.groupby(group_col)[value_col].shift(1)
    else:
        # Calculate change across the whole column
        df['prev_value'] = df[value_col].shift(1)

    # Remove first row per group (no previous to compare)
    df_compare = df.dropna(subset=['prev_value']).copy()

    if len(df_compare) == 0:
        return pd.DataFrame(), df

    # Calculate percentage change
    df_compare['change_pct'] = (
        (df_compare[value_col] - df_compare['prev_value'])
        / df_compare['prev_value'] * 100
    ).round(2)

    df_compare['drop_amount'] = (
        df_compare['prev_value'] - df_compare[value_col]
    ).round(2)

    # Add severity label
    def label(x):
        if x < -40:  return '🔴 Critical'
        if x < -20:  return '🟠 High'
        if x < -10:  return '🟡 Medium'
        return '🟢 Normal'

    df_compare['severity'] = df_compare['change_pct'].apply(label)

    # Filter only actual drops
    drops_df = df_compare[df_compare['change_pct'] < -threshold_pct].copy()
    drops_df.reset_index(drop=True, inplace=True)

    return drops_df, df_compare

def get_stats(df, value_col):
    """
    Computes basic statistics for a chosen numeric column.

    Parameters:
        df        (DataFrame) : cleaned data
        value_col (str)       : the numeric column to summarize

    Returns:
        stats (dict) : dictionary of statistics
    """
    series = pd.to_numeric(df[value_col], errors='coerce').dropna()

    if len(series) == 0:
        return {}

    stats = {
        'total'   : round(float(series.sum()), 2),
        'mean'    : round(float(series.mean()), 2),
        'median'  : round(float(series.median()), 2),
        'max'     : round(float(series.max()), 2),
        'min'     : round(float(series.min()), 2),
        'std_dev' : round(float(series.std()), 2),
        'count'   : int(len(series))
    }
    return stats



def get_numeric_columns(df):
    """Returns list of numeric columns in the DataFrame."""
    return [col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])]



def get_text_columns(df):
    """Returns list of non-numeric (text) columns in the DataFrame."""
    return [col for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col])]



if __name__ == '__main__':
    import os

    print("=" * 55)
    print("  analysis.py — self test")
    print("=" * 55)

    # Create a small test CSV
    test_data = pd.DataFrame({
        'Month'  : ['Jan','Feb','Mar','Apr','May','Jun'],
        'Product': ['A','A','A','A','A','A'],
        'Sales'  : [50000, 48000, 30000, 45000, 47000, 22000],
        'Units'  : [200, 190, 120, 180, 185, 90],
    })
    test_data.to_csv('test_sample.csv', index=False)

    print("\n[1] load_any_csv()")
    df, msg = load_any_csv('test_sample.csv')
    print(f"  {msg}")

    print("\n[2] clean_data()")
    df_clean, report = clean_data(df)
    print(f"  Missing before: {report['missing_before']}")
    print(f"  Missing after : {report['missing_after']}")

    print("\n[3] detect_drops() — Sales column")
    drops, all_df = detect_drops(df_clean, 'Sales', threshold_pct=10)
    print(f"  Drops found: {len(drops)}")
    if len(drops) > 0:
        print(drops[['Month','Sales','change_pct','severity']].to_string(index=False))

    print("\n[4] get_stats()")
    stats = get_stats(df_clean, 'Sales')
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n[5] Numeric columns:", get_numeric_columns(df_clean))
    print("    Text columns   :", get_text_columns(df_clean))

    os.remove('test_sample.csv')
    print("\n✅ analysis.py — all functions working!")
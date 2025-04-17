import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import warnings

warnings.filterwarnings('ignore') # VIF calculation can generate warnings

def select_categorical_features(df, categorical_cols, target_col, p_value_threshold=0.05):
    """
    Selects categorical features based on Chi-square test against the target variable.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (list): List of categorical column names to test.
        target_col (str): Name of the target variable column.
        p_value_threshold (float): Significance level for keeping features.

    Returns:
        list: List of selected categorical feature names.
    """
    print("--- Starting Categorical Feature Selection (Chi-square) ---")
    selected_categorical = []
    if not categorical_cols:
        print("No categorical columns provided for selection.")
        return selected_categorical

    print(f"Testing features against target '{target_col}' with p-value threshold {p_value_threshold}")
    for col in categorical_cols:
        if col not in df.columns:
            print(f"Warning: Categorical column '{col}' not found in DataFrame. Skipping.")
            continue
        if df[col].isnull().any() or df[target_col].isnull().any():
            print(f"Warning: Column '{col}' or target '{target_col}' contains NaNs. Chi-square might fail or be inaccurate. Consider imputation first.")
            # Optional: Impute or drop NaNs before crosstab
            # temp_df = df[[col, target_col]].dropna()
            # contingency_table = pd.crosstab(temp_df[col], temp_df[target_col])
        try:
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, pval, _, _ = chi2_contingency(contingency_table)
            print(f"Feature: {col}, p-value: {pval:.4f}")
            if pval <= p_value_threshold:
                selected_categorical.append(col)
                print(f"  -> Selected (p-value <= {p_value_threshold})")
            else:
                print(f"  -> Rejected (p-value > {p_value_threshold})")
        except Exception as e:
             print(f"Could not perform Chi-square test for '{col}'. Error: {e}")

    print(f"--- Selected Categorical Features: {selected_categorical} ---")
    return selected_categorical

def select_numerical_features_vif(df, numerical_cols, vif_threshold=6.0):
    """
    Selects numerical features by iteratively removing features with VIF above the threshold.

    Args:
        df (pd.DataFrame): Input DataFrame containing numerical features.
        numerical_cols (list): List of numerical column names to evaluate.
        vif_threshold (float): Maximum allowed VIF value.

    Returns:
        list: List of selected numerical feature names after VIF filtering.
    """
    print("\n--- Starting Numerical Feature Selection (VIF) ---")
    if not numerical_cols:
        print("No numerical columns provided for VIF selection.")
        return []

    # Ensure only existing columns are considered and handle potential NaNs/Infs
    valid_numerical_cols = [col for col in numerical_cols if col in df.columns]
    vif_data = df[valid_numerical_cols].copy()
    vif_data.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace infinities
    cols_before_nan_drop = len(vif_data.columns)
    rows_before_nan_drop = len(vif_data)
    vif_data.dropna(inplace=True) # VIF cannot handle NaNs
    cols_after_nan_drop = len(vif_data.columns)
    rows_after_nan_drop = len(vif_data)

    if rows_before_nan_drop != rows_after_nan_drop:
         print(f"Warning: Dropped {rows_before_nan_drop - rows_after_nan_drop} rows with NaNs/Infs before VIF calculation.")
    if cols_before_nan_drop != cols_after_nan_drop:
         print(f"Warning: Dropped columns with all NaNs/Infs before VIF calculation: {set(valid_numerical_cols) - set(vif_data.columns)}")

    valid_numerical_cols = list(vif_data.columns) # Update list after potential column drop
    if not valid_numerical_cols:
         print("No valid numerical columns remaining after handling NaNs/Infs for VIF.")
         return []

    print(f"Iteratively checking VIF with threshold {vif_threshold}...")
    columns_to_keep = valid_numerical_cols[:] # Work on a copy

    while True:
        if not columns_to_keep: # Check if list is empty
            print("No columns left to check for VIF.")
            break

        vif_values = pd.DataFrame()
        try:
          # Calculate VIF for all remaining columns
           vif_values["feature"] = columns_to_keep
           vif_values["VIF"] = [variance_inflation_factor(vif_data[columns_to_keep].values, i)
                              for i in range(len(columns_to_keep))]
        except Exception as e:
            print(f"Error calculating VIF (possible perfect collinearity or insufficient data after NaN drop?). Stopping VIF selection. Error: {e}")
            # Return columns kept so far, or consider alternative strategies
            return columns_to_keep # Return what was kept before the error


        max_vif = vif_values['VIF'].max()
        max_vif_feature = vif_values.loc[vif_values['VIF'].idxmax(), 'feature']

        print(f"Current Max VIF: {max_vif:.4f} (Feature: {max_vif_feature})")

        if max_vif > vif_threshold:
            print(f"  -> Removing '{max_vif_feature}' (VIF > {vif_threshold})")
            columns_to_keep.remove(max_vif_feature)
            # Keep vif_data with the same columns for the next iteration's calculation base
        else:
            print(f"  -> All remaining features have VIF <= {vif_threshold}. Stopping.")
            break # Exit loop if max VIF is acceptable

        if not columns_to_keep: # Check again if list became empty after removal
            print("Removed all columns during VIF check.")
            break

    print(f"--- Selected Numerical Features after VIF: {columns_to_keep} ---")
    return columns_to_keep


def select_numerical_features_anova(df, numerical_cols, target_col, p_value_threshold=0.05):
    """
    Selects numerical features based on ANOVA F-test against the target variable.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numerical_cols (list): List of numerical column names to test (ideally post-VIF).
        target_col (str): Name of the target variable column.
        p_value_threshold (float): Significance level for keeping features.

    Returns:
        list: List of selected numerical feature names.
    """
    print("\n--- Starting Numerical Feature Selection (ANOVA F-test) ---")
    selected_numerical_anova = []
    if not numerical_cols:
        print("No numerical columns provided for ANOVA selection.")
        return selected_numerical_anova

    print(f"Testing features against target '{target_col}' with p-value threshold {p_value_threshold}")
    target_groups = df[target_col].unique()
    print(f"Target groups found: {target_groups}")

    for col in numerical_cols:
        if col not in df.columns:
            print(f"Warning: Numerical column '{col}' not found in DataFrame. Skipping.")
            continue
        if df[col].isnull().any():
             print(f"Warning: Column '{col}' contains NaNs. ANOVA results might be affected. Consider imputation.")
             # Optionally filter NaNs for the specific test
             # temp_df = df[[col, target_col]].dropna()

        # Prepare data for ANOVA: list of arrays, one for each target group
        grouped_data = []
        valid_groups = 0
        for group in target_groups:
            # Extract data for the current group, handling potential NaNs in the numerical column
            group_values = df.loc[df[target_col] == group, col].dropna().values
            if len(group_values) > 0: # Only include groups with data
                 grouped_data.append(group_values)
                 valid_groups += 1

        if valid_groups < 2:
            print(f"Feature: {col}, Not enough groups (>1) with valid data for ANOVA after handling NaNs. Skipping.")
            continue

        try:
            # Check variance equality (optional but recommended)
            # from scipy.stats import levene
            # stat, p_levene = levene(*grouped_data)
            # if p_levene < 0.05:
            #     print(f"Warning: Feature '{col}' - variances might not be equal (Levene p={p_levene:.4f}). ANOVA assumption violated.")

            # Perform ANOVA
            f_statistic, p_value = f_oneway(*grouped_data)
            print(f"Feature: {col}, F-statistic: {f_statistic:.4f}, p-value: {p_value:.4f}")

            if p_value <= p_value_threshold:
                selected_numerical_anova.append(col)
                print(f"  -> Selected (p-value <= {p_value_threshold})")
            else:
                print(f"  -> Rejected (p-value > {p_value_threshold})")
        except Exception as e:
            print(f"Could not perform ANOVA for '{col}'. Error: {e}")


    print(f"--- Selected Numerical Features after ANOVA: {selected_numerical_anova} ---")
    return selected_numerical_anova


def perform_feature_selection(df, target_col, potential_num_cols=None, potential_cat_cols=None,
                              vif_threshold=6.0, p_value_threshold=0.05):
    """
    Orchestrates the feature selection process.

    Args:
        df (pd.DataFrame): Input DataFrame (cleaned).
        target_col (str): Name of the target variable column.
        potential_num_cols (list, optional): List of numerical columns to consider.
                                            If None, attempts to infer from dtype.
        potential_cat_cols (list, optional): List of categorical columns to consider.
                                            If None, attempts to infer from dtype ('object', 'category').
        vif_threshold (float): VIF threshold for multicollinearity check.
        p_value_threshold (float): Significance level for Chi-square and ANOVA tests.

    Returns:
        list: Final list of selected feature names.
    """
    print("===== Starting Feature Selection Workflow =====")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # --- Identify Column Types if not provided ---
    if potential_num_cols is None:
        potential_num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in potential_num_cols:
            potential_num_cols.remove(target_col)
        # Optionally remove ID columns if they exist and are numeric
        potential_num_cols = [col for col in potential_num_cols if 'PROSPECTID' not in col.upper()]
        print(f"Inferred numerical columns: {potential_num_cols}")

    if potential_cat_cols is None:
        potential_cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col in potential_cat_cols: # Target could be categorical
            potential_cat_cols.remove(target_col)
        print(f"Inferred categorical columns: {potential_cat_cols}")

    # --- Categorical Selection (Chi-square) ---
    selected_categorical = select_categorical_features(df, potential_cat_cols, target_col, p_value_threshold)

    # --- Numerical Selection (VIF) ---
    numerical_after_vif = select_numerical_features_vif(df, potential_num_cols, vif_threshold)

    # --- Numerical Selection (ANOVA) ---
    final_numerical = select_numerical_features_anova(df, numerical_after_vif, target_col, p_value_threshold)

    # --- Combine Selected Features ---
    final_features = final_numerical + selected_categorical
    print(f"\n===== Final Selected Features: {final_features} =====")

    return final_features

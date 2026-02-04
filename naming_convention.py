import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
# Constants
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Data Loading
def load_expense_data(file_path):
    """Load expense dataset from CSV file."""
    expense_df = pd.read_csv('D:\Expense Analyzer\Budget_Analysis.csv')
    expense_df.columns = expense_df.columns.str.strip().str.lower()
    return expense_df
# Data Preprocessing
def encode_category(expense_df):
    """Encode categorical column using Label Encoding."""
    label_encoder = LabelEncoder()
    expense_df["category"] = label_encoder.fit_transform(expense_df["category"])
    return expense_df
def remove_amount_outliers(expense_df):
    """Remove outliers from amount column using IQR method."""
    q1 = expense_df["amount"].quantile(0.25)
    q3 = expense_df["amount"].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    filtered_df = expense_df[
        (expense_df["amount"] >= lower_limit) &
        (expense_df["amount"] <= upper_limit)
    ]
    return filtered_df
# Feature Engineering
def create_date_features(expense_df):
    expense_df["date"] = pd.to_datetime(expense_df["date"])
    expense_df["year"] = expense_df["date"].dt.year
    expense_df["month"] = expense_df["date"].dt.month
    expense_df["month_name"] = expense_df["date"].dt.month_name()
    expense_df["day"] = expense_df["date"].dt.day
    expense_df["day_name"] = expense_df["date"].dt.day_name()
    expense_df["weekday"] = expense_df["date"].dt.weekday

    return expense_df
def create_amount_features(expense_df):
    """Create amount-related features."""
    expense_df["amount"] = expense_df["amount"].abs()
    expense_df["log_amount"] = np.log1p(expense_df["amount"])
    expense_df["high_expense_flag"] = (
        expense_df["amount"] > expense_df["amount"].mean()
    ).astype(int)

    return expense_df
def create_monthly_features(expense_df):
    """Create monthly aggregated features."""
    expense_df["year_month"] = expense_df["date"].dt.to_period("M")

    monthly_stats_df = expense_df.groupby("year_month")["amount"].agg(
        monthly_total="sum",
        monthly_avg="mean",
        monthly_max="max"
    )
    expense_df = expense_df.merge(
        monthly_stats_df,
        on="year_month",
        how="left"
    )
    return expense_df
# Regression Model
def train_linear_regression(features, target):
    """Train Linear Regression model."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    regression_model = LinearRegression()
    regression_model.fit(x_train, y_train)
    predictions = regression_model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, rmse, mae, r2
# Classification Model
def train_random_forest_classifier(features, target):
    """Train Random Forest Classifier."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    classifier_model = RandomForestClassifier(random_state=RANDOM_STATE)
    classifier_model.fit(x_train, y_train)
    predictions = classifier_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy
# Visualization
def plot_amount_histogram(expense_df):
    plt.figure(figsize=(8, 5))
    plt.hist(expense_df["amount"], bins=30)
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.title("Expense Amount Distribution")
    plt.show()
# Main Execution
def main():
    expense_df = load_expense_data("budget_analysis.csv")
    expense_df = encode_category(expense_df)
    expense_df = remove_amount_outliers(expense_df)
    expense_df = create_date_features(expense_df)
    expense_df = create_amount_features(expense_df)
    expense_df = create_monthly_features(expense_df)
    print("Final Dataset Shape:", expense_df.shape)
    print(expense_df.head())
    regression_features = expense_df[["log_amount", "monthly_avg", "monthly_max"]]
    regression_target = expense_df["amount"]

    mse, rmse, mae, r2 = train_linear_regression(
        regression_features, regression_target
    )
    print("\nRegression Evaluation:")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    classification_features = expense_df[["log_amount", "monthly_total"]]
    classification_target = expense_df["high_expense_flag"]
    accuracy = train_random_forest_classifier(
        classification_features, classification_target
    )
    print("\nClassification Accuracy:", accuracy)
    plot_amount_histogram(expense_df)
if __name__ == "__main__":
    main()
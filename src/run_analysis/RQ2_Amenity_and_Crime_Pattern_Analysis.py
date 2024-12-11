import argparse
from typing import Tuple
import pandas as pd
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report


def merge_data_with_score(
    crime_data: pd.DataFrame, rental_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge crime data and rental data, calculate weighted crime scores,
    and aggregate amenity features.

    Args:
        crime_data (pd.DataFrame): DataFrame containing crime data
        with postal codes and categories.
        rental_data (pd.DataFrame): DataFrame containing rental data with
        postal codes and amenities.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Merged data without and with
        weighted crime scores and encoded features.
    """
    # rental_data = rental_data[rental_data['city_name'] == 'Los Angeles']
    crime_agg = crime_data.groupby("postal_code")\
        .size().reset_index(name="crime_count")
    avg_total_security = (
        rental_data.groupby("postal_code")["total_security_amenities"]
        .mean()
        .reset_index()
    )
    avg_total_security.rename(
        columns={"total_security_amenities": "avg_total_security_amenities"},
        inplace=True,
    )
    merged_data = pd.merge(
                    avg_total_security,
                    crime_agg, on="postal_code", how="left")
    merged_data["crime_count"] = merged_data["crime_count"].fillna(0)
    merged_data["high_crime"] = (
        merged_data["crime_count"] > merged_data["crime_count"].median()
    ).astype(int)
    amenity_columns = rental_data.loc[:, "security":"patrol"].columns
    amenity_count = (
        rental_data.groupby("postal_code")[amenity_columns].sum().reset_index()
    )
    merged_data = pd.merge(
                        merged_data,
                        amenity_count,
                        on="postal_code", how="left")
    # merged_data
    # Define the crime weights
    crime_weights = {
        "violent_crime": 6,
        "human_trafficking": 5,
        "sexual_offense": 5,
        "assault": 4,
        "theft": 3,
        "fraud": 3,
        "public_order": 2,
        "animal_cruelty": 2,
        "other": 1,
    }

    # Add a weighted score column based on the crime category
    crime_data["crime_weight"] = crime_data["crime_category"]\
        .map(crime_weights)

    # Calculate total weighted crime score per postal code
    weighted_crime_score = (
        crime_data.groupby("postal_code")["crime_weight"].sum().reset_index()
    )
    weighted_crime_score.rename(
        columns={"crime_weight": "weighted_crime_score"}, inplace=True
    )

    # Merge weighted crime scores with rental data
    merged_data_with_score = pd.merge(
        merged_data, weighted_crime_score, on="postal_code", how="inner"
    )
    # print(f"The length of dataframe: {len(weighted_crime_score)}")
    # print(f"The length of dataframe: {len(merged_data_with_score)}")
    # merged_data_with_score.columns

    # One-hot encode the crime_category column
    crime_category_encoded = pd.get_dummies(
        crime_data["crime_category"], prefix="crime"
    )

    # Aggregate one-hot encoded categories by postal code
    crime_category_features = crime_data[["postal_code"]]\
        .join(crime_category_encoded)
    crime_category_aggregated = (
        crime_category_features.groupby("postal_code").sum().reset_index()
    )

    # Merge the one-hot encoded features with merged_data
    merged_data_with_score = pd.merge(
        merged_data_with_score,
        crime_category_aggregated,
        on="postal_code", how="inner"
    )
    # print(f"The length of the dataframe: {len(merged_data_with_score)}")

    return merged_data, merged_data_with_score


def high_crime_area(
    merged_data: pd.DataFrame, merged_data_with_score: pd.DataFrame, model: str
) -> None:
    """
    Analyze high-crime areas and evaluate using specified model
    (Logistic Regression or Random Forest).

    Args:
        merged_data (pd.DataFrame): Merged data without
        weighted crime scores and features.
        merged_data_with_score (pd.DataFrame): Merged data with weighted
        crime scores and features.
        model (str): The model to use for analysis
        ('lr' for Logistic Regression, 'rf' for Random Forest).

    Returns:
        None
    """
    # Define high-crime areas based on the median weighted score
    merged_data_with_score["high_crime"] = (
        merged_data_with_score["weighted_crime_score"]
        > merged_data_with_score["weighted_crime_score"].median()
    ).astype(int)
    # high_crime_data_with_score = merged_data_with_score[
    #     merged_data_with_score["high_crime"] == 1
    # ]

    if model == "lr":
        # Features and target
        X = merged_data_with_score[["avg_total_security_amenities"]].fillna(
            0
        )  # Features
        y = merged_data_with_score["high_crime"]  # Target
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Logistic Regression
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)

        # Evaluation
        print(classification_report(y_test, y_pred))
        print(
                f"Coefficient for Total Security Amenities: "
                f"{log_reg.coef_[0][0]}"
            )
        print(f"Intercept: {log_reg.intercept_[0]}")
    elif model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix

        # Define the features and target
        amenity_columns = merged_data.loc[:, "security":"patrol"]\
            .columns.tolist()
        features = ["avg_total_security_amenities"] + amenity_columns
        X = merged_data_with_score[features].fillna(0)
        # Include the new features
        y = merged_data_with_score["high_crime"]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=28
        )

        # Train a Random Forest model
        rf_model = RandomForestClassifier(
            random_state=42, n_estimators=100, max_depth=10
        )
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Evaluate the model
        print(classification_report(y_test, y_pred))
        # print(confusion_matrix(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        # print(len(X_train))
        # print(len(X_test))
        import matplotlib.pyplot as plt
        import numpy as np

        # Extract feature importance
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features_sorted = np.array(features)[indices]

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(
                range(len(features_sorted)),
                importances[indices], color="#BFA1B7")
        plt.xticks(range(len(features_sorted)), features_sorted, rotation=90)
        plt.ylabel("Importance Score")
        plt.xlabel("Features")
        plt.tight_layout()
        plt.show()


def heat_map(
    crime_data: pd.DataFrame,
    rental_data: pd.DataFrame,
    merged_data_with_score: pd.DataFrame,
) -> None:
    """
    Create a heatmap to visualize the average security amenities by crime type.

    Args:
        crime_data (pd.DataFrame): DataFrame containing crime data.
        rental_data (pd.DataFrame): DataFrame containing rental data.
        merged_data_with_score (pd.DataFrame):
        Merged data with weighted crime scores.

    Returns:
        None
    """
    # Assume `crime_weights` is a dictionary of crime category weights
    crime_weights = {
        "violent_crime": 6,
        "human_trafficking": 5,
        "sexual_offense": 5,
        "assault": 4,
        "theft": 3,
        "fraud": 3,
        "public_order": 2,
        "animal_cruelty": 2,
        "other": 1,
    }

    # Add a weighted score column based on the crime category
    crime_data["weighted_score"] = crime_data["crime_category"]\
        .map(crime_weights)
    amenity_columns = rental_data.loc[:, "security":"patrol"].columns

    # Group by postal code and crime category,
    # and calculate total weighted scores
    weighted_crime = (
        crime_data.groupby(["postal_code", "crime_category"])["weighted_score"]
        .sum()
        .reset_index()
    )

    weighted_crime = pd.merge(
        weighted_crime, merged_data_with_score, on="postal_code", how="left"
    )

    # Group crimes by type and calculate the average number
    # of amenities for each crime type
    crime_amenities = (
        weighted_crime
        .groupby("crime_category")[amenity_columns]
        .mean()
    )

    # Rank amenities for each crime category
    # crime_amenities_ranked = crime_amenities.rank(axis=1, ascending=False)

    # Output top amenities for each crime category
    for crime_type in crime_amenities.index:
        print(f"Top amenities for {crime_type}:")
        top_amenities = (
            crime_amenities
            .loc[crime_type]
            .sort_values(ascending=False)
            .head(5)
        )
        print(top_amenities)
        print("\n")

    # Visualization: Heatmap of crime categories vs. average amenities
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    sns.heatmap(crime_amenities, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Average Security Amenities by Crime Type and Postal Code")
    plt.xlabel("Amenities")
    plt.ylabel("Crime Categories")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="argument for this code")
    parser.add_argument(
        "-f", type=str, help="Which part of the analysis do you want to run"
    )
    parser.add_argument(
        "-m",
        type=str,
        default="lr",
        help="Which model do you want to use in the analysis",
    )
    args = parser.parse_args()
    # pd.read_csv(args.F)
    print(args.f)
    rental_data = pd.read_csv(
        "data/processed/cleaned_rental_data_with_postalcode.csv"
        )
    crime_data = pd.read_csv(
        "data/processed/crime_with_postalcode_cityname.csv"
        )
    merged_data, merged_data_with_score = merge_data_with_score(
        crime_data,
        rental_data
    )
    if args.f == "high_crime_area":
        high_crime_area(merged_data, merged_data_with_score, args.m)
    elif args.f == "heat_map":
        heat_map(crime_data, rental_data, merged_data_with_score)

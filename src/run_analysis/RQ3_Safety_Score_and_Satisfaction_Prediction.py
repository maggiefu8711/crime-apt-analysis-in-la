import pandas as pd
import argparse


def merge_data(
    rental_data: pd.DataFrame,
    crime_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge rental and crime data by postal code, calculate weighted
    crime scores, and aggregate amenities.

    Args:
        rental_data (pd.DataFrame): DataFrame containing
        rental data with postal codes.
        crime_data (pd.DataFrame): DataFrame containing
        crime data with postal codes.

    Returns:
        pd.DataFrame: Merged DataFrame with aggregated
        amenities and crime scores by postal code.
    """
    import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.linear_model import LinearRegression
    # from sklearn.metrics import mean_squared_error, r2_score

    postal_avg_price = (
        rental_data.groupby("postal_code")["min_price"]
        .mean()
        .reset_index(name="postal_avg_price")
    )
    crime_agg = crime_data.groupby("postal_code")\
        .size().reset_index(name="crime_count")
    crime_agg = crime_agg[crime_agg["crime_count"] > 0]
    merged_data = pd.merge(
            postal_avg_price,
            crime_agg,
            on="postal_code",
            how="left"
        )
    merged_data["crime_count"] = merged_data["crime_count"].fillna(0)
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

    merged_data = pd.merge(
        merged_data, weighted_crime_score, on="postal_code", how="left"
    )
    merged_data["weighted_crime_score"] = merged_data["weighted_crime_score"]\
        .fillna(0)
    print(f"Total Data Rows: {len(merged_data)}")
    avg_property_rating = (
        rental_data.groupby("postal_code")["Property Rating"]
        .mean()
        .reset_index()
    )
    avg_property_rating.rename(
        columns={"property_rating": "avg_property_rating"}, inplace=True
    )
    merged_data = pd.merge(
        merged_data, avg_property_rating, on="postal_code", how="left"
    )
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
        merged_data, avg_total_security, on="postal_code", how="left"
    )
    amenity_columns = rental_data.loc[:, "security":"patrol"].columns
    amenity_aggregated = (
        rental_data.groupby("postal_code")[amenity_columns].sum().reset_index()
    )
    merged_data = pd.merge(
        merged_data, amenity_aggregated, on="postal_code", how="left"
    )

    return merged_data


def overall_postal_code(merged_data: pd.DataFrame) -> None:
    """
    Analyze the relationship between property
    ratings and average total security amenities.

    Args:
        merged_data (pd.DataFrame): Merged DataFrame with
        property ratings and security amenities.

    Returns:
        None
    """
    import pandas as pd
    # import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    # from sklearn.metrics import mean_squared_error, r2_score

    X = merged_data[["Property Rating"]]
    y = merged_data["avg_total_security_amenities"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    # Identify the indices of outliers
    outlier_indices = y[(y < lower_bound) | (y > upper_bound)].index

    # Get the removed rows
    removed_rows = X.loc[outlier_indices]
    removed_targets = y.loc[outlier_indices]

    # Print outlier information
    print(f"Outlier Indices: {outlier_indices.tolist()}")
    print(f"Removed Rows:\n{removed_rows}")
    print(f"Removed Target Values:\n{removed_targets}")

    # Remove rows where y contains outliers
    non_outliers = (y >= lower_bound) & (y <= upper_bound)
    X_no_outliers = X[non_outliers]
    y_no_outliers = y[non_outliers]
    filtered_data = pd.concat([X_no_outliers, y_no_outliers], axis=1)
    print(
            f"Number of rows after removing outliers: "
            f"{len(y)}, {len(y_no_outliers)}"
        )

    # Split the data again after outlier removal
    X_train, X_test, y_train, y_test = train_test_split(
        X_no_outliers, y_no_outliers, test_size=0.3, random_state=42
    )

    # Perform Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_score = lin_reg.score(X_test, y_test)

    # Print model scores
    print("Linear Regression R^2 Score:", lin_reg_score)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Property Rating",
        y="avg_total_security_amenities",
        data=filtered_data,
        label="Postal Code",
    )
    plt.plot(
        X["Property Rating"],
        lin_reg.predict(X),
        color="red",
        label="Linear Regression",
        linewidth=2,
    )
    plt.title("Average Total Security Amenities vs. Property Rating")
    plt.xlabel("Property Rating")
    plt.ylabel("Average Total Security Amenities")
    plt.legend()
    plt.show()


def high_crime_area(
    rental_data: pd.DataFrame, merged_data: pd.DataFrame, model: str
) -> None:
    """
    Analyze high-crime areas and their amenities using regression models.

    Args:
        rental_data (pd.DataFrame): DataFrame containing
        rental data with postal codes.
        merged_data (pd.DataFrame): Merged DataFrame with
        aggregated data by postal codes.
        model (str): The type of regression model
        ('lr' forLinear Regression, 'rf' for Random Forest Regression).

    Returns:
        None
    """
    import pandas as pd
    # import argparse
    # import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    merged_data["high_crime"] = (
        merged_data["weighted_crime_score"]
        > merged_data["weighted_crime_score"].median()
    ).astype(int)
    high_crime_data = merged_data[merged_data["high_crime"] == 1]

    if model == "lr":
        X = high_crime_data[["Property Rating"]]
        y = high_crime_data["avg_total_security_amenities"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        # Identify the indices of outliers
        outlier_indices = y[(y < lower_bound) | (y > upper_bound)].index

        # Get the removed rows
        removed_rows = X.loc[outlier_indices]
        removed_targets = y.loc[outlier_indices]

        # Print outlier information
        print(f"Outlier Indices: {outlier_indices.tolist()}")
        print(f"Removed Rows:\n{removed_rows}")
        print(f"Removed Target Values:\n{removed_targets}")

        # Remove rows where y contains outliers
        non_outliers = (y >= lower_bound) & (y <= upper_bound)
        X_no_outliers = X[non_outliers]
        y_no_outliers = y[non_outliers]
        filtered_data = pd.concat([X_no_outliers, y_no_outliers], axis=1)
        print(
                f"Number of rows after removing outliers: "
                f"{len(y)}, {len(y_no_outliers)}"
            )

        # Split the data again after outlier removal
        X_train, X_test, y_train, y_test = train_test_split(
            X_no_outliers, y_no_outliers, test_size=0.3, random_state=42
        )

        # Perform Linear Regression
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        lin_reg_score = lin_reg.score(X_test, y_test)

        # Print model scores
        print("Linear Regression R^2 Score:", lin_reg_score)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="Property Rating",
            y="avg_total_security_amenities",
            data=filtered_data,
            label="Postal Code",
        )
        plt.plot(
            X["Property Rating"],
            lin_reg.predict(X),
            color="red",
            label="Linear Regression",
            linewidth=2,
        )
        plt.title(
            "Average Total Security Amenities vs. "
            "Property Rating in High-Crime Area"
        )
        plt.xlabel("Property Rating")
        plt.ylabel("Average Total Security Amenities")
        plt.legend()
        plt.show()
    elif model == "rf":
        import pandas as pd
        # import numpy as np
        # from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        # from sklearn.inspection import permutation_importance
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_absolute_error
        from math import sqrt

        amenity_columns = rental_data.loc[:, "security":"patrol"].columns
        X = high_crime_data[
            [
                "crime_count",
                "weighted_crime_score",
                "avg_total_security_amenities",
                "postal_avg_price",
            ]
            + list(amenity_columns)
        ]
        y = high_crime_data["Property Rating"]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit a Random Forest Regressor
        model = RandomForestRegressor(random_state=4)
        model.fit(X_train, y_train)
        rf_reg_predictions = model.predict(X_test)
        # rf_reg_score = rf_reg.score(X_test, y_test)

        rf_mae = mean_absolute_error(y_test, rf_reg_predictions)
        rf_rmse = sqrt(mean_squared_error(y_test, rf_reg_predictions))
        rf_mape = (abs(y_test - rf_reg_predictions) / y_test).mean() * 100
        rf_r2 = r2_score(y_test, rf_reg_predictions)

        print("\nRandom Forest Regression Metrics:")
        print(f"MAE: {rf_mae}")
        print(f"RMSE: {rf_rmse}")
        print(f"MAPE: {rf_mape}%")
        print(f"R-squared: {rf_r2}")

        import pandas as pd
        import matplotlib.pyplot as plt

        # After training the Random Forest model
        # Feature importances from the trained model
        feature_importances = model.feature_importances_

        # Create a DataFrame to display feature names and their importance
        importance_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": feature_importances}
        ).sort_values(by="Importance", ascending=False)

        # Normalize to get percentages
        importance_df["Importance (%)"] = (
            importance_df["Importance"] / importance_df["Importance"].sum()
        ) * 100

        # Print the feature importances
        print(importance_df)

        # Plot the feature importances
        plt.figure(figsize=(12, 8))
        plt.barh(
            importance_df["Feature"],
            importance_df["Importance (%)"],
            color="orange"
        )
        plt.xlabel("Importance (%)")
        plt.ylabel("Features")
        plt.title("Feature Importances in Random Forest")
        plt.gca().invert_yaxis()
        plt.show()

        X = high_crime_data[amenity_columns]
        y = high_crime_data["Property Rating"]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2
        )

        # Fit a Random Forest Regressor
        model = RandomForestRegressor(random_state=15)
        model.fit(X_train, y_train)
        rf_reg_predictions = model.predict(X_test)
        # rf_reg_score = rf_reg.score(X_test, y_test)

        rf_mae = mean_absolute_error(y_test, rf_reg_predictions)
        rf_rmse = sqrt(mean_squared_error(y_test, rf_reg_predictions))
        rf_mape = (abs(y_test - rf_reg_predictions) / y_test).mean() * 100
        rf_r2 = r2_score(y_test, rf_reg_predictions)

        print("\nRandom Forest Regression Metrics:")
        print(f"MAE: {rf_mae}")
        print(f"RMSE: {rf_rmse}")
        print(f"MAPE: {rf_mape}%")
        print(f"R-squared: {rf_r2}")

        feature_importances = model.feature_importances_

        # Create a DataFrame to display feature names and their importance
        importance_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": feature_importances}
        ).sort_values(by="Importance", ascending=False)

        # Normalize to get percentages
        importance_df["Importance (%)"] = (
            importance_df["Importance"] / importance_df["Importance"].sum()
        ) * 100

        # Print the feature importances
        print(importance_df)

        # Plot the feature importances
        plt.figure(figsize=(12, 8))
        plt.barh(
            importance_df["Feature"],
            importance_df["Importance (%)"],
            color="green"
        )
        plt.xlabel("Importance (%)")
        plt.ylabel("Features")
        plt.title("Feature Importances in Random Forest")
        plt.gca().invert_yaxis()
        plt.show()


def amenity_among_crime_level(merged_data: pd.DataFrame) -> None:
    """
    Analyze the relationship between crime levels
    and satisfaction with security amenities.

    Args:
        merged_data (pd.DataFrame): Merged DataFrame with
        amenities and weighted crime scores.

    Returns:
        None
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Replace zeros in weighted_crime_score to avoid logarithmic errors
    merged_data["weighted_crime_score"] = merged_data["weighted_crime_score"]\
        .replace(
            0, 0.1
        )

    # Define min and max values for logarithmic scaling
    min_value = merged_data["weighted_crime_score"].min()
    max_value = merged_data["weighted_crime_score"].max()

    # Check for valid range
    if min_value <= 0 or max_value <= 0:
        raise ValueError(
            "All weighted_crime_score values mustbe"
            "positive for logarithmic scaling."
        )

    # Create logarithmic bins
    num_bins = 6
    bins = np.logspace(np.log10(min_value), np.log10(max_value), num=num_bins)

    # Ensure bins are unique and sorted
    bins = np.unique(bins)

    # Define labels (number of labels = number of bins - 1)
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    # Bin the data
    merged_data["crime_level"] = pd.cut(
        merged_data["weighted_crime_score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    # Count unique postal codes per crime level
    postal_code_counts = merged_data.groupby("crime_level")["postal_code"]\
        .nunique()
    print("Number of Postal Codes in Each Crime Level:")
    print(postal_code_counts)

    # Group by crime level and amenity presence,
    # then calculate average satisfaction
    merged_data["has_amenities"] = (
        merged_data["avg_total_security_amenities"] > 0
    )
    grouped = (
        merged_data
        .groupby(["crime_level", "has_amenities"])["Property Rating"]
        .mean()
        .unstack()
    )

    # Define custom colors for the stacked bars
    custom_colors = [
        "#FF9999",
        "#66B2FF",
    ]

    # Stacked Bar Chart with Custom Colors
    ax = grouped.plot(
                        kind="bar",
                        stacked=True,
                        figsize=(10, 6),
                        color=custom_colors
                    )

    # Add annotations for average satisfaction values
    for i, (level, row) in enumerate(grouped.iterrows()):
        for j, val in enumerate(row):
            if not pd.isna(val):
                ax.text(
                    i,
                    row.cumsum()[j] - (row[j] / 2),
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                )

    # Add title and labels
    plt.title(
        "Satisfaction Differences by Crime Level and Amenity Presence",
        fontsize=14
    )
    plt.xlabel("Crime Level", fontsize=12)
    plt.ylabel("Average Property Rating", fontsize=12)
    plt.legend(
        ["Without Security Amenities", "With Security Amenities"],
        title="Security-related Amenity Presence",
        fontsize=10,
    )
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the stacked bar chart
    plt.show()

    # Plot postal code counts for additional context
    postal_code_counts.plot(kind="bar", color="skyblue", figsize=(8, 5))
    plt.title("Number of Postal Codes in Each Crime Level", fontsize=14)
    plt.xlabel("Crime Level", fontsize=12)
    plt.ylabel("Number of Postal Codes", fontsize=12)
    plt.xticks(rotation=45)
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
    merged_data = merge_data(rental_data, crime_data)
    if args.f == "overall_postal_code":
        overall_postal_code(merged_data)
    elif args.f == "high_crime_area":
        high_crime_area(rental_data, merged_data, args.m)
    elif args.f == "amenity_among_crime_level":
        amenity_among_crime_level(merged_data)

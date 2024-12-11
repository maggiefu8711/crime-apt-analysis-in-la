import argparse
import pandas as pd


def result_group_by_cities() -> None:
    """
    Analyze the relationship between crime counts
    and average rental prices by city.

    Steps:
    1. Aggregate rental and crime data by city.
    2. Perform linear regression and random forest
    regression to evaluate the relationship.
    3. Visualize the results through scatter plots.

    Returns:
        None
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the datasets
    rental_data = pd.read_csv(
        "data/processed/cleaned_rental_data_with_postalcode.csv"
    )
    crime_data = pd.read_csv(
        "data/processed/crime_with_postalcode_cityname.csv"
    )

    # Aggregate rental data by city
    city_avg_price = (
        rental_data.groupby("city_name")["min_price"]
        .mean()
        .reset_index(name="city_avg_price")
    )

    # Aggregate crime data by city
    crime_agg = crime_data.groupby("city_name")\
        .size().reset_index(name="crime_count")
    crime_agg = crime_agg[
        crime_agg["crime_count"] > 0
    ]  # Keep cities with non-zero crimes

    # Strip whitespace from city names
    city_avg_price["city_name"] = city_avg_price["city_name"].str.strip()
    crime_agg["city_name"] = crime_agg["city_name"].str.strip()

    # Merge the datasets on 'city_name'
    merged_data_ini = pd.merge(
        city_avg_price,
        crime_agg,
        on="city_name",
        how="inner"
    )
    # Print the merged data
    print(merged_data_ini.head())

    X = merged_data_ini[["crime_count"]]  # Feature: crime count
    y = merged_data_ini["city_avg_price"]  # Target: average rental price

    # Step 2: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Step 3: Perform Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_score = lin_reg.score(X_test, y_test)

    # Step 4: Perform Random Forest Regression
    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train, y_train)
    rf_reg_score = rf_reg.score(X_test, y_test)

    # Step 5: Print Results
    print("Linear Regression R^2 Score:", lin_reg_score)
    print("Random Forest Regression R^2 Score:", rf_reg_score)

    # Step 6: Visualize Regression Results
    # Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="crime_count", y="city_avg_price", data=merged_data_ini, label="City"
    )
    plt.plot(X, lin_reg.predict(X), color="red", label="Linear Regression")
    plt.title(
        "Crime Count vs. City Average Rental Price in Los Angeles County"
    )
    plt.xlabel("Crime Count")
    plt.ylabel("City Average Rental Price")
    plt.legend()
    plt.show()

    # remove "Los Angeles", since the crime_count is too large
    merged_data = merged_data_ini.drop(13)
    # merged_data
    import matplotlib.pyplot as plt
    import seaborn as sns

    X = merged_data[["crime_count"]]  # Feature: crime count
    y = merged_data["city_avg_price"]  # Target: average rental price

    # Step 2: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Step 3: Perform Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_score = lin_reg.score(X_test, y_test)

    # Step 4: Perform Random Forest Regression
    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train, y_train)
    rf_reg_score = rf_reg.score(X_test, y_test)

    # Step 5: Print Results
    print("Linear Regression R^2 Score:", lin_reg_score)
    print("Random Forest Regression R^2 Score:", rf_reg_score)

    # Step 6: Visualize Regression Results
    # Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="crime_count",
        y="city_avg_price",
        data=merged_data,
        label="City"
    )
    plt.plot(X, lin_reg.predict(X), color="red", label="Linear Regression")
    plt.title(
        "Crime Count vs. City Average Rental Price in Los Angeles County"
    )
    plt.xlabel("Crime Count")
    plt.ylabel("City Average Rental Price")
    plt.legend()
    plt.show()


def result_group_by_la_postal(
    rental_data: pd.DataFrame, crime_data: pd.DataFrame, model: str
) -> None:
    """
    Analyze data for Los Angeles postal codes using regression models.

    Args:
        rental_data (pd.DataFrame): DataFrame containing rental data.
        crime_data (pd.DataFrame): DataFrame containing crime data.
        model (str): Model to use (
            'lr' for Linear Regression,
            'rf' for Random Forest Regression
        ).

    Steps:
    1. Aggregate rental and crime data by postal code in Los Angeles.
    2. Perform regression analysis based on the chosen model.
    3. Visualize regression results and feature importance (for Random Forest).

    Returns:
        None
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )
    from math import sqrt

    la_data = rental_data[rental_data["city_name"] == "Los Angeles"]
    la_price_grouped = (
        la_data.groupby("postal_code")["min_price"]
        .mean()
        .reset_index(name="postal_avg_price")
    )
    # la_price_grouped
    la_crime_data = crime_data[crime_data["city_name"] == "Los Angeles"]
    la_crime_grouped = (
        la_crime_data
        .groupby("postal_code")
        .size()
        .reset_index(name="crime_count")
    )
    # la_crime_grouped
    avg_property_rating = (
        la_data.groupby("postal_code")["Property Rating"].mean().reset_index()
    )
    avg_property_rating.rename(
        columns={"property_rating": "avg_property_rating"}, inplace=True
    )
    # avg_property_rating
    ttl_review_count = (
        la_data.groupby("postal_code")["Review Count"].sum().reset_index()
    )
    ttl_review_count.rename(
        columns={"review_count": "avg_review_count"},
        inplace=True
    )
    # ttl_review_count
    # Group by postal_code and calculate
    # the average of total_security_amenities
    avg_total_security = (
        la_data.groupby("postal_code")["total_security_amenities"]
        .mean()
        .reset_index()
    )
    avg_total_security.rename(
        columns={"total_security_amenities": "avg_total_security_amenities"},
        inplace=True,
    )
    # avg_total_security
    amenity_columns = la_data.loc[:, "security":"patrol"].columns
    amenity_count = la_data.groupby("postal_code")[amenity_columns]\
        .sum().reset_index()
    # amenity_count
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
    # weighted_crime_score
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
    la_merged = pd.merge(
        la_price_grouped, la_crime_grouped, on="postal_code", how="inner"
    )
    la_merged = pd.merge(
        la_merged, avg_property_rating, on="postal_code", how="inner"
    )
    la_merged = pd.merge(
        la_merged, ttl_review_count, on="postal_code", how="inner"
    )
    la_merged = pd.merge(
        la_merged, avg_total_security, on="postal_code", how="inner"
    )
    la_merged = pd.merge(
        la_merged, weighted_crime_score, on="postal_code", how="inner"
    )
    la_merged = pd.merge(
        la_merged, amenity_count, on="postal_code", how="inner"
    )
    la_merged = pd.merge(
        la_merged, crime_category_aggregated, on="postal_code", how="inner"
    )
    # la_merged

    if model == "lr":
        X = la_merged[["crime_count"]]
        y = la_merged["postal_avg_price"]

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
            "Number of rows after removing outliers: "
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
            x="crime_count",
            y="postal_avg_price",
            data=filtered_data,
            label="Postal Code",
        )
        plt.plot(
            X["crime_count"],
            lin_reg.predict(X),
            color="red",
            label="Linear Regression",
            linewidth=2,
        )
        plt.title(
            "Crime Count vs. Average Minimum Rental Price in Los Angeles City"
        )
        plt.xlabel("Crime Count")
        plt.ylabel("Average Minimum Rental Price for Each Postal Code")
        plt.legend()
        plt.show()
    elif model == "rf":
        X = la_merged[
            [
                "crime_count",
                "Property Rating",
                "Review Count",
                "avg_total_security_amenities",
                "weighted_crime_score",
            ]
            + list(crime_category_encoded)
        ]  # + amenity_columns.tolist()] # Feature: crime count
        y = la_merged["postal_avg_price"]
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
            f"Number of rows after removing outliers: {len(y)}, "
            f"{len(y_no_outliers)}"
        )

        # Split the data again after outlier removal
        X_train, X_test, y_train, y_test = train_test_split(
            X_no_outliers, y_no_outliers, test_size=0.3, random_state=42
        )
        # Perform Random Forest Regression
        rf_reg = RandomForestRegressor(random_state=2)
        rf_reg.fit(X_train, y_train)
        rf_reg_predictions = rf_reg.predict(X_test)
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

        # After training the Random Forest model
        # Feature importances from the trained model
        feature_importances = rf_reg.feature_importances_

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
            color="skyblue"
        )
        plt.xlabel("Importance (%)")
        plt.ylabel("Features")
        plt.title("Feature Importances in Random Forest")
        plt.gca().invert_yaxis()
        plt.show()


def remove_unimportant_features(
    rental_data: pd.DataFrame, crime_data: pd.DataFrame, model: str
) -> None:
    """
    Remove unimportant features and apply regression analysis.

    Args:
        rental_data (pd.DataFrame): DataFrame containing rental data.
        crime_data (pd.DataFrame): DataFrame containing crime data.
        model (str): Model to use (
            'rf' for Random Forest,
            'xg' for XGBoost,
            'cat' for CatBoost).

    Steps:
    1. Aggregate data by postal code in Los Angeles.
    2. Apply regression analysis after feature selection and outlier removal.

    Returns:
        None
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )
    from math import sqrt

    la_data = rental_data[rental_data["city_name"] == "Los Angeles"]
    la_price_grouped = (
        la_data.groupby("postal_code")["min_price"]
        .mean()
        .reset_index(name="postal_avg_price")
    )
    # la_price_grouped
    la_crime_data = crime_data[crime_data["city_name"] == "Los Angeles"]
    la_crime_grouped = (
        la_crime_data.groupby("postal_code")
        .size()
        .reset_index(name="crime_count")
    )
    # la_crime_grouped
    avg_property_rating = (
        la_data.groupby("postal_code")["Property Rating"].mean().reset_index()
    )
    avg_property_rating.rename(
        columns={"property_rating": "avg_property_rating"}, inplace=True
    )
    # avg_property_rating
    ttl_review_count = (
        la_data.groupby("postal_code")["Review Count"].sum().reset_index()
    )
    ttl_review_count.rename(columns={"review_count": "avg_review_count"},
                            inplace=True)
    # ttl_review_count
    avg_total_security = (
        la_data.groupby("postal_code")["total_security_amenities"]
        .mean()
        .reset_index()
    )
    avg_total_security.rename(
        columns={"total_security_amenities": "avg_total_security_amenities"},
        inplace=True,
    )
    # avg_total_security
    amenity_columns = la_data.loc[:, "security":"patrol"].columns
    amenity_count = la_data.groupby("postal_code")[amenity_columns]\
        .sum().reset_index()
    # amenity_count
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
    # weighted_crime_score
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
    la_merged = pd.merge(
        la_price_grouped, la_crime_grouped, on="postal_code", how="inner"
    )
    la_merged = pd.merge(
                            la_merged,
                            avg_property_rating,
                            on="postal_code",
                            how="inner"
                        )
    la_merged = pd.merge(
                            la_merged,
                            ttl_review_count, on="postal_code", how="inner")
    la_merged = pd.merge(
                            la_merged, avg_total_security,
                            on="postal_code", how="inner")
    la_merged = pd.merge(
                            la_merged, weighted_crime_score,
                            on="postal_code", how="inner")
    la_merged = pd.merge(
                            la_merged, amenity_count,
                            on="postal_code", how="inner")
    la_merged = pd.merge(
        la_merged, crime_category_aggregated, on="postal_code", how="inner"
    )
    # la_merged

    # prepare data
    X = la_merged[
        [
            "crime_count",
            "Property Rating",
            "avg_total_security_amenities",
            "weighted_crime_score",
            "crime_assault",
            "crime_public_order",
            "crime_violent_crime",
        ]
    ]  # + amenity_columns.tolist()] # Feature: crime count
    y = la_merged["postal_avg_price"]  # Target: average minimum rental price
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    # import numpy as np

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
    # filtered_data = pd.concat([X_no_outliers, y_no_outliers], axis=1)
    print(
            f"Number of rows after removing outliers: {len(y)}, "
            f"{len(y_no_outliers)}"
        )

    # Split the data again after outlier removal
    X_train, X_test, y_train, y_test = train_test_split(
        X_no_outliers, y_no_outliers, test_size=0.3, random_state=42
    )

    if model == "rf":

        # Perform Random Forest Regression
        rf_reg = RandomForestRegressor(random_state=2)
        rf_reg.fit(X_train, y_train)
        rf_reg_predictions = rf_reg.predict(X_test)
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
    elif model == "xg":
        import xgboost as xgb
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score
        )
        from math import sqrt

        # Train the XGBoost model
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)

        # Make predictions
        xgb_predictions = xgb_model.predict(X_test)

        # Evaluate the model
        xgb_mae = mean_absolute_error(y_test, xgb_predictions)
        xgb_rmse = sqrt(mean_squared_error(y_test, xgb_predictions))
        xgb_mape = (abs(y_test - xgb_predictions) / y_test).mean() * 100
        xgb_r2 = r2_score(y_test, xgb_predictions)

        print("\nXGBoost Metrics:")
        print(f"MAE: {xgb_mae}")
        print(f"RMSE: {xgb_rmse}")
        print(f"MAPE: {xgb_mape}%")
        print(f"R-squared: {xgb_r2}")

        # Fine-tune the XGBoost model
        param_grid = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        grid_search = GridSearchCV(
            estimator=xgb.XGBRegressor(random_state=42),
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=3,
            verbose=2,
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        # Best parameters and performance
        print("\nBest Parameters from GridSearchCV:")
        print(grid_search.best_params_)

        # Evaluate the tuned model
        best_xgb_model = grid_search.best_estimator_
        best_xgb_predictions = best_xgb_model.predict(X_test)

        best_xgb_mae = mean_absolute_error(y_test, best_xgb_predictions)
        best_xgb_rmse = sqrt(mean_squared_error(y_test, best_xgb_predictions))
        best_xgb_mape = (abs(y_test - best_xgb_predictions) / y_test)\
            .mean() * 100
        best_xgb_r2 = r2_score(y_test, best_xgb_predictions)

        print("\nTuned XGBoost Metrics:")
        print(f"MAE: {best_xgb_mae}")
        print(f"RMSE: {best_xgb_rmse}")
        print(f"MAPE: {best_xgb_mape}%")
        print(f"R-squared: {best_xgb_r2}")
    elif model == "cat":
        import catboost as cb
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, r2_score
        )
        from math import sqrt

        # Train the CatBoost model
        cat_model = cb.CatBoostRegressor(verbose=0, random_state=42)
        cat_model.fit(X_train, y_train)

        # Make predictions
        cat_predictions = cat_model.predict(X_test)

        # Evaluate the model
        cat_mae = mean_absolute_error(y_test, cat_predictions)
        cat_rmse = sqrt(mean_squared_error(y_test, cat_predictions))
        cat_mape = (abs(y_test - cat_predictions) / y_test).mean() * 100
        cat_r2 = r2_score(y_test, cat_predictions)

        print("\nCatBoost Metrics:")
        print(f"MAE: {cat_mae}")
        print(f"RMSE: {cat_rmse}")
        print(f"MAPE: {cat_mape}%")
        print(f"R-squared: {cat_r2}")

        # Fine-tune the CatBoost model
        param_grid = {
            "iterations": [100, 200, 300, 400],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5, 7],
            "subsample": [0.8, 1.0],
        }

        grid_search = GridSearchCV(
            estimator=cb.CatBoostRegressor(verbose=0, random_state=42),
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=3,
            verbose=2,
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        # Best parameters and performance
        print("\nBest Parameters from GridSearchCV:")
        print(grid_search.best_params_)

        # Evaluate the tuned model
        best_cat_model = grid_search.best_estimator_
        best_cat_predictions = best_cat_model.predict(X_test)

        best_cat_mae = mean_absolute_error(y_test, best_cat_predictions)
        best_cat_rmse = sqrt(mean_squared_error(y_test, best_cat_predictions))
        best_cat_mape = (abs(y_test - best_cat_predictions) / y_test)\
            .mean() * 100
        best_cat_r2 = r2_score(y_test, best_cat_predictions)

        print("\nTuned CatBoost Metrics:")
        print(f"MAE: {best_cat_mae}")
        print(f"RMSE: {best_cat_rmse}")
        print(f"MAPE: {best_cat_mape}%")
        print(f"R-squared: {best_cat_r2}")


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
    if args.f == "result_group_by_cities":
        result_group_by_cities()
    elif args.f == "result_group_by_la_postal":
        result_group_by_la_postal(rental_data, crime_data, args.m)
    elif args.f == "remove_unimportant_features":
        remove_unimportant_features(rental_data, crime_data, args.m)

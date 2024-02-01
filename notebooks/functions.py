import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timezone

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import pickle


def customer_clusters(df: pd.DataFrame, recency= 'recency', frequency= 'frequency', monetary= 'monetary', k= 3):
    """
    Input data frame with RFM-alike columns that will be used for clustering.
    Scales variables and performs a K-Means Clustering with k=3 as default.
    Returns a df with generated K-means Clusters.
    """

    X = df[[recency, frequency, monetary]]

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    #filename = "scaler.pickle" # Path with filename

    #with open(filename, "wb") as file:
    #        pickle.dump(scaler,file)

    X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)


    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled_df)
    X['cluster_kmean'] = kmeans.predict(X_scaled_df)

    return X


def customer_info_for_timeframe(df, start= '2023-01-01', end= '2023-06-30'):
    """

    Filters given dataframe on given timeframe and returns new customers_with_orders additional columns.
    Input: DataFrame, Start Date, and End Date
    Output: Customer-Centric df with additional columns

    """
    
    # first creating users with orders df:
    df['order_created_at'] = pd.to_datetime(df['order_created_at'])
    filtered_df = df[(df['order_created_at'] >= pd.Timestamp(start, tz='UTC')) & (df['order_created_at'] <= pd.Timestamp(end, tz='UTC'))]
    customer_df = filtered_df.groupby(['user_id', 'registered_on', 'age', 'gender', 'city', 'country']).agg({
        'order_created_at': ['min', 'max'],  # Min and Max for recency
        'order_id': 'nunique',                # Number of orders for frequency
        'num_of_item': 'mean',                # Average number of items of order
        'revenue': ['sum', 'mean']            # Total revenue of orders (CLV) and average revenugre
    }).reset_index()


    customer_df.columns = ['user_id', 'registration_date', 'age', 'gender', 'city',
                           'country', 'first_order_date', 'last_order_date', 
                           'total_orders', 'avg_order_items', 'total_revenue', 'avg_order_value']
    
    customer_df['order_frequency_per_year'] = customer_df['total_orders'] / (pd.to_datetime('now', utc=True) - customer_df['first_order_date']).dt.total_seconds() / (60 * 60 * 24 * 30 *12)
    customer_df['days_since_first_order'] = (pd.to_datetime('now', utc=True) - customer_df['first_order_date']).dt.total_seconds() / (60 * 60 * 24)
    customer_df['days_since_last_order'] = (pd.to_datetime('now', utc=True) - customer_df['last_order_date']).dt.total_seconds() / (60 * 60 * 24)
    customer_df['days_since_registration'] = (pd.to_datetime('now', utc=True) - customer_df['registration_date']).dt.total_seconds() / (60 * 60 * 24)
    customer_df['registration_year'] = customer_df['registration_date'].dt.year
    customer_df['registration_month'] = customer_df['registration_date'].dt.month
    customer_df['days_until_first_order'] = (customer_df['first_order_date'] - customer_df['registration_date']).dt.total_seconds() / (60 * 60 * 24)

    # as mentioned before, there are users that haven't ordered yet
    customer_df['has_ordered'] = customer_df['first_order_date'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    customer_df['user_id'] = customer_df['user_id'].astype(object)
    
    # Adding age groups:
    bins = [0, 18, 30, 40, 50, 60, 70, 120]
    labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71+']

    customer_df['age_class'] = pd.cut(customer_df['age'], bins=bins, labels=labels, right=False)
    
        
    return new_df

def error_metrics(y_train_pred: str, y_test_pred: str):
    """
    Generates an error metrics report based on predictions of the model.

    Input: y_train_pred and y_test_pred of your model
    Output: Error Metrics Report

    """
   
    # Evaluate the model
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    # Create a DataFrame with the error metrics
    error_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'R2'],
        'Train': [mae_train, mse_train, rmse_train, r2_train],
        'Test': [mae_test, mse_test, rmse_test, r2_test]})

    return error_df
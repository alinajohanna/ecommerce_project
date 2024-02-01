# Customer Segmentation and Customer Value Prediction

![Picture](https://d1eipm3vz40hy0.cloudfront.net/images/AMER/customersegmentation.jpg)

## Problem definition:

The ecommerce industry faces the challenge of understanding and optimizing customer behavior. 
By leveraging machine learning techniques, this project aims to create a customer segmentation model and predict customer value to enhance targeted marketing and improve overall business strategies.

## Project Purpose

The overarching purpose of this project is to:

- Develop a customer segmentation model based on relevant features such as purchase recency, frequency, and monetary value.
- Predict individual customer value to tailor marketing strategies and improve customer engagement.


## Project Questions

1. How can customers be effectively put into segments  based on their purchase behavior?
2. Can we predict the future value of individual customers to optimize marketing efforts?
3. Are there correlations between customer segments and other business metrics (e.g., average order value, total revenue)?


## Data Sources

The dataset used for this analysis is from a fictitious eCommerce clothin sited developed by the Looker team and hsoted on [Google BigQuery](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce?hl=de&project=prime-poetry-400408).  
It's encompassing information on customer transactions, registration dates, and other relevant features.

The raw dataset is stored in the `data/raw` folder of this repository and is based on a predefined SQL query.

(SELECT 
   u.id AS user_id, u.first_name, u.last_name, u.age, u.gender, u.state, u.street_address, u.postal_code,
   u.city, u.country, u.created_at AS registered_on, u.traffic_source AS user_traffic_source, o.order_id, o.status AS order_status, o.created_at AS order_created_at, o.num_of_item, SUM(oi.sale_price) AS revenue
FROM `bigquery-public-data.thelook_ecommerce.users` u
LEFT JOIN `bigquery-public-data.thelook_ecommerce.orders` o ON u.id = o.user_id
LEFT JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi ON u.id = oi.user_id
WHERE  u.country IN ('France', 'United Kingdom', 'Germany', 'Spain', 'Belgium', 'Poland', 'Austria', 'España')
GROUP BY 
   u.id AS user_id, u.first_name, u.last_name, u.age, u.gender, u.state, u.street_address, u.postal_code,
   u.city, u.country, u.created_at, u.traffic_source, o.order_id, o.status, o.created_at, o.num_of_item; )

The cleaned and processed data sets can be found within the `data/cleaned` folder.

The cleaned files serve as the basis for the conducted exploratory data analysis, segmentation, and value prediction.
The corresponding Jupyter notebooks are stored in the `notebooks/` folder.


## Data challenges:

- The data source is huged and needed intensive investigations to figure out which information is significant and should be queried.
- Handling missing or inconsistent data in customer records.


## Findings

An overview with the procedure and main findings can be found in the presentation in the `slides/` folder.

Initial analysis and exploration have revealed several noteworthy aspects:

1. Customer Segmentation:
   - Segmentation based on recency, frequency, and monetary value reveals distinct customer groups.
   - High-value customers exhibit different behavior compared to low-value or occasional shoppers.
   - Three customer segments have been determined:
      1. Active Customers: spend relatively higher amounts and have made multiple recent purchases
      2. Casual Customers: recently made purchases but spent an average to low amount
      3. Quiet Customers: haven't made a purchase in quite a while and have low to average spending

2. Customer Value Prediction:
   - Utilizing machine learning models, we can predict individual customer value with a reasonable degree of accuracy.
   - Predicted values can guide targeted marketing efforts and personalized customer interactions.

3. Correlations with Business Metrics:
   - Segmented customer groups show varying impacts on average order value and total revenue.
   - Understanding these correlations can inform marketing strategies for different customer segments.


## Instructions


To replicate the analysis or contribute to the project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Access the Jupyter notebooks in the `notebooks` directory to explore the analysis.

## Next Steps

Based on the preliminary findings, the project's next steps include:

- Refinement of the segmentation model to incorporate additional features.
- Further validation and tuning of the customer value prediction model.
- Integration of insights into marketing and business strategies.
- Collaboration with marketing teams to implement targeted campaigns based on customer segments.
- Continuous monitoring and evaluation of the model's performance.


Picture derived from [Zendesk Blog](https://www.zendesk.com/blog/customer-segmentation/)

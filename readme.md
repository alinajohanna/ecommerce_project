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

Most datasets used for this analysis are sourced from [OECD's Global Plastic Outlook](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce?hl=de&project=prime-poetry-400408). 
Additionally, 2010 data from [World Bank](https://ourworldindata.org/grapher/per-capita-plastic-waste-vs-gdp-per-capita) was added as this provides GDP per capita data for more context.

The raw datasets are stored in the `data/raw_data` folder of this repository.

The cleaned and joined data sets can be found within the `data/cleaned_data`folder.

The cleaned files are the basis for the conducted exploratory data analysis.
The corresponding jupyter notebooks are stored.

The dataset used for this analysis is from a fictitious eCommerce clothin sited developed by the Looker team and hsoted on [Google BigQuery](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce?hl=de&project=prime-poetry-400408)  
It's encompassing information on customer transactions, registration dates, and other relevant features.

The raw dataset is stored in the `data/raw` folder of this repository and is based on a predefined SQL query.

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

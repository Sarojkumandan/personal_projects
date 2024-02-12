# Sentiment Analysis on Amazon Cellphones Reviews

## Overview

This project focuses on sentiment analysis using the Amazon Cellphones Reviews dataset. The goal is to analyze customer sentiments expressed in reviews to gain insights into the overall satisfaction and sentiment trends related to cellphone products on Amazon.

## Dataset

The dataset used for this analysis is sourced from Amazon customer reviews for cellphones. It includes features such as customer ratings, review text, and product information. The dataset provides a rich source of information for understanding customer sentiments towards different cellphone products.

### Dataset Information:

- **Source:** [Amazon Cellphones Reviews Dataset](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones/data)
- **Format:** CSV
- **Columns:**
  - `Product Name`: model of the cellphone
  - `Brand Name`: Brand of the model 
  - `Price`: cost of that cellphone
  - `Rating`: Numerical content of the customer review
  - `Reviews`: Textual review of that cellphone 
  - `Review Votes`: votes to that review

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries:
  - Pandas
  - Numpy
  - NLTK (for natural language processing)
  - Matplotlib
  - Seaborn
  - Scikit-learn (for machine learning tasks)
  - transformers (for machine learning tasks)
  - tqdm (for machine learning tasks)


## Project Structure
- Importing Libraries
- Data Preparation:
    - Data Labelling: Creating a target column for model prediction
    - Data Cleaning: Cleaning the dataset for further analysis.
    - Data Preprocessing: Text preprocessings are implemented to convert raw reviews to cleaned review
- Data Analysis & Visualization
    - Use natural language processing techniques to analyze sentiments in customer reviews.
    - Implement machine learning models for sentiment classification.
- NLP BERT Model Implementation:
    - Visualize sentiment trends over time or across different products.
    - Create insightful visualizations to present findings.
- Fine-Tuning
    - Finding Accuracy, Classification Report and Confusion Matrix

## Results
- Include visualizations, accuracy metrics, and key findings from the sentiment analysis.

## Acknowledgments
- Credits to kaggle for providing the dataset.



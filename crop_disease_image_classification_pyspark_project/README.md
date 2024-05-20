# Crop Disease Image Classification Project

## Project Summary
The Crop Disease Image Classification Project aims to develop a robust and efficient machine learning model for classifying plant diseases from images. Leveraging big data tools like PySpark, MongoDB and Convolutional Neural Networks (CNNs), the project strives for high classification accuracy.

## File Description
**603_final_crop_disease_image_classification.ipynb**: This Notebook contains the complete code for data preprocessing, model development, training, evaluation, and analysis of the crop disease classification model.
**603_final_crop_disease_image_classification_ppt.pptx**: A PowerPoint presentation providing an overview of the project, including the problem statement, methodology, results, and conclusions.

## Project Goals
- Develop a CNN model for accurate crop disease classification using image data.
- Utilize PySpark for efficient data processing and handling.
- Provide clear instructions for running and contributing to the project.

## System Requirements
### Software:
- Python 3.x
- PySpark environment (Google Colab recommended)

### Libraries:
- **Pymongo**: For database interaction.
- **Pyspark**: For distributed data processing.
- **Pillow**: Image processing library.
- **Pandas**: Data manipulation and analysis.
- **Tensorflow**: Deep learning framework.
- **matplotlib**: Data visualization.
- **openCV**: Image processing.

## Installation
```bash
!pip install pymongo pyspark pillow pandas tensorflow matplotlib opencv-python
```

## Data
- The data used in this project is obtained from Kaggle API (https://www.kaggle.com/datasets/mexwell/crop-diseases-classification/data). It consists ofimages of various crop diseases in JPG format. 
- Preprocessing steps include resizing images to a standard size and filtering based on quality.

## Usage
### Data Collection:
- Download the dataset from Kaggle API.
- Retrieve the data using PySpark for efficient handling of large datasets.
### Data Cleaning and Preprocessing:
- Convert the data into a Spark DataFrame for structured manipulation.
- Implement data cleaning steps to ensure data integrity.
- Filter images based on size and aspect ratio for model consistency.
- Resize images to the expected input size for the CNN model.

##  Model Development:
- Define a CNN architecture suitable for crop disease classification (e.g., Sequential).
- Split the data into training, and test sets for model training and evaluation.
- Train the model using the training set.
- Evaluate model performance using accuracy.
- Save the trained model for future use (deployment).

## Visualization and Results:
- Generate predictions on unseen data (test set) using the trained model.
- Utilize learning curves to visualize the training and validation performance over epochs.
- Generate a confusion matrix to analyze the model's performance and identify potential classification errors.

## Running the Project
- Open the project in a PySpark environment (e.g., Google Colab).
- Install the required dependencies and ensure the dataset is available.
- Run each cell in the provided notebook sequentially to execute the data collection, preprocessing, model training, evaluation, and visualization steps.

##  Contributing
We welcome contributions to improve the project! Here's how you can contribute:
- Fork the project repository.
- Make changes to the code or documentation.
- Submit a pull request with a clear description of your contribution.

## Credits
- Project Members: Rami Reddy Kancharla, Ashok Sai Doredla.
- Dataset Source: Kaggle
- Libraries: TensorFlow, OpenCV, NumPy, etc.

## License
This project is licensed under the MIT License, allowing for free use and modification with proper attribution.
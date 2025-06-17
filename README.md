# Sentiment analysis of Grocery Reviews on Google Cloud with Pyspark

Summery of the project 
In this project,machine learning techniques in PySpark are used to perform sentiment analysis on reviews of grocery products.

The entire pipeline consists of: 
- Preprocessing and data cleaning.
- Feature extraction (TF-IDF).
- Random Forest is used for model training.
- Evaluation of the model (accuracy, precision, recall, and F1 score).
- Confusion matrix visualization.
- Google Cloud Platform (GCP) deployment and execution.
 
Tools and Technologies used:
- PySpark (MLlib). 
- Google Cloud Storage (GCS) : cluster to run Spark jobs.
- Pandas & Seaborn : evaluation and confusion matrix visualization.
- GitHub : Project code repository repository.

How to run the project:
 - Provide GCS with data upload the cleaned CSV data in a bucket on Google Cloud Storage 

  Out put:   
 - cleaned data saved here "gs://sunnyk/cleaned_grocery_reviews_csv/cleand.csv"
 - confusion matrix png is here "gs://sunnyk/path/to/output_csv/confusion_matrix.png"
 - Evaluation metrics printed in logs: Accuracy, Precision, Recall, F1 Score

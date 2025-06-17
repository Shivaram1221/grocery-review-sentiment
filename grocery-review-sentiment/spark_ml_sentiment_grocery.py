from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

# === Spark Session ===
spark = SparkSession.builder \
    .appName("GroceryReviewsBrandComparison_RF") \
    .getOrCreate()

# === Configuration ===
input_path = "gs://sunnyk/cleaned_grocery_reviews_csv/cleand.csv"
model_output_path = "gs://sunnyk/path/to/output_csv/rf_model"
plot_output_path = "gs://sunnyk/path/to/output_csv/confusion_matrix_rf.png"  # updated name

# === Load Data ===
print(f"Reading data from: {input_path}")
df = spark.read.csv(input_path, header=True, inferSchema=True)
df.printSchema()

# === Clean and Prepare Columns ===
df = df.withColumn("text", col("text").cast("string"))
df = df.withColumn("rating", col("rating").cast("double"))
df = df.dropna(subset=["text", "rating"])

# Create sentiment label
df = df.withColumn("sentiment_label", when(col("rating") >= 4, "positive")
                                    .when(col("rating") == 3, "neutral")
                                    .otherwise("negative"))

# === Sample Down to ~2.5GB ===
sample_fraction = 0.25
df = df.sample(False, sample_fraction, seed=42)
print(f"Sampled rows: {df.count()}")

# === Label Indexing ===
label_indexer = StringIndexer(inputCol="sentiment_label", outputCol="label").fit(df)
df = label_indexer.transform(df)

# === Text Feature Engineering ===
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")
text_pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])

# === Train/Test Split ===
(trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)

# === Manual Oversampling ===
print("Balancing classes with manual oversampling...")
class_counts = trainingData.groupBy("label").count().collect()
majority = max([row["count"] for row in class_counts])
oversampled = []

for row in class_counts:
    ratio = majority / row["count"]
    sampled = trainingData.filter(col("label") == row["label"]).sample(True, ratio, seed=42)
    oversampled.append(sampled)

trainingData = oversampled[0]
for o in oversampled[1:]:
    trainingData = trainingData.union(o)

print(f"Rebalanced training data: {trainingData.count()} rows")

# === Train Model (Random Forest) ===
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
pipeline = Pipeline(stages=[text_pipeline, rf])

print("Training Random Forest model...")
model = pipeline.fit(trainingData)

# === Predictions ===
print("Running predictions on test set...")
predictions = model.transform(testData).cache()

# === Evaluation ===
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
print(f"Accuracy: {evaluator.evaluate(predictions, {evaluator.metricName: 'accuracy'})}")
print(f"F1 Score: {evaluator.evaluate(predictions, {evaluator.metricName: 'f1'})}")
print(f"Precision: {evaluator.evaluate(predictions, {evaluator.metricName: 'weightedPrecision'})}")
print(f"Recall: {evaluator.evaluate(predictions, {evaluator.metricName: 'weightedRecall'})}")

# === Confusion Matrix ===
print("Generating confusion matrix plot...")
max_plot = 100000
if predictions.count() > max_plot:
    fraction = max_plot / predictions.count()
    pred_df = predictions.select("label", "prediction").sample(False, fraction, seed=42).toPandas()
else:
    pred_df = predictions.select("label", "prediction").toPandas()

labels = label_indexer.labels
numeric_labels = list(range(len(labels)))
cm = confusion_matrix(pred_df["label"], pred_df["prediction"], labels=numeric_labels)

cm_df = pd.DataFrame(cm, index=[f"Actual: {l}" for l in labels],
                     columns=[f"Predicted: {l}" for l in labels])

plt.figure(figsize=(9, 7))
sns.heatmap(cm_df, annot=True, fmt="g", cmap="Blues", linewidths=.5)
plt.title("Confusion Matrix - Grocery Reviews (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# === Save Plot to /tmp and Upload to GCS ===
local_path = "/tmp/confusion_matrix_rf.png"
plt.savefig(local_path)

try:
    jvm_uri = spark.sparkContext._jvm.java.net.URI
    jvm_conf = spark.sparkContext._jvm.org.apache.hadoop.conf.Configuration()
    jvm_fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem
    jvm_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path

    jvm_fs.get(jvm_uri(plot_output_path), jvm_conf).copyFromLocalFile(
        jvm_path(local_path),
        jvm_path(plot_output_path)
    )
    print(f"Plot uploaded to: {plot_output_path}")
except Exception as e:
    print(f"Error uploading plot: {e}")
    print(f"Run manually:\n  gsutil cp {local_path} {plot_output_path}")

if os.path.exists(local_path):
    os.remove(local_path)

# === Save Model ===
print(f"Saving model to: {model_output_path}")
model.write().overwrite().save(model_output_path)

spark.stop()

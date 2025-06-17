from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, when, to_date

# Start Spark session
spark = SparkSession.builder.appName("GroceryFoodCSVWrangling").getOrCreate()

# Load data from CSV (adjust path as needed)
df = spark.read.option("header", "true").csv("gs://sunnyk/path/to/output_csv/part-00000-b1fa930c-6e33-4132-88ac-2c133d361815-c000.csv")

# 1. Drop duplicate reviews
df = df.dropDuplicates(["review_id"])

# 2. Drop rows with missing essential fields
df = df.dropna(subset=["review_id", "product_id", "review_text", "star_rating"])

# 3. Clean text: trim whitespace, lowercase reviewer_name
df = df.withColumn("reviewer_name", trim(lower(col("reviewer_name"))))

# 4. Convert star_rating to integer
df = df.withColumn("star_rating", col("star_rating").cast("int"))

# 5. Create a verified_purchase flag (1 for 'Y', 0 otherwise)
df = df.withColumn("verified_flag", when(col("verified_purchase") == "Y", 1).otherwise(0))

# 6. Convert review_date to date type
df = df.withColumn("review_date", to_date(col("review_date"), "yyyy-MM-dd"))

# 7. Filter out invalid ratings
df = df.filter((col("star_rating") >= 1) & (col("star_rating") <= 5))

# 8. Select relevant columns for analysis
columns_to_keep = [
    "review_id", "product_id", "product_title", "brand",
    "reviewer_id", "reviewer_name", "review_text", "review_title",
    "star_rating", "helpful_votes", "total_votes",
    "verified_flag", "review_date"
]
df_clean = df.select(columns_to_keep)

# 9. Show a sample of cleaned data
df_clean.show(5)

# 10. Save cleaned data as CSV (overwrite existing files)
df_clean.coalesce(1).write.mode("overwrite").option("header", "true").csv("gs://sunnyk/path/to/output_csv/part-00000-b1fa930c-6e33-4132-88ac-2c133d361815-c000.csv")

spark.stop()

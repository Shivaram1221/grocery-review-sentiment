from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, when, to_date, from_unixtime

spark = SparkSession.builder.appName("GroceryFoodCSVWrangling").getOrCreate()

# Load data from CSV
df = spark.read.option("header", "true").csv("gs://sunnyk/path/to/output_csv/Grocery_and_Gourmet_Food.csv")

# 1. Drop duplicate reviews based on user_id and asin (since review_id does not exist)
df = df.dropDuplicates(["user_id", "asin"])

# 2. Drop rows with missing essential fields
df = df.dropna(subset=["user_id", "asin", "text", "rating"])

# 3. Clean text: trim whitespace, lowercase title
df = df.withColumn("title", trim(lower(col("title"))))

# 4. Convert rating to integer
df = df.withColumn("rating", col("rating").cast("int"))

# 5. Create a verified_purchase flag (1 for 'Y', 0 otherwise)
df = df.withColumn("verified_flag", when(col("verified_purchase") == "Y", 1).otherwise(0))

# 6. Convert timestamp (assuming it's in Unix time) to date
df = df.withColumn("review_date", from_unixtime(col("timestamp")).cast("date"))

# 7. Filter out invalid ratings
df = df.filter((col("rating") >= 1) & (col("rating") <= 5))

# 8. Select relevant columns for analysis
columns_to_keep = [
    "asin", "parent_asin", "user_id", "title", "text",
    "rating", "helpful_vote", "verified_flag", "review_date"
]
df_clean = df.select(*columns_to_keep)

# 9. Show a sample of cleaned data
df_clean.show(5)

# 10. Save cleaned data as CSV (overwrite existing files)
df_clean.coalesce(1).write.mode("overwrite").option("header", "true").csv("gs://sunnyk/cleaned_grocery_reviews_csv")

spark.stop()


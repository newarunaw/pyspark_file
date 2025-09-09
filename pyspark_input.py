from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# 1. Create Spark session
spark = SparkSession.builder \
    .appName("AI_Pipeline_Scale") \
    .getOrCreate()

# 2. Load dataset (example: local CSV)
# Replace with S3, HDFS, or database source when scaling
input_path = "data/input.csv"
data = spark.read.csv(input_path, header=True, inferSchema=True)

# 3. Basic preprocessing
# Drop rows with null values
data = data.na.drop()

# Example: Assume dataset has ["feature1", "feature2", "label"]

# Convert categorical labels to numeric
indexer = StringIndexer(inputCol="label", outputCol="label_index")

# Assemble features
assembler = VectorAssembler(
    inputCols=["feature1", "feature2"],
    outputCol="features_raw"
)

# Scale features
scaler = StandardScaler(inputCol="features_raw", outputCol="features")

# 4. Model training
lr = LogisticRegression(featuresCol="features", labelCol="label_index")

# 5. Pipeline definition
pipeline = Pipeline(stages=[indexer, assembler, scaler, lr])

# 6. Train model
model = pipeline.fit(data)

# 7. Save model output
output_path = "models/logistic_model"
model.write().overwrite().save(output_path)

print(f"Model saved to {output_path}")

# Stop Spark session
spark.stop()

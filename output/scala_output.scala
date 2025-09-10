import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, StandardScaler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline

object AIMLPipelineScale {
  def main(args: Array[String]): Unit = {
    
    // 1. Create Spark session
    val spark = SparkSession.builder
      .appName("AI_Pipeline_Scale")
      .getOrCreate()
    
    import spark.implicits._
    
    // 2. Load dataset (example: local CSV)
    // Replace with S3, HDFS, or database source when scaling
    val inputPath = "data/input.csv"
    var data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath)
    
    // 3. Basic preprocessing
    // Drop rows with null values
    data = data.na.drop()
    
    // Example: Assume dataset has ["feature1", "feature2", "label"]
    
    // Convert categorical labels to numeric
    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("label_index")
    
    // Assemble features
    val assembler = new VectorAssembler()
      .setInputCols(Array("feature1", "feature2"))
      .setOutputCol("features_raw")
    
    // Scale features
    val scaler = new StandardScaler()
      .setInputCol("features_raw")
      .setOutputCol("features")
    
    // 4. Model training
    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label_index")
    
    // 5. Pipeline definition
    val pipeline = new Pipeline()
      .setStages(Array(indexer, assembler, scaler, lr))
    
    // 6. Train model
    val model = pipeline.fit(data)
    
    // 7. Save model output
    val outputPath = "models/logistic_model"
    model.write.overwrite().save(outputPath)
    
    println(s"Model saved to $outputPath")
    
    // Stop Spark session
    spark.stop()
  }
}
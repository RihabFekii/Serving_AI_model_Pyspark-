import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler

spark = SparkSession.builder.appName("Steel faults classification with PySpark").getOrCreate()  

df=spark.read.csv(path='./Data/steel_faults.csv',header=True, inferSchema=True)

#df.printSchema()

#faults=df.groupBy("Target").count().show()  

#labels=['Stains','Z_Scratch','Other_Faults','Bumps','K_Scatch','Dirtiness','Pastry']

from pyspark.ml.feature import VectorAssembler

# Pre-process the data
assembler = VectorAssembler(inputCols=['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter',
                                  'Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity',
                                  'Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness',
                                  'Edges_Index','Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index',
                                  'Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index',
                                  'Orientation_Index','Luminosity_Index','SigmoidOfAreas'], 
                            outputCol="raw_features")
vector_df = assembler.transform(df)

#vector_df.select("raw_features").show()

from pyspark.ml.feature import StandardScaler

# Scale features to have zero mean and unit standard deviation
standarizer = StandardScaler(withMean=True, withStd=True,
                              inputCol='raw_features',
                              outputCol='features')
model = standarizer.fit(vector_df)
vector_df = model.transform(vector_df)

#vector_df.select("features").show()


from pyspark.ml.feature import StringIndexer

# Convert categorical label to number
indexer = StringIndexer(inputCol="Target", outputCol="label")
indexed = indexer.fit(vector_df).transform(vector_df)

#indexed.select("label").show()

# Select features and labels dataset to inject to the model
data = indexed.select(['features', 'label'])

# train test split 
train, test = data.randomSplit([0.7, 0.3])

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
model = rf.fit(train)

rfpredictions = model.transform(test)

# Select example rows to display.
rfpredictions.select("prediction", "label").show(5)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

f1= MulticlassClassificationEvaluator(labelCol ='label',predictionCol = "prediction",metricName="f1")
print('f1:', f1.evaluate(rfpredictions))

#saving the trained model
modelDir= "./Trained_model" 
model.write().overwrite().save(modelDir)

from pyspark.ml.classification import RandomForestClassificationModel    
rff = RandomForestClassificationModel.load(modelDir)

tes = rff.transform(test)

tes.select("prediction", "label").show(5)
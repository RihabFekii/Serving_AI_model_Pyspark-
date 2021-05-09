import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler


sp = SparkSession.builder.appName("test").getOrCreate() 

df=sp.read.csv(path='./Data/steel_faults.csv',header=True, inferSchema=True)

X = df.schema

print(X)

p=[270900, 44, 108, 24220, 0.2415, -0.2913, 42, 0.5822, 80, 1.0, 1, 0.4706, 0.8182, 1.0, 1.6435, 0.0047, 0.1818, 0, 50, 2.4265, 270944, 267, 76, 1687, 0.0498, 0.9031, 17]
cols=['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter',
                                  'Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity',
                                  'Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness',
                                  'Edges_Index','Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index',
                                  'Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index',
                                  'Orientation_Index','Luminosity_Index','SigmoidOfAreas']
DF = sp.createDataFrame(data=[p], schema = cols) 

#DF.select('X_Minimum','X_Maximum','Y_Minimum','Y_Maximum').show()

#DF.printSchema()

assembler = VectorAssembler(inputCols=['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter',
                                  'Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity',
                                  'Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness',
                                  'Edges_Index','Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index',
                                  'Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index',
                                  'Orientation_Index','Luminosity_Index','SigmoidOfAreas'], 
                            outputCol="assembled_features")
vector_df = assembler.transform(DF)


standarizer = StandardScaler(withMean=True, withStd=True,
                                    inputCol="assembled_features",
                                    outputCol="features")
model = standarizer.fit(vector_df)
feat = model.transform(vector_df)

feat.printSchema()




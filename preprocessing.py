import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import StructType,StructField, IntegerType, DoubleType

Schema= StructType([StructField("X_Minimum",IntegerType(),True),StructField("X_Maximum",IntegerType(),True),
StructField("Y_Minimum",IntegerType(),True),StructField("Y_Maximum",IntegerType(),True),
StructField("Pixels_Areas",IntegerType(),True),StructField("X_Perimeter",IntegerType(),True),
StructField("Y_Perimeter",IntegerType(),True),StructField("Sum_of_Luminosity",IntegerType(),True),
StructField("Minimum_of_Luminosity",IntegerType(),True),StructField("Maximum_of_Luminosity",IntegerType(),True),
StructField("Length_of_Conveyer",IntegerType(),True),StructField("TypeOfSteel_A300",IntegerType(),True),
StructField("TypeOfSteel_A400",IntegerType(),True),StructField("Steel_Plate_Thickness",IntegerType(),True),
StructField("Edges_Index",DoubleType(),True),StructField("Empty_Index",DoubleType(),True),
StructField("Square_Index",DoubleType(),True),StructField("Outside_X_Index",DoubleType(),True),
StructField("Edges_X_Index",DoubleType(),True),StructField("Edges_Y_Index",DoubleType(),True),
StructField("Outside_Global_Index",DoubleType(),True),StructField("LogOfAreas",DoubleType(),True),
StructField("Log_X_Index",DoubleType(),True),StructField("Log_Y_Index",DoubleType(),True),
StructField("Orientation_Index",DoubleType(),True),StructField("Luminosity_Index",DoubleType(),True),
StructField("SigmoidOfAreas",DoubleType(),True)])


spark = SparkSession.builder.appName("Steel faults classification with PySpark").getOrCreate()  


df=spark.read.csv(path='./Data/steel_faults.csv',header=True, inferSchema=True)

#df.printSchema()
print('++++++++++++++')

p=[270900, 44, 108, 24220,2415, 2913, 42, 5822, 80, 1, 1, 4706, 8182, 1, 1.6435, 0.0047, 0.1818, 0.6, 0.50, 2.4265, 2.70944, 2.67, 7.6, 1.687, 0.0498, 0.9031, 1.7]
cols=['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter',
                                  'Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity',
                                  'Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness',
                                  'Edges_Index','Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index',
                                  'Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index',
                                  'Orientation_Index','Luminosity_Index','SigmoidOfAreas']
DF = spark.createDataFrame(data=[p], schema = Schema) 

#DF.printSchema()

assembler = VectorAssembler(inputCols=['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter',
                                'Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity',
                                'Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness',
                                'Edges_Index','Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index',
                                'Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index',
                                'Orientation_Index','Luminosity_Index','SigmoidOfAreas'], outputCol="assembled_features")
vector_df = assembler.transform(DF)
#vector_df.printSchema()
#vector_df.getOutputCol().show()
h = vector_df.select("assembled_features")

vector_df.select("assembled_features").write.save("./Data/feature.csv")

print(type(h))

#standarizer = StandardScaler(withMean=True, withStd=True,inputCol="assembled_features",outputCol="features")
#model = standarizer.fit(h) 
#feat = model.transform(h)
#feat.printSchema()


def preprocess(df):

    assembler = VectorAssembler(inputCols=['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter',
                                'Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity',
                                'Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness',
                                'Edges_Index','Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index',
                                'Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index',
                                'Orientation_Index','Luminosity_Index','SigmoidOfAreas'], 
                        outputCol="assembled_features")
    vector_df = assembler.transform(df)
    
    # Scale features to have zero mean and unit standard deviation
    standarizer = StandardScaler(withMean=True, withStd=True,
                                inputCol="assembled_features",
                                outputCol="features")
    model = standarizer.fit(vector_df)
    feat = model.transform(vector_df)
    feat.printSchema()
    return feat


#preprocess(DF)




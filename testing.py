import json
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler

sc = SparkSession.builder.appName("Steel faults prediction").getOrCreate() 

class SteelFaultPredictor():
    # constructor 
    def __init__(self):
        self.parsed = []
        self.preprocessed = []
        self.prediction = None
    
    # class methods 

    def parse_json(self, input):
        features=['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter',
                                  'Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity',
                                  'Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness',
                                  'Edges_Index','Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index',
                                  'Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index',
                                  'Orientation_Index','Luminosity_Index','SigmoidOfAreas']
        
        for i in input.keys(): 
            if i in features: 
                x = input.get(i).get('value')
                print(x)
                self.parsed.append(x)
        print(self.parsed)

    def create_spark_dataframe(self):
        #p=[270900, 44, 108, 24220, 0.2415, -0.2913, 42, 0.5822, 80, 1.0, 1, 0.4706, 0.8182, 1.0, 1.6435, 0.0047, 0.1818, 0, 50, 2.4265, 270944, 267, 76, 1687, 0.0498, 0.9031, 17]
        cols=['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter',
                                  'Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity',
                                  'Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness',
                                  'Edges_Index','Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index',
                                  'Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index',
                                  'Orientation_Index','Luminosity_Index','SigmoidOfAreas']
        DF = sc.createDataFrame(data=[self.parsed], schema = cols)
        return DF 


    def preprocess(self,df):

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

        
                

            
if __name__ == '__main__':

    with open('./Data/ngsi-ld-data.json') as json_file:
        data = json.load(json_file)  

    pred = SteelFaultPredictor()
    x = pred.parse_json(data)
    DF=pred.create_spark_dataframe()
    print(type(DF))
    DF.printSchema()
    pred.preprocess(DF)
    print("done!")

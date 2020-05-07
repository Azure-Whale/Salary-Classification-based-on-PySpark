# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

from pyspark.ml.feature import VectorAssembler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# COMMAND ----------

# Build a spark context
hc = (SparkSession.builder
                  .appName('Toxic Comment Classification')
                  .enableHiveSupport()
                  .config("spark.executor.memory", "4G")
                  .config("spark.driver.memory","18G")
                  .config("spark.executor.cores","7")
                  .config("spark.python.worker.memory","4G")
                  .config("spark.driver.maxResultSize","0")
                  .config("spark.sql.crossJoin.enabled", "true")
                  .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                  .config("spark.default.parallelism","2")
                  .getOrCreate())

# COMMAND ----------

hc.sparkContext.setLogLevel('INFO')

# COMMAND ----------

hc.version

# COMMAND ----------

columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship",
          "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

train_data = pd.read_csv('/dbfs/FileStore/tables/adult.data', names=columns, 
             sep=' *, *', na_values='?')
test_data  = pd.read_csv('/dbfs/FileStore/tables/adult.test', names=columns, 
             sep=' *, *', skiprows=1, na_values='?')

le = preprocessing.LabelEncoder()

# COMMAND ----------

def preprocessing(train_data,test_data):
  '''Missing Value Process'''
  fill_mode = lambda col: col.fillna(col.mode())
  for column in train_data.columns:
    train_data[column].fillna(train_data[column].mode()[0], inplace=True)
  for column in test_data.columns:
    test_data[column].fillna(test_data[column].mode()[0], inplace=True)
  '''Columns Filtering'''
  print(train_data.info(),test_data.info())
  num_attributes = train_data.select_dtypes(include=['int64'])
  num_attributes = num_attributes.columns
  cat_attributes = train_data.select_dtypes(include=['object'])
  cat_attributes = cat_attributes.columns
  '''Simple Modfication'''
  test_data['income'] = test_data['income'].str.replace('.','')
  '''LabelEncoder Processed'''
  scaler = StandardScaler()
  total_data = pd.concat([train_data,test_data])
  for col in cat_attributes:
    le.fit(total_data[col])
    total_data[col] = le.transform(total_data[col])
  '''Data Normalization'''
  scaler.fit(total_data[num_attributes])
  total_data[num_attributes] = scaler.transform(total_data[num_attributes])
  '''Data Split'''
  train_data = total_data[:len(train_data)]
  test_data = total_data[len(train_data):]
  '''Test'''
  #print(num_attributes)
  #print(total_data[num_attributes])
  print(train_data)
  print(test_data)
  return train_data,test_data

# COMMAND ----------

train_data,test_data = preprocessing(train_data,test_data)

# COMMAND ----------

def to_spark_df(fin):
    """
    Parse a filepath to a spark dataframe using the pandas api.
    
    Parameters
    ----------
    fin : str
        The path to the file on the local filesystem that contains the csv data.
        
    Returns
    -------
    df : pyspark.sql.dataframe.DataFrame
        A spark DataFrame containing the parsed csv data.
    """
    df = hc.createDataFrame(fin)
    return(df)

# Load the train-test sets
train_data = to_spark_df(train_data)
test_data = to_spark_df(test_data)

# COMMAND ----------

print(train_data.show(15))

# COMMAND ----------

print(test_data.show(15))

# COMMAND ----------

test_data.printSchema()

# COMMAND ----------

#Assemble Jobs
cols=train_data.columns
cols.remove("income")
# Let us import the vector assembler
assembler = VectorAssembler(inputCols=cols,outputCol="features")
# Now let us use the transform method to transform our dataset
train_data=assembler.transform(train_data)
test_data=assembler.transform(test_data)

# COMMAND ----------

train_data.select("features").show(truncate=False)
test_data.select("features").show(truncate=False)

# COMMAND ----------

lr = LogisticRegression(featuresCol = "features",labelCol = 'income')
lrModel = lr.fit(train_data)


# COMMAND ----------

trainingSummary = lrModel.summary
accuracy = trainingSummary.accuracy

# COMMAND ----------

print(accuracy)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
res_test = lrModel.transform(test_data)
#print(res_test.select('prediction').show(15))
evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='income')
accuracy = evaluator.evaluate(res_test)

# COMMAND ----------

print(accuracy)

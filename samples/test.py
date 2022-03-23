from notspark.sql import SparkSession
import notspark.sql.functions as F
import pandas as pd

### Read and write using Spark API

spark = SparkSession.builder.getOrCreate()

path = "samples/data/starwars.csv.gz"
df = spark.read.format("csv").load(path)

df.limit(5)
df.head(5)
df.tail(5)

output_path = "samples/data/test.csv"
df.write.mode("overwrite").format("csv").save(output_path)

### read/write using pandas functions

spark = SparkSession.builder.getOrCreate()

path = "samples/data/starwars.csv.gz"
df = spark.createDataFrame(pd.read_csv(path))

df.head(5)

output_path = "samples/data/test.csv"
df.toPandas().to_csv(output_path)

### Function tests

spark = SparkSession.builder.getOrCreate()

path = "samples/data/starwars.csv.gz"
df = spark.createDataFrame(pd.read_csv(path))

df.head(5)

exp = F.col("mass") + 1
exp.eval(df.toPandas())

df.toPandas()["mass"].max()

(F.sqrt(F.col("mass")) + 1).eval(df.toPandas())

import math
math.atan2(172,77)
math.atan2(167, 75)

math.atan2(172,77)
1.1498780382155047
math.atan2(167, 75)
1.1486895985303731

(F.atan2(F.col("height"), F.col("mass"))).eval(df.toPandas())
(F.pow(F.col("height"), F.col("mass"))).eval(df.toPandas())

math.pow(172,77)

(F.corr(F.col("height"), F.col("mass"))).eval(df.toPandas())

pdf = df.toPandas()
from notspark.expressions.statistics_ext import correlation
correlation(pdf["height"], pdf["mass"])

import numpy as np
np.corrcoef(pdf["height"].to_numpy(), pdf["mass"].to_numpy())
np.correlate(pdf["height"], pdf["mass"], mode="valid")

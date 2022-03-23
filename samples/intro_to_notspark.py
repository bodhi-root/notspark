# File: intro_to_pplyr.py
#
# A copy of my 'intro_to_pplyr.py' code -- which itself a copy of
# the R dplyr vignette:
#
#   https://cran.r-project.org/web/packages/dplyr/vignettes/dplyr.html
#
# This is an attempt to show that we can reproduce the core features of
# common data manipulation frameworks.

from notspark.sql import SparkSession
import notspark.sql.functions as F
import pandas as pd
import numpy as np

### Load the Star Wars data ###################################################

spark = SparkSession.builder.getOrCreate()

path = "samples/data/starwars.csv.gz"
starwars = spark.read.format("csv").load(path)
starwars


starwars.columns
starwars.dtypes


### Choose rows using their position with slice() #############################

starwars.slice(4, 9)
starwars.limit(3)
starwars.head(3)
starwars.tail(3)


### Filter rows with "filter()" ###############################################

starwars.filter((F.col("skin_color") == "light") &
                (F.col("eye_color") == "brown"))


### Sort rows with "sort()" ###################################################

starwars.sort(F.col("height"), F.col("mass"))
starwars.sort("height", ascending=False)
starwars.sort(["mass", "height"])


### Select columns by name ####################################################

starwars.select("hair_color", "skin_color", "eye_color")
starwars.select(F.col("hair_color").alias("abc"), F.col("skin_color"))
starwars.select(F.col("mass") + F.col("height"))
starwars.select((F.col("mass") + F.col("height")).alias("result"))
starwars.select("*")
starwars.select(F.col("*"), F.col("mass"))

starwars.drop("hair_color", "skin_color", "eye_color")
starwars.drop(F.col("hair_color"), F.col("skin_color"))


### Rename columns ############################################################

starwars.withColumnRenamed("homeworld", "home_world")
starwars.withColumnsRenamed("homeworld", "home_world",
                            "mass", "weight")


### Add new columns with mutate() #############################################

starwars.withColumn("height_m", F.col("height") / 100)

starwars.withColumns({
  'height_m': (F.col("height") / 100),
  'BMI': (F.col("mass") / F.pow(F.col("height") / 100, 2)),
  'mass2': F.pow("mass", 2)
})

starwars.select(
  (F.col("height") / 100).alias("height_m"),
  (F.col("mass") / F.pow(F.col("height") / 100, 2)).alias("BMI"),
  (F.pow("mass", 2))
)


### Summarise values with summarise() #########################################

starwars.select(F.avg(F.col("height")))
starwars.select(F.avg(F.col("height")).alias("avg_height"))

(starwars
    .groupby("sex")
    .agg(
      F.count("*").alias("n"),
      F.mean("height").alias("height")
    )
)


### Groups ####################################################################

dfg = starwars.groupby("sex")

(dfg
  .applyInPandas(
    lambda df: pd.DataFrame({
      'sex': [df['sex'][0]],
      'length': [len(df)]
    })
  )
)

(dfg
 .applyInPandas(
    lambda df: pd.DataFrame({
        'sex': [df['sex'][0]],
        'rows': [len(df)],
        'cols': [len(df.columns)]
    })
 )
)

(dfg
 .applyInSpark(
    lambda df: df.select(
       F.count("*").alias("rows"),
       F.mean("height").alias("avg_height")
    ),
    keep_keys=True
 )
)

dfg.count()

dfg.applyInSpark(
    lambda df: df.sort(F.col("height").desc()).head(2)
).select("name", "sex", "height")

# TODO: rank
#window = Window.partitionBy("sex").orderBy(F.col("height").desc_nulls_last())
#df = df.withColumn("rank", F.rank().over(window)) \
#       .withColumn("dense_rank", F.dense_rank().over(window))

### Other tests ###############################################################

starwars.filter(~F.isnan("height")) \
        .sort("height", ascending=False) \
        .slice(1, 5)

(starwars
    .groupby("sex")
    .agg(
      F.count("*").alias("n"),
      F.avg("height").alias("avg_height"),
      F.max("height").alias("max_height")
    )
)

starwars.groupby("species").count()
starwars.groupby("sex", "gender").count()

#CONSIDER: should we provide a function like cut() to bin data?
#          I use this a lot...

(starwars
 .withColumn('bmi', F.col('mass') / F.pow(F.col('height') / 100, 2))
 .withColumn('bmi_cat', F.when(F.col('mass') < 18.5, '[0-18.5)')
                         .when(F.col('mass') < 25, '[18.5-25)')
                         .when(F.col('mass') < 30, '[25-30)')
                         .otherwise('[30,inf)'))
 .groupby("bmi_cat")
 .count()
 #.agg(F.count('*').alias('n_rows'))
 )

(starwars
 .groupby("species")
 .applyInSpark(
    lambda df: df.filter(F.col("height") == F.max("height"))
  )
 .select("name", "species", "height")
 .sort("species")
 )

### Joins ######################################################################

band_members = spark.createDataFrame(pd.DataFrame({
    "name": ["Mick", "John", "Paul"],
    "band": ["Stones", "Beatles", "Beatles"]
}))

band_instruments = spark.createDataFrame(pd.DataFrame({
    "name": ["John", "Paul", "Keith"],
    "plays": ["guitar", "bass", "guitar"]
}))

set(band_members.columns).intersection(set(band_instruments.columns))

band_members.join(band_instruments)
band_members.join(band_instruments, on=[F.col("name")])
band_members.join(band_instruments, on=["name"])
band_members.join(band_instruments, on="name")

band_members.join(band_instruments, on="name", how="left")
band_members.join(band_instruments, on="name", how="right")
band_members.join(band_instruments, on="name", how="outer")
band_members.join(band_instruments, how="cross") # FIXME

band_members.join(band_instruments, on="name", how="left_semi")
band_members.join(band_instruments, on="name", how="left_anti")

#TODO: support joins where columns aren't the same name.
#      not really sure how to do that though...
#band_instruments = band_instruments.withColumnRenamed("name", "first_name")
#band_members.join(band_instruments, band_members.name == band_instruments.first_name, how="left")
#band_members.join(band_instruments.alias("b"), F.col("name") == F.col("b.first_name"), how="left")

### Functions #################################################################

### isnan() and isnull() ###

spark.createDataFrame([
  (1, float('nan')),
  (None, 1.0)  # NOTE: pandas converts None to np.nan for float column
], ("a", "b"))

spark.createDataFrame(pd.DataFrame([
  (1, float('nan')),
  (None, 1.0)   # NOTE: pandas converts None to np.nan
], columns = ["a", "b"]))

df = spark.createDataFrame(pd.concat([
    pd.Series([1, None], name='a', dtype=object), # NOTE: pandas keeps None as-is if we specify dtype=object
    pd.Series([float('nan'), 1.0], name='b')
], axis=1))

df = df.withColumn("isnan(a)", F.isnan("a")) \
       .withColumn("isnull(a)", F.isnull("a")) \
       .withColumn("isnan(b)", F.isnan("b")) \
       .withColumn("isnull(b)", F.isnull("b"))

df

### F.when() ###

#x = pd.Series(np.arange(0,6))
#pd.Series(np.select([x < 3, x>3], [x, x**2], pd.Series(np.arange(10,16))))

df = spark.createDataFrame(pd.DataFrame({'x': pd.Series(np.arange(0,6))}))
df = df.withColumn("y", F.when(df.x < 3, F.col('x')))
df = df.withColumn("y", F.when(df.x < 3, F.col('x')).when(df.x > 3, df['x'] ** 2))
df = df.withColumn("y", F.when(df.x < 3, F.col('x')).otherwise(df['x'] ** 2))

x = F.when(df.x < 3, F.col('x'))
x.otherwise(2)

### isin() ###

starwars.filter(F.col("species").isin(["Human", "Droid"]))
starwars.filter(F.col("species").isin("Human", "Droid"))

### Corr and Cov ###

starwars.corr("mass", "height")
starwars.cov("mass", "height")

F.corr("mass", "height").eval(starwars.toPandas())
F.covar_samp("mass", "height").eval(starwars.toPandas())

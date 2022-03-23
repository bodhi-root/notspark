# "Not" Spark (notspark)

"Not Spark" (or "notspark") provides a spark-like API for DataFrame operations without
having to use Spark.  The hope is that this will make life easier
for developers who work in PySpark and in normal (non-Spark) Python
by allowing them to learn one API fully rather than have to switch
between different contexts (such as PySpark and pandas) and remember
the details of each framework.  Behind the scenes we are of course
just using pandas for DataFrame operations, but the notspark API should
hide the inner workings of pandas from the user so that they need
only be minimally aware of or adept at using the pandas package.

## Design

The project is designed to mimic Spark as closely as possible.
With the exception of "pyspark" being replaced by "notspark", the
package names, classes, and functions should all be exactly
the same.  The parameter names should be the same (although we may
not have implemented all functions or parameters yet).  Even the
code layout is intended to mimic PySpark's as closely as possible.
For reference, the PySpark source code in GitHub is available here:

* https://github.com/apache/spark/tree/master/python/pyspark

For developers, it is helpful to reference the documented behavior
of both PySpark and pandas.  Links to documentation for each 
framework's DataFrame object are below:

* https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.html
* https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html


## Loading and Saving Data

Since we follow Spark's API, you can load data just as you would in 
Spark with something like:

```
from ntspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.format("csv").load("path/to/file.csv")
```

Since pandas already has such a great set of data loading functions,
implementing all of Spark's DataReader methods has not been our first
priority.  This means the Spark API is not currently the best way
to load data. Rather, you will get much more power and flexibility 
by using pandas data reading methods and converting the resulting 
pandas DataFrame to a notspark DataFrame.  This can be done as shown 
below:

```
from notspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame(pd.read_csv(path))
```

Similarly, you can save data using either the Spark API but currently
the pandas API is a much better option.  This means you can save
data easily by converting it to a pandas DataFrame and then saving it.
This can be one as shown:

```
df.toPandas().to_csv(...)
```

Since pandas is used behind-the-scenes in notspark the ```toPandas()```
function is simply returning the wrapped pandas DataFrame that notspark
was already using.  Unlike in Spark environments, there is no overhead
in this function or any risk of your data not fitting in memory.  It
is all already in memory.


## Functions, Expressions, and Column Objects

At the core of Spark's API are DataFrames and functions.  The 
```pyspark.sql.functions``` module allows us to do such great things
as:

```
import notspark.sql.functions as F

df = df.withColumn("z", F.col("x") + F.col("y") * 2)
```

Implementing these in native Python is actually pretty tricky.
Investigation of the PySpark code reveals that PySpark is delegating
all of these calls to wrapped Java objects interfacing with Spark's
core "Catalyst" evaluation engine (written in Scala).  Even simple
functions like addition and subtraction of columns or multiplying
columns by scalars is actually just invoking functions on these wrapped
Java objects that return new wrapped Java objects.  This means we
have to take a look at the Scala code if we want to see what is 
actually happen when these functions and expressions are built
and executed.

Within the context of DataFrames, the most important object is the
Column.  ```F.col()``` returns a Column object, but so does
```F.lit()```.  And the result of adding a Column object to another
Column object, or multiplying a Column by a scalar, is of course 
another Column object.  Columns in Spark's Scala implementation
are extensions of an Expression column which provides an
```eval(InternalRow)``` method to evaluate the expression in the context
of a row of data.  This reveals that Spark uses a row-by-row
execution pattern as it performs operations such as manipulating
or filtering DataFrames.  In pandas, we want to vectorize as many
operations as possible and would suffer very poor performance if
we tried to use Spark's row-by-row processing model.  In our solution,
Columns are still the fundamental objects in expressions, but executing
expressions such as ```F.col("x") + F.col("y")``` will internally 
perform a vectorized addition using pandas and numpy rather than 
a row-by-row operation.

For reference, here are links to Spark's source code for the
Column object and for PySparks functions implementation in GitHub:

* https://github.com/apache/spark/blob/master/sql/core/src/main/scala/org/apache/spark/sql/Column.scala
* https://github.com/apache/spark/blob/master/python/pyspark/sql/functions.py

This has been very helpful while building this package.

### Expressions package

As a stepping-stone to building the Column-based expression language
used by notspark, I also wrote a similar package that allows expressions
to be defined and evaluated later without any knowledge of or 
reference to pandas.  This package still uses the Spark Functions
API, allowing you to define functions such as:

```
import notspark.expressions.functions as F

exp = F.var("x") + F.var("y") * 2
```

These expressions are evaluated within the context of an object
responsible for providing values for named variables.  As an example,
we can use the DefaultContext object to manage these variables
in a dict.  This lets us evaluate the expression above with:

```
from notspark.expressions.core import DefaultContext

ctx = DefaultContext({"x": 1, "y": 2})
exp.eval(ctx)
```

In this package, the fundamental class is called Expression.
The two simplest expressions are Literal (for literal values)
and Variable (to reference a named variable in the context).
The base Expression class then overloads as many operators as
possible so that adding, multiplying, or comparing expressions
to other expressions produces new expressions that can be 
chained together.  If an attempt is made to add something other
than an expression to an expression, it will be interpreted as
a literal value.  This is how the expression ```F.var("y") * 2```
evaluates to an expression.  Behind-the-scenes, ```2``` is 
converted to a Literal and then passed to a BinaryExpression that
multiplies two terms together.  This can then be combined with
other expressions or operated upon by functions.  The functions
defined in ```notspark.expressions.functions``` have the same
appearance as Spark functions, but they instead invoke common
Python functions such as 'math.sqrt' or 'avg' to whatever value
is given to them.  Using the same Context from earlier, the
expression ```F.sqrt(F.var("y"))``` would obtain the value of ```2```
for the variable ```y``` and evaluate the python function ```sqrt(2)```.
It will not attempt to apply ```sqrt()``` to each element in a list
(or Column) as Spark and notspark would.  Originally, it was thought
that this sub-package could be used in the development of notspark,
but this turned out not to be the case.  When notspark invokes
```F.avg()``` on a column we can't dispatch this to python's built-in
average function.  It wouldn't know what to do with our columnar
data.  Instead, we have to define our own implementation of average
(using pandas or numpy) and return the result as Column.  Similarly,
```F.sqrt()``` will need to be applied to every element in our Column
and again return a Column as a result.  For this reason, the Column-based
expressions in ```notspark.sql``` do not use any of the expression
objects in ```notspark.expression```.  However, since this package
still might be useful to someone, somewhere, we have kept it around.

## Notes

In order to avoid pandas and numpy annoying index, let's try
to enforce a rule: every index should be zero-based after
every operation.

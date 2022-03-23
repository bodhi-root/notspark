from .column import \
    Column, Literal, Variable, \
    RowCount, RowNumber, WhenExpression, \
    UnaryExpression, BinaryExpression, \
    to_col

import math
import statistics
from notspark.expressions.statistics_ext import \
    covariance as _covariance, \
    correlation as _correlation

import pandas as pd
import numpy as np


def lit(value):
    """Returns a literal value"""
    return Literal(value)


def col(name):
    """Returns a reference to a named variable"""
    return Variable(name)


def _papply(func, x, y):
    """Applies a binary function to x and y.  This will call
    func(x, y) pairwise for each element in x and y.  x
    and y can be pandas.Series objects or other values.

    NOTE: This is intended to operate on Columns after they
    have been evaluated (not as part of the expression)
    """
    if isinstance(x, pd.Series) or isinstance(y, pd.Series):
        if not isinstance(x, pd.Series):
            x = pd.Series(np.repeat(x, len(y)))
        elif not isinstance(y, pd.Series):
            y = pd.Series(np.repeat(y, len(x)))

        df = pd.concat([x, y], axis=1)
        return df.apply(lambda row: func(row[0], row[1]), axis=1)
    else:
        return func(x, y)


def _unary(x, func):
    """Applies a unary function to x.  This will calculate func(x) when
    the value is evaluated."""
    return UnaryExpression(func, to_col(x))


def _binary(x, y, func):
    """Applies a binary function to x and y.  This will call func(x,y) for
    when evaluated.  x and y are expected to be pd.Series values or scalars.

    NOTE: This function operates on Columns and produces a Column result
    for evaluation later (in contrast to _papply which evaluates the result
    """
    return BinaryExpression(func, to_col(x), to_col(y))


def _binaryapply(x, y, func):
    """Applies a binary function to x and y.  This will call func(x,y) for
    each element in x and y, returning the results as a pd.Series.  This
    is in contract to _binary() which only calls f(x,y) once with two series
    objects."""
    return BinaryExpression(lambda x,y: _papply(func, x, y), to_col(x), to_col(y))


def asc(col):
    """Returns a SortOrder to sort the column ascending.
    'col' may be a Column or the name of a Column."""
    return to_col(col).asc()

def desc(col):
    """Returns a SortOrder to sort the column descending.
    'col' may be a Column or the name of a Column."""
    return to_col(col).desc()

def asc_nulls_first(col):
    """Returns a SortOrder to sort the column ascending nulls first.
    'col' may be a Column or the name of a Column."""
    return to_col(col).asc_nulls_first()

def asc_nulls_last(col):
    """Returns a SortOrder to sort the column ascending, nulls last.
    'col' may be a Column or the name of a Column."""
    return to_col(col).asc_nulls_last()

def desc_nulls_first(col):
    """Returns a SortOrder to sort the column descending, nulls first.
    'col' may be a Column or the name of a Column."""
    return to_col(col).desc_nulls_first()

def desc_nulls_last(col):
    """Returns a SortOrder to sort the column descending, nulls first.
    'col' may be a Column or the name of a Column."""
    return to_col(col).desc_nulls_last()

def sqrt(col):
    """Square root"""
    return to_col(col).apply(math.sqrt)

def abs(col):
    """Absolute value"""
    return _unary(col, lambda x: x.abs() if isinstance(x, pd.Series) else abs(x))

def max(col):
    """Maximum"""
    return _unary(col, lambda x: x.max() if isinstance(x, pd.Series) else max(x))

def min(col):
    """Minimum"""
    return _unary(col, lambda x: x.min() if isinstance(x, pd.Series) else min(x))

#def max_by
#def min_by


def count(col):
    if (isinstance(col, str) and col == '*') or \
            (isinstance(col, Variable) and col.name == '*'):
        return RowCount()  # count including nulls
    else:
        return _unary(col,
                      lambda x: x.count() if isinstance(x, pd.Series)  # non-null count
                      else 1)  # if single element, count()=1 (don't count list elements)

def sum(col):
    """Sum"""
    return _unary(col, lambda x: x.sum() if isinstance(x, pd.Series) else sum(x))

def avg(col):
    """Average"""
    return _unary(col, lambda x: x.mean() if isinstance(x, pd.Series) else statistics.mean(x))

def mean(col):
    """Mean"""
    return _unary(col, lambda x: x.mean() if isinstance(x, pd.Series) else statistics.mean(x))

#def sumDistinct
#def sum_distinct

def product(col):
    """Product"""
    return _unary(col, lambda x: x.product() if isinstance(x, pd.Series) else math.prod(x))

def acos(col):
    """Inverse cosine"""
    return to_col(col).apply(math.acos)

def acosh(col):
    """Inverse hyberbolic cosine"""
    return to_col(col).apply(math.acosh)

def asin(col):
    """Inverse sine"""
    return to_col(col).apply(math.asin)

def asinh(col):
    """Inverse hyperbolic sine"""
    return to_col(col).apply(math.asinh)

def atan(col):
    """Tangent"""
    return to_col(col).apply(math.atan)

def atanh(col):
    """Inverse hyperbolic tangent"""
    return to_col(col).apply(math.atanh)

#def cbrt

def ceil(col):
    """Ceiling"""
    return to_col(col).apply(math.ceil)

def cos(col):
    """Cosine"""
    return to_col(col).apply(math.cos)

def cosh(col):
    """Hyperbolic cosine"""
    return to_col(col).apply(math.cosh)

def cot(col):
    """Cotangent"""
    return to_col(col).apply(math.cot)

def csc(col):
    return to_col(col).apply(lambda x: 1.0 / math.sin(x))

def exp(col):
    return to_col(col).apply(math.exp)

def expm1(col):
    return to_col(col).apply(math.expm1)

def floor(col):
    """Floor"""
    return to_col(col).apply(math.floor)

def log(col):
    """Natural logarithm"""
    return to_col(col).apply(math.log)

def log10(col):
    """Base 10 logarithm"""
    return to_col(col).apply(math.log10)

def log1p(col):
    return to_col(col).apply(math.log1p)

#def rint

def sec(col):
    """Secant"""
    return to_col(col).apply(lambda x: 1.0 / math.cos(x))

#def signum

def sin(col):
    """Sine"""
    return to_col(col).apply(math.sin)

def sinh(col):
    """Hyperbolic sine"""
    return to_col(col).apply(math.sinh)

def tan(col):
    """Tangent"""
    return to_col(col).apply(math.tan)

def tanh(col):
    """Hyperbolic tangent"""
    return to_col(col).apply(math.tanh)

#def toDegrees  # DEPRECATED
#def toRadians  # DEPRECATED
#def bitwiseNOT
#def bitwise_not

def stddev(col):
    """Standard deviation (sample)"""
    return _unary(col,
                  lambda x: x.std(ddof=1) if isinstance(x, pd.Series)
                  else statistics.stdev(x))

def stddev_samp(col):
    """Standard deviation (sample)"""
    return stddev(col)

def stdev_pop(col):
    """Standard deviation (population)"""
    return _unary(col,
                  lambda x: x.std(ddof=0) if isinstance(x, pd.Series)
                  else statistics.pstdev(x))

def variance(col):
    return _unary(col,
                  lambda x: x.var(ddof=1) if isinstance(x, pd.Series)
                  else statistics.var(x))

def var_samp(col):
    return variance(col)

def var_pop(col):
    return _unary(col,
        lambda x: x.var(ddof=0) if isinstance(x, pd.Series)
        else statistics.var(x))

#def skewness
#def kurtosis
#def collect_list
#def collect_set

def degrees(col):
    """Convert radians to degrees"""
    return to_col(col).apply(math.degrees)

def radians(col):
    """Convert degrees to radians"""
    return to_col(col).apply(math.radians)

def atan2(y, x):
    """Inverse tangent"""
    return _binaryapply(y, x, math.atan2)

def hypot(a, b):
    """Computes sqrt(a ^ 2 + b ^ 2)"""
    return _binaryapply(a, b, math.hypot)

def pow(col1, col2):
    """col1 raised to the power of col2"""
    return _binary(col1, col2,
                   lambda x, y: x.pow(y) if isinstance(x, pd.Series)
                   else _papply(math.pow, x, y))

def row_number() -> Column:
    """Returns a column with a 1-based row number"""
    return RowNumber()

#def dense_rank
#def rank
#def cume_dist
#def percent_rank
#def approxCountDistinct
#def approx_count_distinct
#def broadcast
#def coalesce

def corr(x, y):
    """Correlation"""
    return _binary(x, y,
                   lambda x, y: x.corr(y) if isinstance(x, pd.Series)
                   else _correlation(x, y))

def covar_pop(x, y):
    """Covariance (population)"""
    def _(x, y):
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            return x.cov(y) * (len(x) - 1) / len(x)
        else:
            return _covariance(x, y) * (len(x) - 1) / len(x)

    return _binary(x, y, _)

def covar_samp(x, y):
    """Covariance (sample)"""
    def _(x, y):
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            return x.cov(y)
        else:
            return _covariance(x, y)

    return _binary(x, y, _)

#def countDistinct
#def count_distinct
#def first
#def grouping
#def grouping_id
#def input_file_name

def isnan(col):
    """Returns True if a value is NaN.
    NOTE: This is different from pandas.Series.isna in that None returns False"""
    return to_col(col).apply(lambda x: False if x is None else np.isnan(x))

def isnull(col):
    """Returns True if a value is Null (or None in python)
    NOTE: Null is different from NaN.  "np.nan is None) returns False"""
    return to_col(col).apply(lambda x: x is None)

#def last
#def monotonically_increasing_id
#def nanv1
#def percentile_approx
#def rand
#def randn

def round(col, scale:int = 0):
    """Rounds the number to the given number of decimal places.
    NOTE: This uses either pandas.Series.round() or the
    built-in python 'round()' function even though
    it should be ROUND_UP"""
    return _unary(col,
                  lambda x: x.round(scale) if isinstance(x, pd.Series)
                  else round(x, scale))

def bround(col, scale:int = 0):
    """Rounds the number to the given number of decimal places.
    NOTE: This uses either pandas.Series.round() or the
    built-in python 'round()' function even though
    it should be ROUND_EVEN"""
    return _unary(col,
                  lambda x: x.round(scale) if isinstance(x, pd.Series)
                  else round(x, scale))

#def shiftLeft   # DEPRECATED
#def shiftRight  # DEPRECATED
#def shiftRightUnsigned  # DEPRECATED

def shiftleft(col, numBits:int):
    return _unary(col, lambda x: x << numBits)

def shiftright(col, numBits:int):
    return _unary(col, lambda x: x >> numBits)  # Python does not have a signed right-shift

def shiftrightunsigned(col, numBits:int):
    return _unary(col, lambda x: x >> numBits)

#def spark_partition_id
#def expr
#def struct
#def greatest
#def least

def when(condition: Column, value) -> Column:
    exp = WhenExpression()
    exp.when(condition, value)
    return exp

def log2(col):
    """Base-2 logarithm"""
    return to_col(col).apply(math.log2)

#def conv

def factorial(col):
    return to_col(col).apply(math.factorial)

def lag(col, count=1, default=None):
    def _(x):
        if isinstance(x, pd.Series):
            return x.shift(count, fill_value=default)
        else:
            raise Exception("lag() is only defined for pandas.Series objects")
    return _unary(col, _)

def lead(col, count=1, default=None):
    def _(x):
        if isinstance(x, pd.Series):
            return x.shift(-count, fill_value=default)
        else:
            raise Exception("lag() is only defined for pandas.Series objects")
    return _unary(col, _)

#def nth_value
#def ntile

# ---------------------------- date functions ----------------------------------

#def current_date
#def current_timestamp
#def date_format
#def year
#def quarter
#def month
#def dayofweek
#def dayofmonth
#def dayofyear
#def hour
#def minute
#def second
#def weekofyear
#def make_date
#def date_add
#def date_sub
#def datediff
#def add_months
#def months_between
#def to_date
#def to_timestamp
#def trunc
#def date_trunc
#def next_day
#def last_day
#def from_unixtime
#def unix_timestamp
#def from_utc_timestamp
#def to_utc_timestamp
#def timestamp_seconds
#def window
#def session_window

# ---------------------------- misc functions ----------------------------------

#def crc32
#def md5
#def sha1
#def sha2
#def hash
#def xxhash64
#def assert_true
#def raise_error

# ---------------------- String/Binary functions ------------------------------

def upper(col):
    """Convert string to upper case"""
    return _unary(col, lambda x: x.str.upper() if isinstance(x, pd.Series) else str(x).upper())

def lower(col):
    """Convert string to lower case"""
    return _unary(col, lambda x: x.str.lower() if isinstance(x, pd.Series) else str(x).lower())

#def ascii
#def base64
#def unbase64

def ltrim(col):
    """Trim whitespace from left of string"""
    return _unary(col, lambda x: x.str.ltrim() if isinstance(x, pd.Series) else str(x).lstrip())

def rtrim(col):
    """Trim whitespace from right of string"""
    return _unary(col, lambda x: x.str.rtrim() if isinstance(x, pd.Series) else str(x).rstrip())

def trim(col):
    """Trim whitespace from both ends of string"""
    return _unary(col, lambda x: x.str.trim() if isinstance(x, pd.Series) else str(x).trim())

#def concat_ws
#def decode
#def encode
#def format_number
#def format_string
#def instr
#def overlay
#def sentences

def substring(col, pos, len):
    """Return the substring beginning at position 'pos' (1-based) of length 'len'"""
    return _unary(col,
                  lambda x: x.str.slice(pos-1, pos-1+len) if isinstance(x, pd.Series)
                  else str(x)[(pos-1):(pos-1+len)])

#def substring_index

#def levenshtein
#def locate

def lpad(col, len, pad):
    """Left pad string to length 'len' using pad character 'pad'"""
    return _unary(col,
                  lambda x: x.str.rjust(len, pad) if isinstance(x, pd.Series)
                  else str(x).rjust(len, pad))

def rpad(col, len, pad):
    """Right pad string to length 'len' using pad character 'pad'"""
    return _unary(col,
                  lambda x: x.str.ljust(len, pad) if isinstance(x, pd.Series) \
                  else str(x).ljust(len, pad))

#def repeat
#def split
#def regexp_extract
#def regexp_replace
#def initcap
#def soundex
#def bin
#def hex
#def unhex

def length(col):
    """Return the length of a string"""
    return _unary(col, lambda x: x.str.len() if isinstance(x, pd.Series) else len(x))

#def octet_length
#def bit_length
#def translate

# ---------------------- Collection functions ------------------------------

#def create_map
#def map_from_arrays
#def array
#def array_contains
#def arrays_overlap
#def slice
#def array_join
#def concat
#def array_position
#def element_at
#def array_remove
#def array_distinct
#def array_intersect
#def array_union
#def array_except
#def explode
#def posexplode
#def explode_outer
#def posexplode_outer
#def get_json_objct
#def json_tuple
#def from_json
#def to_json
#def schema_of_json
#def schema_of_csv
#def to_csv
#def size
#def array_min
#def array_max
#def sort_array
#def array_sort
#def shuffle
#def reverse
#def flatten
#def map_keys
#def map_values
#def map_entries
#def map_from_entries
#def array_repeat
#def arrays_zip
#def map_concat
#def sequence
#def from_csv
#def transform
#def exists
#def forall
#def filter
#def aggregate
#def zip_with
#def transform_keys
#def transform_values
#def map_filter
#def map_zip_with

# ---------------------- Partition transform functions --------------------------------

#def years
#def months
#def days
#def hours
#def bucket
#def udf


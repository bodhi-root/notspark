from .core import \
    Literal, Variable, \
    UnaryExpression, BinaryExpression, \
    to_expression

import math
import statistics
import numpy as np

from .statistics_ext import \
    covariance as _covariance, \
    correlation as _correlation


# save references to functions that have the same name as ours
# (so we can use them in our functions)
_abs = abs
_sum = sum
_max = max
_min = min


def lit(value):
    """Returns a literal value"""
    return Literal(value)

def var(name):
    """Returns a reference to a named variable"""
    return Variable(name)


#def asc
#def desc

def _unary(x, func):
    """Returns an expression that will apply the given function
    to the value"""
    # CONSIDER: Should we extend this to iterate over iterables? That might not be desirable...
    return UnaryExpression(func, to_expression(x))

def _binary(x, y, func):
    """Applies a bivariate function to two values: func(x, y)"""
    return BinaryExpression(func, to_expression(x), to_expression(y))

def sqrt(x):
    """Square root"""
    return _unary(x, math.sqrt)

def abs(x):
    """Absolute value"""
    return _unary(x, _abs)

def max(x):
    """Maximum"""
    return _unary(x, _max)

def min(x):
    """Minimum"""
    return _unary(x, _min)

#def max_by
#def min_by
#def count

def sum(x):
    """Sum"""
    return _unary(x, _sum)

def avg(x):
    """Average"""
    return _unary(x, statistics.mean)

def mean(x):
    """Mean"""
    return _unary(x, statistics.mean)

#def sumDistinct
#def sum_distinct

def product(x):
    """Product"""
    return _unary(x, math.prod)

def acos(x):
    """Inverse cosine"""
    return _unary(x, math.acos)

def acosh(x):
    """Inverse hyberbolic cosine"""
    return _unary(x, math.acosh)

def asin(x):
    """Inverse sine"""
    return _unary(x, math.asin)

def asinh(x):
    """Inverse hyperbolic sine"""
    return _unary(x, math.asinh)

def atan(x):
    """Tangent"""
    return _unary(x, math.tan)

def atanh(x):
    """Inverse hyperbolic tangent"""
    return _unary(x, math.tanh)

#def cbrt

def ceil(x):
    """Ceiling"""
    return _unary(x, math.ceil)

def cos(x):
    """Cosine"""
    return _unary(x, math.cos)

def cosh(x):
    """Hyperbolic cosine"""
    return _unary(x, math.cosh)

def cot(x):
    """Cotangent"""
    return _unary(x, math.cot)

def csc(x):
    """Cosecant"""
    return _unary(x, lambda x: 1.0 / math.sin(x))

def exp(x):
    return _unary(x, math.exp)

def expm1(x):
    return _unary(x, math.expm1)

def floor(x):
    """Floor"""
    return _unary(x, math.floor)

def log(x):
    """Natural Logarithm"""
    return _unary(x, math.log)

def log10(x):
    """Base-10 logarithm"""
    return _unary(x, math.log10)

def log1p(x):
    """Natural log plus 1"""
    return _unary(x, math.log1p)

#def rint

def sec(x):
    return _unary(x, lambda x: 1.0 / math.cos(x))

#def signum

def sin(x):
    """Sine"""
    return _unary(x, math.sin)

def sinh(x):
    """Hyperbolic sine"""
    return _unary(x, math.sinh)

def tan(x):
    """Tangent"""
    return _unary(x, math.tan)

def tanh(x):
    """Hyperbolic tangent"""
    return _unary(math.tanh)

#def toDegrees  # DEPRECATED
#def toRadians  # DEPRECATED
#def bitwiseNOT
#def bitwise_not
#def asc_nulls_first
#def asc_nulls_last
#def desc_nulls_first
#def desc_nulls_last

def stddev(x):
    """Standard deviation (sample)"""
    return _unary(x, statistics.stdev)

def stddev_samp(x):
    """Standard deviation (sample)"""
    return _unary(x, statistics.stdev)

def stdev_pop(x):
    """Standard deviation (population)"""
    return _unary(x, statistics.pstdev)

def variance(x):
    """Variance (sample)"""
    return _unary(x, statistics.variance)

def var_samp(x):
    """Variance (sample)"""
    return _unary(x, statistics.variance)

def var_pop(x):
    """Variance (population)"""
    return _unary(x, statistics.pvariance)

#def skewness
#def kurtosis
#def collect_list
#def collect_set

def degrees(x):
    """Converts radians to degrees"""
    return _unary(x, math.degrees)

def radians(x):
    """Converts degrees to radians"""
    return _unary(x, math.radians)

def atan2(y, x):
    """Inverse tangent"""
    return _binary(y, x, math.atan2)

def hypot(a, b):
    """Computes sqrt(a ^ 2 + b ^ 2)"""
    return _binary(a, b, math.hypot)

def pow(x, y):
    """Computes x ** y"""
    return _binary(x, y, math.pow)

#def row_number
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
    return _binary(x, y, _correlation)

def covar_pop(x, y):
    """Covariance (population)"""
    return _binary(x, y, lambda x, y: _covariance(x, y) * (len(x) - 1) / len(x))

def covar_samp(x, y):
    """Covariance (sample)"""
    return _binary(x, y, _covariance)

#def countDistinct
#def count_distinct
#def first
#def grouping
#def grouping_id
#def input_file_name

def isnan(x):
    return _unary(x, np.isnan)

def isnull(x):
    return _unary(x, lambda x: x is None)

#def last
#def monotonically_increasing_id
#def nanv1
#def percentile_approx
#def rand
#def randn

def round(x, scale:int = 0):
    """Rounds the number to the given number of decimal places.
    NOTE: This just uses the built-in python 'round()' function even though
    it should be ROUND_UP"""
    return _unary(x, lambda x: round(x, scale))

def bround(x, scale:int = 0):
    """Rounds the number to the given number of decimal places.
    NOTE: This just uses the built-in python 'round()' function even though
    it should be ROUND_EVEN"""
    return _unary(x, lambda x: round(x, scale))

#def shiftLeft   # DEPRECATED
#def shiftRight  # DEPRECATED
#def shiftRightUnsigned  # DEPRECATED

def shiftleft(x, numBits:int):
    return _unary(x, lambda x: x << numBits)

def shiftright(x, numBits:int):
    return _unary(x, lambda x: x >> numBits)  # Python does not have a signed right-shift

def shiftrightunsigned(x, numBits: int):
    return _unary(x, lambda x: x >> numBits)

#def spark_partition_id
#def expr
#def struct
#def greatest
#def least
#def when

def log2(x):
    return _unary(x, math.log2)

#def conv

def factorial(x):
    return _unary(x, math.factorial)

#def lag
#def lead
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

def upper(x):
    """Convert string to upper case"""
    return _unary(x, lambda x: str(x).upper())

def lower(x):
    """Convert string to lower case"""
    return _unary(x, lambda x: str(x).lower())

#def ascii
#def base64
#def unbase64

def ltrim(x):
    """Trim whitespace from left of string"""
    return _unary(x, lambda x: str(x).lstrip())

def rtrim(x):
    """Trim whitespace from right of string"""
    return _unary(x, lambda x: str(x).rstrip())

def trim(x):
    """Trim whitespace from both ends of string"""
    return _unary(x, lambda x: str(x).trim())

#def concat_ws
#def decode
#def encode
#def format_number
#def format_string
#def instr
#def overlay
#def sentences

def substring(x, pos, len):
    """Return the substring beginning at position 'pos' (1-based) of length 'len'"""
    return _unary(x, lambda x: str(x)[(pos-1):(pos-1+len)])

#def substring_index

#def levenshtein
#def locate

def lpad(x, len, pad):
    """Left pad string to length 'len' using pad character 'pad'"""
    return _unary(x, lambda x: str(x).rjust(len, pad))

def rpad(x, len, pad):
    """Right pad string to length 'len' using pad character 'pad'"""
    return _unary(x, lambda x: str(x).ljust(len, pad))

#def repeat
#def split
#def regexp_extract
#def regexp_replace
#def initcap
#def soundex
#def bin
#def hex
#def unhex

def length(x):
    """Return the length of a string"""
    return _unary(x, len)

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


from abc import ABC, abstractmethod
import operator
import pandas as pd
import numpy as np


def to_col(x, strings_as_cols=True) -> "Column":
    """Converts the object to a Column.  If it already is a Column, return it.
    If it is a string, return a Variable referencing the column with that name.
    Otherwise, convert to a Literal value.  This allows us to convert objects
    like the number 1 to Columns.
    """
    if x is None:
        return Literal(None)
    elif isinstance(x, Column):
        return x
    elif strings_as_cols and isinstance(x, str):
        return Variable(x)
    else:
        return Literal(x)


def to_series(x) -> pd.Series:
    """Converts the object to a pandas.Series (if it isn't one yet)"""
    if isinstance(x, pd.Series):
        return x
    else:
        return pd.Series(x)


class Column(ABC):
    """Columns are the fundamental units in our expression language.
    They are evaluated within the context of a pandas.DataFrame. The two
    basic types of columns are Literals and Variables.  Variables
    reference a column in the DataFrame whereas literals do not.
    Internally, a Column is usually represented as pandas.series.Series,
    but this may not always be the case (as with literal scalar values).
    All built-in python operators are overloaded so that expressions
    can be added, subtracted, etc. and always return another
    Expression.  Any non-expression that is used in a calculation
    will be converted to a literal value.
    """

    @abstractmethod
    def eval(self, df: pd.DataFrame):
        pass

    # NOTE: We use the default python operator implementations for
    # now.  When used with pandas.series.Series objects these
    # implement things like 1+col and 2*col the way we'd want.
    # If we ever find that's not the case we can create our
    # own custom function and use it instead of the built-in operator
    # implementations.

    def __add__(self, other):
        return BinaryExpression(operator.add, self, to_col(other))

    def __sub__(self, other):
        return BinaryExpression(operator.sub, self, to_col(other))

    def __mul__(self, other):
        return BinaryExpression(operator.mul, self, to_col(other))

    def __pow__(self, other):
        return BinaryExpression(operator.pow, self, to_col(other))

    def __truediv__(self, other):
        return BinaryExpression(operator.truediv, self, to_col(other))

    def __floordiv__(self, other):
        return BinaryExpression(operator.floordiv, self, to_col(other))

    def __mod__(self, other):
        return BinaryExpression(operator.mod, self, to_col(other))

    def __lshift__(self, other):
        return BinaryExpression(operator.lshift, self, to_col(other))

    def __rshift__(self, other):
        return BinaryExpression(operator.rshift, self, to_col(other))

    def __and__(self, other):
        return BinaryExpression(operator.and_, self, to_col(other))

    def __or__(self, other):
        return BinaryExpression(operator.or_, self, to_col(other))

    def __xor__(self, other):
        return BinaryExpression(operator.xor, self, to_col(other))

    def __invert__(self):
        return UnaryExpression(operator.invert, self)

    def __lt__(self, other):
        return BinaryExpression(operator.lt, self, to_col(other))

    def __le__(self, other):
        return BinaryExpression(operator.le, self, to_col(other))

    def __gt__(self, other):
        return BinaryExpression(operator.gt, self, to_col(other))

    def __ge__(self, other):
        return BinaryExpression(operator.ge, self, to_col(other))

    def __eq__(self, other):
        return BinaryExpression(operator.eq, self, to_col(other))

    def __ne__(self, other):
        return BinaryExpression(operator.ne, self, to_col(other))

    def apply(self, func):
        """Applies a function to each element in this Column.
        This will use x.apply(f) if the object is a pd.Series.
        Otherwise we will use a simple f(x) to support scalars
        or other custom values.
        CONSIDER: should we apply functions to other iterable objects like lists?
        WARNING: This function is not in the Spark API (but it is pretty useful)
        """
        return UnaryExpression(
                lambda x: x.apply(func) if isinstance(x, pd.Series) else func(x),
                self)

    def isin(self, *cols) -> "Column":
        """
        A boolean expression that is evaluated to true if the value of this
        expression is contained by the evaluated values of the arguments.

        Examples
        --------
        >>> df[df.name.isin("Bob", "Mike")].collect()
        [Row(age=5, name='Bob')]
        >>> df[df.age.isin([1, 2, 3])].collect()
        [Row(age=2, name='Alice')]
        """
        if len(cols) == 1 and isinstance(cols[0], (list, set)):
            cols = cols[0]

        return self.apply(lambda x: x in cols)

    # Alias

    def alias(self, alias):
        return Alias(self, alias)

    # Sort functions:

    def asc(self):
        return SortOrder(self)

    def asc_nulls_first(self):
        return SortOrder(self, nulls_first=True)

    def asc_nulls_last(self):
        return SortOrder(self, nulls_first=False)

    def desc(self):
        return SortOrder(self, ascending=False)

    def desc(self):
        return SortOrder(self, ascending=False)

    def desc_nulls_first(self):
        return SortOrder(self, ascending=False, nulls_first=True)

    def desc_nulls_last(self):
        return SortOrder(self, ascending=False, nulls_first=False)


class Literal(Column):
    """Literal value.  This value is returned when eval() is called.  No
    execution context is actually required.
    """

    def __init__(self, value):
        self.value = value

    def eval(self, df: pd.DataFrame):
        return self.value


class Variable(Column):
    """A reference to a column in a DataFrame.  This column will be
    accessed in the underlying pandas.DataFrame and returned when this
    is evaluated.
    """

    def __init__(self, name):
        self.name = name

    def eval(self, df: pd.DataFrame):
        return df[self.name]


class UnaryExpression(Column):

    def __init__(self, func, child):
        self.func = func
        self.child = child

    def eval(self, df: pd.DataFrame):
        return self.func(self.child.eval(df))


class BinaryExpression(Column):

    def __init__(self, func, left, right):
        self.func = func
        self.left = left
        self.right = right

    def eval(self, df: pd.DataFrame):
        return self.func(self.left.eval(df), self.right.eval(df))


class SortOrder(object):
    """Object used for sorting.  This is not actually a Column, but
    it wraps one.
    """
    # Scala defines this here: org/apache/spark/sql/catalyst/expressions/SortOrder.scala

    def __init__(self, col, ascending=True, nulls_first=True):
        self.col = col
        self.ascending = ascending
        self.nulls_first = nulls_first

class Alias(object):
    """Alias for a wrapped column.  This is not itself a Column and
    no operations can be performed on it."""

    def __init__(self, col, alias):
        self.col = col
        self.alias = alias


class RowCount(Column):
    """Special expression to count the number of rows in the DataFrame
    and return this as a scalar."""

    def __init__(self):
        pass

    def eval(self, df: pd.DataFrame):
        return len(df)


class RowNumber(Column):
    """Special expression to return a 1-based series containing the
    number of each row."""

    def __init__(self):
        pass

    def eval(self, df: pd.DataFrame):
        return pd.Series(range(1, len(df)+1))


class WhenExpression(Column):
    """Implementation of F.when(cond, value).when(cond2, value2).otherwise(...)"""

    def __init__(self, default=None):
        self.when_list = []
        self.default = default
        self.otherwise_called = False

    def when(self, cond, value):
        self.when_list.append((cond, value))
        return self

    def otherwise(self, value):
        if self.otherwise_called:
            raise ValueError("'otherwise()' can only be called once")
        self.default = value
        self.otherwise_called = True
        return self

    def eval(self, df: pd.DataFrame):
        if len(self.when_list) == 0:
            raise Exception("Expression must have at least one 'when()' component to be evaluated")

        cond_list = []
        choice_list = []
        for (cond, choice) in self.when_list:
            cond_list.append(to_col(cond).eval(df))
            choice_list.append(to_col(choice, strings_as_cols=False).eval(df))

        default_val = to_col(self.default, strings_as_cols=False).eval(df)

        # NOTE: np.select() is pretty resilient.  It can take series-like objects or
        #       scalars for any of its parameters.
        result = np.select(cond_list, choice_list, default_val)
        return pd.Series(result)


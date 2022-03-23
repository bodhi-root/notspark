# This is the key to making it work
# Spark Columns extend Expression.  Support for unary, binary, and other
# expressions is available.  These can then be evaluated within the context
# of a data row.  Perhaps in other contexts too?  (vectorized evaluation
# would make sense in our case).
#
# PySpark delegates all of its column operations to Scala.  Functions like
# 'add()' and 'subtract()' are implemented in Scala as expressions.
#
# This is the key to things like:
#
#  F.col("a") + F.col("b") == F.col("c")
#
# It is parsed to an expression tree that can then be evaluated later
# against a row (or in our case also a DataFrame).
#
# Even outside of spark this type of expression language could be useful.
# I think I've seen other python libraries try to implement this.  It's
# key to doing some things in an R-like syntax.

from abc import ABC, abstractmethod
import operator


class Context(ABC):
    """A Context that provides values for named variables."""

    @abstractmethod
    def get(self, var):
        pass


class DefaultContext(Context):
    """Default Context that stores variables in a dict."""

    def __init__(self, vars={}):
        self.vars = vars

    def get(self, var):
        return self.vars[var]

    def set(self, var, value):
        self.vars[var] = value

    def unset(self, var):
        del self.vars[var]

    def has(self, var):
        return var in self.vars


def to_expression(x):
    if isinstance(x, Expression):
        return x
    return Literal(x)


class Expression(ABC):
    """An expression is an object that can be evaluated using eval()
    and a Context for named variables.  All built-in python operators
    are overloaded so that expressions can be added, subtracted, etc.
    and always return another Expression.  Any non-expression that
    is used in a calculation will be converted to a literal value.
    """

    @abstractmethod
    def eval(self, ctx: Context):
        pass

    def __add__(self, other):
        return BinaryExpression(operator.add, self, to_expression(other))

    def __sub__(self, other):
        return BinaryExpression(operator.sub, self, to_expression(other))

    def __mul__(self, other):
        return BinaryExpression(operator.mul, self, to_expression(other))

    def __pow__(self, other):
        return BinaryExpression(operator.pow, self, to_expression(other))

    def __truediv__(self, other):
        return BinaryExpression(operator.truediv, self, to_expression(other))

    def __floordiv__(self, other):
        return BinaryExpression(operator.floordiv, self, to_expression(other))

    def __mod__(self, other):
        return BinaryExpression(operator.mod, self, to_expression(other))

    def __lshift__(self, other):
        return BinaryExpression(operator.lshift, self, to_expression(other))

    def __rshift__(self, other):
        return BinaryExpression(operator.rshift, self, to_expression(other))

    def __and__(self, other):
        return BinaryExpression(operator.and_, self, to_expression(other))

    def __or__(self, other):
        return BinaryExpression(operator.or_, self, to_expression(other))

    def __xor__(self, other):
        return BinaryExpression(operator.xor, self, to_expression(other))

    def __invert__(self):
        return UnaryExpression(operator.invert, self)

    def __lt__(self, other):
        return BinaryExpression(operator.lt, self, to_expression(other))

    def __le__(self, other):
        return BinaryExpression(operator.le, self, to_expression(other))

    def __gt__(self, other):
        return BinaryExpression(operator.gt, self, to_expression(other))

    def __ge__(self, other):
        return BinaryExpression(operator.ge, self, to_expression(other))

    def __eq__(self, other):
        return BinaryExpression(operator.eq, self, to_expression(other))

    def __ne__(self, other):
        return BinaryExpression(operator.ne, self, to_expression(other))


class Literal(Expression):

    def __init__(self, value):
        self.value = value

    def eval(self, ctx: Context):
        return self.value


class Variable(Expression):

    def __init__(self, name):
        self.name = name

    def eval(self, ctx: Context):
        return ctx.get(self.name)


class UnaryExpression(Expression):

    def __init__(self, func, child):
        self.func = func
        self.child = child

    def eval(self, ctx: Context):
        return self.func(self.child.eval(ctx))


class BinaryExpression(Expression):

    def __init__(self, func, left, right):
        self.func = func
        self.left = left
        self.right = right

    def eval(self, ctx: Context):
        return self.func(self.left.eval(ctx), self.right.eval(ctx))

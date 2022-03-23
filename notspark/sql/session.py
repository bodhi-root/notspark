from .readwriter import DataFrameReader
from .dataframe import DataFrame
import pandas as pd


class SparkSession(object):

    class Builder:

        def getOrCreate(self) -> "SparkSession.Builder":
            # CONSIDER: add thread locking?
            session = SparkSession._instantiatedSession
            if session is None:
                session = SparkSession()
            return session

    builder = Builder()
    _instantiatedSession = None

    def __init__(self):
        if SparkSession._instantiatedSession is None:
            SparkSession._instantiatedSession = self

    #CONSIDER:
    #def range(self):

    def createDataFrame(self, data, schema=None, samplingRatio=None, verifySchema=True):
        """Creates a DataFrame object.  'data' can be a pandas.DataFrame or
        a list or lists where each inner list represents a row.  If a list is
        provided you can use schema to specify the column names (as a list of strings)
        """
        if samplingRatio is not None:
            raise ValueError("'samplingRatio' not yet supported")
        if not verifySchema:
            raise ValueError("'verifySchema' does not have any effect")

        if isinstance(data, pd.DataFrame):
            return DataFrame(data)

        elif isinstance(data, list) or isinstance(data, tuple):
            if schema is None:
                return DataFrame(pd.DataFrame(data))
            else:
                if isinstance(schema, list) or isinstance(schema, tuple):
                    return DataFrame(pd.DataFrame(data, columns=schema))
                else:
                    raise ValueError("'schema' only supports an array of column names")

        else:
            raise ValueError("Unsupported type for parameter 'data'")

    @property
    def read(self) -> DataFrameReader:
        """
        Returns a :class:`DataFrameReader`
        """
        return DataFrameReader(self)




from .dataframe import DataFrame
from .functions import count

import pandas as pd


class GroupedData(object):
    """Represents a grouped DataFrame."""

    def __init__(self, gpdf: "DataFrameGroupBy"):
        self._gpdf = gpdf

    def agg(self, *exprs):
        pdf = self._gpdf.apply(
            lambda df: DataFrame(df).select(*exprs).toPandas()
        )

        # drop the index that was automatically added for each DataFrame
        pdf.reset_index(
            level=len(pdf.index.names) - 1,
            drop=True,
            inplace=True)

        pdf.reset_index(drop=False, inplace=True)

        return DataFrame(pdf)

    #def apply(self, udf):
    #    """Alias for applyInPandas() that takes a udf rather than
    #    a native Python function
    #    """
    #    pass

    def applyInPandas(self, func, schema=None):

        #NOTE: we can not use DataFrameGroupBy.apply()
        #      because it automatically adds the group by keys as columns
        #      in the result.  Spark does not do this.  It instead expects
        #      the function to return the group by keys if it wants them.
        #      This generates an error if you try to add a column that already
        #      exists.
        #pdf = self._gpdf.apply(lambda df: func(df.reset_index(drop=True)))

        results = []
        for key, df in self._gpdf:
            results.append(func(df.reset_index(drop=True)))
            # NOTE: result_index() ensures each DataFrame has an index beginning at 0
        pdf = pd.concat(results, ignore_index=True)
        return DataFrame(pdf)

    def applyInSpark(self, func, keep_keys=False):
        """Applies the given function to a notspark.DataFrame for each group.

        NOTE: This is not part of the PySpark API.  In PySpark you would
        use Window functions, but these are a little trickier.
        """
        if keep_keys:
            pdf = self._gpdf.apply(lambda df: func(DataFrame(df.reset_index(drop=True))).toPandas())

            # drop the index that was automatically added for each DataFrame
            pdf.reset_index(
                level=len(pdf.index.names) - 1,
                drop=True,
                inplace=True)

            pdf.reset_index(drop=False, inplace=True)
            return DataFrame(pdf)
        else:
            results = []
            for key, df in self._gpdf:
                results.append(func(DataFrame(df.reset_index(drop=True))).toPandas())
            pdf = pd.concat(results, ignore_index=True)
            return DataFrame(pdf)

    def count(self):
        return self.agg(count("*").alias("count"))

    #def avg(self, *cols):
    #def max(self, *cols):
    #def mean(self, *cols):
    #def min(self, *cols):
    #def sum(self, *cols):

    #def cogroup(self, other):
    #def pivot(self, pivot_col, values):

from .readwriter import DataFrameWriter
from .column import to_col, Column, Variable, SortOrder, Alias
from .functions import \
    asc, covar_samp, \
    corr as _corr

import pandas as pd

__all__ = ["DataFrame"]  # "DataFrameNaFunctions", "DataFrameStatFunctions"]


class DataFrame(object):
    """DataFrame object.  Wraps a pandas.DataFrame and provides the PySpark API."""

    def __init__(self, pdf, alias=None):
        self._pdf = pdf
        self.alias = alias

    def __str__(self):
        return self._pdf.__str__()

    def __repr__(self):
        return self._pdf.__repr__()

    # def __getitem__(self, item: Union[int, str, Column, List, Tuple]) -> Union[Column, "DataFrame"]:
    def __getitem__(self, item):
        """Returns the column as a :class:`Column`.

        Examples
        --------
        >>> df.select(df['age']).collect()
        [Row(age=2), Row(age=5)]
        >>> df[ ["name", "age"]].collect()
        [Row(name='Alice', age=2), Row(name='Bob', age=5)]
        >>> df[ df.age > 3 ].collect()
        [Row(age=5, name='Bob')]
        >>> df[df[0] > 3].collect()
        [Row(age=5, name='Bob')]
        """
        if isinstance(item, str):
            return Variable(item)
        elif isinstance(item, Column):
            return self.filter(item)
        elif isinstance(item, (list, tuple)):
            return self.select(*item)
        elif isinstance(item, int):
            col_name = self.columns[item]
            return Variable(col_name)
        else:
            raise TypeError("unexpected item type: %s" % type(item))

    def __getattr__(self, name: str) -> Column:
        """Returns the :class:`Column` denoted by ``name``.

        Examples
        --------
        >>> df.select(df.age).collect()
        [Row(age=2), Row(age=5)]
        """
        if name not in self.columns:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (self.__class__.__name__, name)
            )
        return Variable(name)

    #def _repr_html_(self):

    def _todf(self, pdf):
        """Wraps the given pandas DataFrame in our DataFrame object.
        This is put in its own function in case we decide to start
        passing through context objects.
        """
        return DataFrame(pdf)

    def toPandas(self):
        """Return a pandas DataFrame.  In this case this is just the wrapped
        DataFrame"""
        return self._pdf

    def count(self) -> int:
        """Returns the number of rows"""
        return len(self._pdf)

    def isEmpty(self) -> bool:
        """Returns ``True`` if this :class:`DataFrame` is empty"""
        return len(self._pdf) == 0

    @property
    def columns(self):
        """Returns a list of column names"""
        return self._pdf.columns.to_list()

    #def colRegex(self, colName: str):
    #    """Selects a column basd on the column name as a regular expression"""
    #    pass

    @property
    def dtypes(self):  #-> List[Tuple[str, str]]:
        """Returns all column names and their data types as a list"""
        x = self._pdf.dtypes
        result = []
        for i in range(0, len(x)):
            result.append((x.index[i], str(x[i])))
        return result

    def slice(self, from_row, to_row):
        """Returns a DataFrame containing the given rows.  These are selected
        by row number.  In keeping with the PySpark API (which is based on SQL)
        the row numbers begin with 1 and are to_row is inclusive.

        NOTE: This function is not in the actual PySpark API, but we added
        it anyway."""
        pdf = self._pdf.iloc[(from_row-1):to_row]
        pdf.reset_index(drop=True, inplace=True)
        return self._todf(pdf)

    def limit(self, num=1):
        return self.head(num)

    def head(self, n=1):
        return self._todf(self._pdf.head(n))

    def tail(self, n=1):
        pdf = self._pdf.tail(n)
        pdf.reset_index(drop=True, inplace=True)
        return self._todf(pdf)

    @property
    def write(self) -> DataFrameWriter:
        """Returns a DataFrameWriter for saving the data."""
        return DataFrameWriter(self)

    #@property
    #def writeStream

    def filter(self, condition):
        """Filter rows using the given condition.  The condition should be a
        Column of bools"""
        condition = to_col(condition).eval(self._pdf)
        pdf = self._pdf[condition]
        pdf.reset_index(drop=True, inplace=True)
        return self._todf(pdf)

    def sort(self, *cols, **kwargs):
        """Returns a new DataFrame sorted by the specified column(s).
        ```cols``` can be a string, Column, or list of strings/Columns.
        """
        ascending = None
        if 'ascending' in kwargs:
            val = kwargs['ascending']
            if isinstance(val, list):
                if len(val) != len(cols):
                    raise ValueError("When 'ascending' is provided as a list it must contain one value for each column")
                ascending = val
            else:
                ascending = [val] * len(cols)

        # Convert *cols to list of SortOrder objects
        sort_orders = []
        for col in cols:
            if isinstance(col, list):
                sort_orders.extend(x if isinstance(x, SortOrder) else asc(x) for x in col)
            else:
                sort_orders.append(col if isinstance(col, SortOrder) else asc(col))

        na_position = 'first' if sort_orders[0].nulls_first else 'last'
        for so in sort_orders:
            if so.nulls_first != sort_orders[0].nulls_first:
                raise Exception("'nulls_first' setting must be the same for all values in the sort criteria")

        # if ascending wasn't specified, use values from SortOrder
        if ascending is None:
            ascending = [so.ascending for so in sort_orders]

        # create a DataFrame containing only the columns we want to sort on:
        # NOTE: these columns may not be in self._pdf.  They may be derived via eval().
        df_sort = pd.concat([so.col.eval(self._pdf) for so in sort_orders], axis=1)

        df_sort.sort_values(df_sort.columns.tolist(),
                            ascending=ascending, na_position=na_position,
                            inplace=True)
        pdf = self._pdf.iloc[df_sort.index]
        pdf.reset_index(drop=True, inplace=True)
        return self._todf(pdf)

    def select(self, *cols):
        """Projects a set of expressions and returns a new DataFrame.
        Columns can be strings (names) or Columns.  If '*' is present
        it will be expanded to include all columns in the current DataFrame."""

        all_strings = True
        asterisk_present = False
        for col in cols:
            if isinstance(col, str):
                if col == '*':
                    asterisk_present = True
            else:
                all_strings = False

        if all_strings:
            if asterisk_present:
                return self._todf(self._pdf.copy())
            else:
                return self._todf(self._pdf[list(cols)].copy())
        else:
            helper = SelectHelper(self._pdf)
            for col in cols:
                helper.add(col)

            return self._todf(helper.toPandas())

    def drop(self, *cols):
        """Removes the given columns from the DataFrame.  These can be
        indicated by strings or F.col(name) objects.
        """
        col_names = []
        for col in cols:
            if isinstance(col, str):
                col_names.append(col)
            elif isinstance(col, Variable):
                col_names.append(col.name)
            else:
                raise Exception("Only 'str' and 'F.col(name)' objects are supported in 'drop()'")

        return self._todf(self._pdf.drop(col_names, axis=1))

    def withColumnRenamed(self, existing, new):
        """Returns a DataFrame with the given column renamed.  This is a no-op if the
        column does not exist.
        """
        result = self._pdf.rename({existing: new}, axis=1)
        return self._todf(result)

    def withColumnsRenamed(self, *args):
        """Returns a DataFrame with multiple columns renamed.
        The even-numbered parameters are the names of existing columns
        while the following (odd-numbered) parameters are the new names.
        NOTE: This is not in the PySpark API (but I always wished it was)"""
        if len(args) % 2 != 0:
            raise Exception("An even number of parameters must be supplied")

        mapper = {}
        for i in range(0, len(args), 2):
            mapper[args[i]] = args[i+1]
        result = self._pdf.rename(mapper, axis=1)
        return self._todf(result)

    def withColumn(self, colName, col):
        """Returns a new DataFrame by adding a column or replacing the
        existing column that has the same name."""
        self._pdf[colName] = to_col(col).eval(self._pdf)
        return self

    def withColumns(self, *colsMap):
        """Adds multiple columns to the DataFrame

         Parameters
        ----------
        colsMap : dict
            a dict of column name and :class:`Column`. Currently, only single map is supported.
        Examples
        --------
        >>> df.withColumns({'age2': df.age + 2, 'age3': df.age + 3}).collect()
        [Row(age=2, name='Alice', age2=4, age3=5), Row(age=5, name='Bob', age2=7, age3=8)]
        """
        assert len(colsMap) == 1
        colsMap = colsMap[0]  # type: ignore[assignment]

        if not isinstance(colsMap, dict):
            raise TypeError("colsMap must be dict of column name and column.")

        df = self
        for key, value in colsMap.items():
            df = self.withColumn(key, value)
        return df

    def groupBy(self, *cols) -> "GroupedData":
        """Groups the data by the given columns.  Columns should be either
        strings or F.col('name') objects.
        """
        col_names = []
        for col in cols:
            if isinstance(col, str):
                col_names.append(col)
            elif isinstance(col, Variable):
                col_names.append(col.name)
            else:
                raise ValueError("Only column names and F.col('name') objects are supported in groupBy()")

        from .group import GroupedData
        return GroupedData(self._pdf.groupby(col_names, dropna=False))

    def groupby(self, *cols) -> "GroupedData":
        """Alias for 'groupBy()'"""
        return self.groupBy(*cols)

    def alias(self, alias:str):
        """Returns a new :class:`DataFrame` with an alias set"""
        return DataFrame(self._pdf, alias=alias)

    def join(
            self,
            other: "DataFrame",
            on = None,
            how:str = "inner",
    ) -> "DataFrame":
        """Joins with another :class:`DataFrame`, using the given join expression.
        Parameters
        ----------
        other : :class:`DataFrame`
            Right side of the join
        on : str, list or :class:`Column`, optional
            a string for the join column name, a list of column names,
            a join expression (Column), or a list of Columns.
            If `on` is a string or a list of strings indicating the name of the join column(s),
            the column(s) must exist on both sides, and this performs an equi-join.
        how : str, optional
            default ``inner``. Must be one of: ``inner``, ``cross``, ``outer``,
            ``full``, ``fullouter``, ``full_outer``, ``left``, ``leftouter``, ``left_outer``,
            ``right``, ``rightouter``, ``right_outer``, 
            TODO: ``semi``, ``leftsemi``, ``left_semi``, ``anti``, ``leftanti`` and ``left_anti``.
        """
        # convert 'on' to list of column names.  These must be common to both tables.
        # TODO: handle different names in tables:
        #       F.col("left_table.name") == F.col("right_table.first_name")
        on_names = []

        if on is None:
            # it would be nice to automatically find common columns using:
            #     on = list(set(self.columns).intersection(set(other.columns)))
            # but PySpark instead performs a cross join.  We add a check to make
            # sure how='cross' to prevent the user from doing an unintended cross join.
            if how != "cross":
                raise ValueError("'on' should only be ommitted for cross join")
        else:
            # ensure 'on' is a list:
            if not isinstance(on, list):
                on = [on]

            for col in on:
                if isinstance(col, str):
                    on_names.append(col)
                elif isinstance(col, Variable):
                    on_names.append(col.name)
                else:
                    raise ValueError("Only string and F.col(name) names are supported")

        pandas_how = None
        if how in ["inner", "left", "right", "cross", "outer"]:
            pandas_how = how
        elif how in ["fullouter", "full_outer"]:
            pandas_how = "outer"
        elif how in ["leftouter", "left_outer"]:
            pandas_how = "left"
        elif how in ["rightouter", "right_outer"]:
            pandas_how = "right"

        if pandas_how is None:
            if how in ["semi", "leftsemi", "left_semi"]:
                other_keys = other.toPandas()[on_names]
                result = pd.merge(self.toPandas(), other_keys, how="inner", on=on_names)
                return DataFrame(result)
            elif how in ["anti", "leftanti", "left_anti"]:
                other_keys = other.toPandas()[on_names]
                result = pd.merge(self.toPandas(), other_keys, how="left", on=on_names, indicator=True)
                result = result[result["_merge"] == "left_only"]
                result.drop("_merge", axis=1, inplace=True)
                result.reset_index(drop=True, inplace=True)
                return DataFrame(result)
            else:
                raise ValueError("Unknown join type: '" + str(how) + "'")
        else:
            if pandas_how == "cross":
                result = pd.merge(self.toPandas(), other.toPandas(), how="cross")
                return DataFrame(result)
            else:
                result = pd.merge(self.toPandas(), other.toPandas(), how=pandas_how, on=on_names)
                return DataFrame(result)

    def crossJoin(self, other: "DataFrame"):
        """Returns the cartesian product with another :class:`DataFrame`"""
        return self.join(other, how="cross")

    def transform(self, func, *args, **kwargs) -> "DataFrame":
        """Returns a new :class:`DataFrame`. Concise syntax for chaining custom transformations."""
        result = func(self, *args, **kwargs)
        assert isinstance(
            result, DataFrame
        ), "Func returned an instance of type [%s], " "should have been DataFrame." % type(result)
        return result

    def union(self, other: "DataFrame") -> "DataFrame":
        """Return a new :class:`DataFrame` containing union of rows in this and another
        :class:`DataFrame`.
        This is equivalent to `UNION ALL` in SQL. To do a SQL-style set union
        (that does deduplication of elements), use this function followed by :func:`distinct`.
        Also as standard in SQL, this function resolves columns by position (not by name).

        NOTE: Currently we just use pandas.concat()
        """
        return DataFrame(pd.concat([self.toPandas(), other.toPandas()]))

    def unionAll(self, other: "DataFrame") -> "DataFrame":
        """Alias for union()"""
        return self.union(other)

    #unionByName  # TODO: look into semantics of pd.concat to see if we need this

    def corr(self, col1: str, col2: str, method: str = None) -> float:
        """
        Calculates the correlation of two columns of a :class:`DataFrame` as a double value.
        Currently only supports the Pearson Correlation Coefficient.
        """
        if not isinstance(col1, str):
            raise TypeError("col1 should be a string.")
        if not isinstance(col2, str):
            raise TypeError("col2 should be a string.")
        if not method:
            method = "pearson"
        if not method == "pearson":
            raise ValueError(
                "Currently only the calculation of the Pearson Correlation "
                + "coefficient is supported."
            )

        return _corr(Variable(col1), Variable(col2)).eval(self._pdf)

    def cov(self, col1: str, col2: str) -> float:
        """
        Calculate the sample covariance for the given columns, specified by their names, as a
        double value.
        Parameters
        ----------
        col1 : str
            The name of the first column
        col2 : str
            The name of the second column
        """
        if not isinstance(col1, str):
            raise TypeError("col1 should be a string.")
        if not isinstance(col2, str):
            raise TypeError("col2 should be a string.")

        return covar_samp(Variable(col1), Variable(col2)).eval(self._pdf)

    # TODO:

    # def distinct(self) -> "DataFrame":
    #    """Returns a new DataFrame with distinct rows"""

    # def show(self, n: int=20, truncate: Union[bool, int] = True, vertical: bool = False) -> None:
    #    """Prints the first ``n`` rows to the console.

    # CONSIDER DOING:
    # sample
    # sampleBy
    # randomSplit
    # describe
    # summary
    # first
    # selectExpr
    # toDF
    # intersect
    # intersectAll
    # subtract
    # dropDuplicates
    # dropna
    # fillna
    # replace
    # approxQuantile
    # crosstab
    # freqItems
    # sparkSession @property

    # Probably don't do:
    # def sql_ctx @property

    # rdd
    # na @property
    # stat @property
    # toJSON
    # registerTempTable
    # createTempView
    # createOrReplaceTempView
    # createGlobalTempView
    # createOrReplaceGlobalTempView
    # schema @property
    # printSchema
    # explain
    # exceptAll
    # isLocal
    # isStreaming @property
    # withMetadata
    # checkpoint
    # localCheckpoint
    # hint
    # collect
    # toLocalIterator
    # take
    # foreach
    # foreachPartition
    # cache
    # persist
    # storageLevel @property
    # unpersist
    # coalesce
    # repartition
    # repartitionByRange
    # sortWithinPartitions
    # rollup
    # cube
    # observe
    # sameSemantics
    # semanticHash
    # inputFiles
    # writeTo
    # to_pandas_on_spark
    # pandas_api
    # withWatermark # streaming


class SelectHelper(object):
    """Helper for the 'DataFrame.select()' that lets us add any of the following
    when constructing the result:
    * string
    * Column (Variable, or other)
    * Alias
    """
    def __init__(self, pdf):
        self._pdf = pdf
        self.cols = []
        self.col_names = []
        
    def add(self, col):
        if isinstance(col, Alias):
            self.cols.append(col.col)
            self.col_names.append(col.alias)
        else:
            col = to_col(col)
            if isinstance(col, Variable):
                if col.name == '*':
                    for x in self._pdf.columns.tolist():
                        self.cols.append(Variable(x))
                        self.col_names.append(x)
                else:
                    self.cols.append(col)
                    self.col_names.append(col.name)
            else:
                self.cols.append(col)
                self.col_names.append("_" + str(len(self.cols)))  # 1-indexed
                # NOTE: spark.createDataFrame([('Alice', 1)]) names columns _1 and _2

    def toPandas(self) -> pd.DataFrame:
        series_list = []
        for col in self.cols:
            series = col.eval(self._pdf)
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            series_list.append(series)

        result = pd.concat(series_list, axis=1)
        result.set_axis(self.col_names, axis=1, inplace=True)
        return result


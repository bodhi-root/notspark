from typing import Optional, Union
import pandas as pd
import os.path

#TODO: reconsider all this.  Let's have users use the pandas syntax if
#      they want to load data that way.  For example:
#
#          df = spark.createDataFrame(pd.read_csv('input.csv'))
#          df.toPandas().to_csv('output.csv')
#
#      We should only implement methods here that conform to the Spark
#      API - which means interpreting the parameters and then calling
#      to_csv() for them.

__all__ = ["DataFrameReader", "DataFrameWriter"]


class DataFrameReader(object):
    """Reads data.  This provides aliases for the pandas data reading functions
    such as 'read_csv()'.  It also supports a Spark-style syntax:
      spark.read.format("csv").options(**kwargs).load(path)
    """
    def __init__(self, spark):
        self._spark = spark

        self._format = None    # can be set by config: spark.sql.sources.default
        self._options = {}

    def _df(self, pdf: pd.DataFrame) -> "DataFrame":
        from .dataframe import DataFrame
        return DataFrame(pdf)

    def format(self, source: str) -> "DataFrameReader":
        """Specifies the input data source format."""
        self._format = source
        return self

    def option(self, key: str, value: Optional) -> "DataFrameReader":
        """Adds an input option for the underlying data source."""
        self._options[key] = value
        return self

    def options(self, **options: Optional) -> "DataFrameReader":
        """Adds input options for the underlying data source."""
        self._options.update(options)
        return self

    def load(
            self,
            path: Optional = None,
            format: Optional[str] = None,
            **options
    ) -> "DataFrame":
        """Loads data from a data source and returns it as a :class:`DataFrame`."""
        if format is None:
            format = self._format
            if format is None:
                raise ValueError("Missing required parameter: 'format'")
        if options is None:
            options = {}
        options.update(self._options)
        f = getattr(self, format)
        return f(path, **options)

    #def json(self):
    #def table(self):
    #def parquet(self):
    #def text(self):

    def csv(self, path, **kwargs):
        if len(kwargs) > 0:
            raise Exception("Function parameters not implemented correctly.  Use spark.createDataFrame(pd.read_csv(...)) instead")
        df = pd.read_csv(path)
        return self._df(df)

    #def orc(self):
    #def jdbc


class DataFrameWriter(object):
    """Data writer.  This provides a Spark-like interface for saving DataFrames.
    For example:

      DataFrame.write.format("csv").mode("overwrite").options(**options).save(path)

    Currently these methods just dispatch to their pandas.DataFrame equivalents
    (such as DataFrame.to_csv()) passing whatever options were provided to the
    writer as additional parameters.  This may be changed in the future to more
    strictly adhere to the Spark interface.

    NOTE: We did not include aliases for pandas save functions because these are easily
    available via DataFrame.toPandas().to_csv().
    """
    def __init__(self, df: "DataFrame"):
        self._df = df
        self._format = None
        self._mode = "errorifexists"
        self._options = {}

    def mode(self, saveMode: Optional[str]) -> "DataFrameWriter":
        """Specifies the behavior when data or table already exists.
        Options include:
        * `append`: Append contents of this :class:`DataFrame` to existing data.
        * `overwrite`: Overwrite existing data.
        * `error` or `errorifexists`: Throw an exception if data already exists.
        * `ignore`: Silently ignore this operation if data already exists.
        """
        self._mode = saveMode
        return self

    def format(self, source: str) -> "DataFrameWriter":
        """Specifies the underlying output data source.
        """
        self._format = source
        return self

    def option(self, key: str, value: "OptionalPrimitiveType") -> "DataFrameWriter":
        """Adds an output option for the underlying data source."""
        self._options[key] = value
        return self

    def options(self, **options: "OptionalPrimitiveType") -> "DataFrameWriter":
        """Adds output options for the underlying data source."""
        self._options.update(options)
        return self

    def save(
            self,
            path: Optional[str] = None,
            format: Optional[str] = None,
            mode: Optional[str] = None,
            #partitionBy: Optional[Union[str, List[str]]] = None,
            **options: "OptionalPrimitiveType",
    ) -> None:
        """Saves the contents of the :class:`DataFrame` to a data source."""
        if format is None:
            format = self._format
            if format is None:
                raise ValueError("Missing required parameter: 'format'")
        if mode is None:
            mode = self._mode
        if options is None:
            options = {}
        options.update(self._options)
        f = getattr(self, format)
        f(path, mode, **options)

    #def saveAsTable

    def json(
            self,
            path: str,
            mode: Optional[str] = None,
            compression: Optional[str] = None,
            dateFormat: Optional[str] = None,
            timestampFormat: Optional[str] = None,
            lineSep: Optional[str] = None,
            encoding: Optional[str] = None,
            ignoreNullFields: Optional[Union[bool, str]] = None,
    ) -> None:
        if compression is not None:
            raise Exception("Parameter not (yet) supported: 'compression'")
        if dateFormat is not None:
            raise Exception("Parameter not (yet) supported: 'dateFormat'")
        if timestampFormat is not None:
            raise Exception("Parameter not (yet) supported: 'timestampFormat'")
        if lineSep is not None:
            raise Exception("Parameter not (yet) supported: 'lineSep'")
        if encoding is not None:
            raise Exception("Parameter not (yet) supported: 'encoding'")
        if ignoreNullFields is not None:
            raise Exception("Parameter not (yet) supported: 'ignoreNullFields'")

        if mode is None:
            mode = self._mode

        if mode == "ignore":
            return
        elif mode == "error" or self._mode == "errorifexists":
            if os.path.exists(path):
                raise Exception("File exists. Use 'ovewrite' mode to continue.")
        elif mode == "overwrite":
            pass
        elif mode == "append":
            raise Exception("'append' mode not supported for JSON files")
        else:
            raise ValueError("Invalid parameter 'mode' = '" + str(mode) + "'")

        self._df.toPandas().to_json(path)

    #def parquet
    #def text

    def csv(
            self,
            path: str,
            mode: Optional[str] = None,
            #compression: Optional[str] = None,
            #sep: Optional[str] = None,
            #quote: Optional[str] = None,
            #escape: Optional[str] = None,
            #header: Optional[Union[bool, str]] = None,
            #nullValue: Optional[str] = None,
            #escapeQuotes: Optional[Union[bool, str]] = None,
            #quoteAll: Optional[Union[bool, str]] = None,
            #dateFormat: Optional[str] = None,
            #timestampFormat: Optional[str] = None,
            #ignoreLeadingWhiteSpace: Optional[Union[bool, str]] = None,
            #ignoreTrailingWhiteSpace: Optional[Union[bool, str]] = None,
            #charToEscapeQuoteEscaping: Optional[str] = None,
            #encoding: Optional[str] = None,
            #emptyValue: Optional[str] = None,
            #lineSep: Optional[str] = None,
    ) -> None:

        kwargs = {}

        if mode == "ignore":
            return
        elif mode == "error" or self._mode == "errorifexists":
            if os.path.exists(path):
                raise Exception("File exists. Use 'ovewrite' mode to continue.")
        elif mode == "overwrite":
            kwargs["mode"] = "w"
        elif mode == "append":
            kwargs["mode"] = "a"
        else:
            raise ValueError("Invalid parameter 'mode' = '" + str(mode) + "'")

        self._df.toPandas().to_csv(path, **kwargs)

   #def orc
   #def jdbc


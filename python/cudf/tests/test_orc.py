# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest


@pytest.fixture(scope='module')
def datadir(datadir):
    return datadir / 'orc'


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.parametrize('engine', ['pyarrow', 'cudf'])
@pytest.mark.parametrize(
    'inputfile, columns',
    [
        ('TestOrcFile.emptyFile.orc', ['boolean1']),
        ('TestOrcFile.test1.orc', ['boolean1', 'byte1', 'short1',
                                   'int1', 'long1', 'float1', 'double1']),
        ('TestOrcFile.testSnappy.orc', None),
        ('TestOrcFile.demo-12-zlib.orc', ['_col2', '_col3', '_col4', '_col5'])
    ]
)
def test_orc_reader_basic(datadir, inputfile, columns, engine):
    path = datadir / inputfile
    try:
        orcfile = pa.orc.ORCFile(path)
    except Exception as excpr:
        if type(excpr).__name__ == 'ArrowIOError':
            pytest.skip('.orc file is not found')
        else:
            print(type(excpr).__name__)

    expect = orcfile.read(columns=columns).to_pandas()
    got = cudf.read_orc(path, engine=engine, columns=columns)

    assert_eq(expect, got, check_categorical=False)


def test_orc_reader_decimal(datadir):
    path = datadir / 'TestOrcFile.decimal.orc'
    try:
        orcfile = pa.orc.ORCFile(path)
    except Exception as excpr:
        if type(excpr).__name__ == 'ArrowIOError':
            pytest.skip('.orc file is not found')
        else:
            print(type(excpr).__name__)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(path, engine='cudf').to_pandas()

    # Convert the decimal dtype from PyArrow to float64 for comparison to cuDF
    # This is because cuDF returns as float64 as it lacks an equivalent dtype
    pdf = pdf.apply(pd.to_numeric)

    np.testing.assert_allclose(pdf, gdf)


def test_orc_reader_filenotfound(tmpdir):
    with pytest.raises(FileNotFoundError):
        cudf.read_orc('TestMissingFile.orc')

    with pytest.raises(FileNotFoundError):
        cudf.read_orc(tmpdir.mkdir("cudf_orc"))


def test_orc_reader_trailing_nulls(datadir):
    path = datadir / 'TestOrcFile.nulls-at-end-snappy.orc'
    try:
        orcfile = pa.orc.ORCFile(path)
    except Exception as excpr:
        if type(excpr).__name__ == 'ArrowIOError':
            pytest.skip('.orc file is not found')
        else:
            print(type(excpr).__name__)

    expect = orcfile.read().to_pandas().fillna(0)
    got = cudf.read_orc(path, engine='cudf').fillna(0)

    # PANDAS uses NaN to represent invalid data, which forces float dtype
    # For comparison, we can replace NaN with 0 and cast to the cuDF dtype
    for col in expect.columns:
        expect[col] = expect[col].astype(got[col].dtype)

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.parametrize('inputfile', ['TestOrcFile.testDate1900.orc',
                                       'TestOrcFile.testDate2038.orc'])
def test_orc_reader_datetimestamp(datadir, inputfile):
    path = datadir / inputfile
    try:
        orcfile = pa.orc.ORCFile(path)
    except Exception as excpr:
        if type(excpr).__name__ == 'ArrowIOError':
            pytest.skip('.orc file is not found')
        else:
            print(type(excpr).__name__)

    pdf = orcfile.read().to_pandas(date_as_object=False)
    gdf = cudf.read_orc(path, engine='cudf')

    # cuDF DatetimeColumn currenly only supports millisecond units
    # Convert to lesser precision for comparison
    timedelta = np.timedelta64(1, 'ms').astype('timedelta64[ns]')
    pdf['time'] = pdf['time'].astype(np.int64) // timedelta.astype(np.int64)
    gdf['time'] = gdf['time'].astype(np.int64)

    assert_eq(pdf, gdf, check_categorical=False)


def test_orc_reader_strings(datadir):
    path = datadir / 'TestOrcFile.testStringAndBinaryStatistics.orc'
    try:
        orcfile = pa.orc.ORCFile(path)
    except Exception as excpr:
        if type(excpr).__name__ == 'ArrowIOError':
            pytest.skip('.orc file is not found')
        else:
            print(type(excpr).__name__)

    expect = orcfile.read(columns=['string1'])
    got = cudf.read_orc(path, engine='cudf', columns=['string1'])

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.parametrize('num_rows', [1, 100, 3000])
@pytest.mark.parametrize('skip_rows', [0, 1, 3000])
def test_orc_read_rows(datadir, skip_rows, num_rows):
    path = datadir / 'TestOrcFile.decimal.orc'
    try:
        orcfile = pa.orc.ORCFile(path)
    except Exception as excpr:
        if type(excpr).__name__ == 'ArrowIOError':
            pytest.skip('.orc file is not found')
        else:
            print(type(excpr).__name__)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(
        path,
        engine='cudf',
        skip_rows=skip_rows,
        num_rows=num_rows
    ).to_pandas()

    # Convert the decimal dtype from PyArrow to float64 for comparison to cuDF
    # This is because cuDF returns as float64 as it lacks an equivalent dtype
    pdf = pdf.apply(pd.to_numeric)

    # Slice rows out of the whole dataframe for comparison as PyArrow doesn't
    # have an API to read a subsection of rows from the file
    pdf = pdf[skip_rows:skip_rows + num_rows]

    np.testing.assert_allclose(pdf, gdf)

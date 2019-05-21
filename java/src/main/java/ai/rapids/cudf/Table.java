/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Class to represent a collection of ColumnVectors and operations that can be performed on them collectively.
 * The refcount on the columns will be increased once they are passed in
 */
public final class Table implements AutoCloseable {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    private long nativeHandle;
    private ColumnVector[] columns;
    private final long rows;

    /**
     * Table class makes a copy of the array of {@link ColumnVector}s passed to it. The class will decrease the refcount
     * on itself and all its contents when closed and free resources if refcount is zero
     * @param columns - Array of ColumnVectors
     */
    public Table(ColumnVector... columns) {
        assert columns != null : "ColumnVectors can't be null";
        rows = columns[0].getRowCount();

        for (ColumnVector columnVector : columns) {
            assert (null != columnVector) : "ColumnVectors can't be null";
            assert (rows == columnVector.getRowCount()) : "All columns should have the same number of rows";
        }

        // Since Arrays are mutable objects make a copy
        this.columns = new ColumnVector[columns.length];
        long[] cudfColumnPointers = new long[columns.length];
        for (int i = 0 ; i < columns.length ; i++) {
            this.columns[i] = columns[i];
            columns[i].incRefCount();
            cudfColumnPointers[i] = columns[i].getNativeCudfColumnAddress();
        }

        nativeHandle = createCudfTable(cudfColumnPointers);
    }

    private Table(long[] cudfColumns) {
        assert cudfColumns != null : "CudfColumns can't be null";

        this.columns = new ColumnVector[cudfColumns.length];
        for (int i = 0 ; i < cudfColumns.length ; i++) {
            this.columns[i] = new ColumnVector(cudfColumns[i]);
        }
        nativeHandle = createCudfTable(cudfColumns);
        this.rows = columns[0].getRowCount();
    }

    /**
     * Return the {@link ColumnVector} at the specified index. If you want to keep a reference to
     * the column around past the life time of the table, you will need to increment the reference
     * count on the column yourself.
     */
    public ColumnVector getColumn(int index) {
        assert index < columns.length;
        return columns[index];
    }

    public final long getRowCount() {
        return rows;
    }

    public final int getNumberOfColumns() {
        return columns.length;
    }

    @Override
    public void close() {
        if (nativeHandle != 0) {
            freeCudfTable(nativeHandle);
            nativeHandle = 0;
        }
        if (columns != null) {
            for (int i = 0; i < columns.length; i++) {
                columns[i].close();
                columns[i] = null;
            }
            columns = null;
        }
    }

    @Override
    public String toString() {
        return "Table{" +
                "columns=" + Arrays.toString(columns) +
                ", cudfTable=" + nativeHandle +
                ", rows=" + rows +
                '}';
    }

    /////////////////////////////////////////////////////////////////////////////
    // NATIVE APIs
    /////////////////////////////////////////////////////////////////////////////

    private static native long createCudfTable(long[] cudfColumnPointers) throws CudfException;

    private static native void freeCudfTable(long handle) throws CudfException;

    /**
     * Ugly long function to read CSV.  This is a long function to avoid the overhead of reaching into a java
     * object to try and pull out all of the options.  If this becomes unwieldy we can change it.
     * @param columnNames names of all of the columns, even the ones filtered out
     * @param dTypes types of all of the columns as strings.  Why strings? who knows.
     * @param filterColumnNames name of the columns to read, or an empty array if we want to read all of them
     * @param filePath the path of the file to read, or null if no path should be read.
     * @param address the address of the buffer to read from or 0 if we should not.
     * @param length the length of the buffer to read from.
     * @param headerRow the 0 based index row of the header can be -1
     * @param delim character deliminator (must be ASCII).
     * @param quote character quote (must be ASCII).
     * @param comment character that starts a comment line (must be ASCII) use '\0'
     * @param nullValues values that should be treated as nulls
     */
    private static native long[] gdfReadCSV(String[] columnNames, String[] dTypes, String[] filterColumnNames,
                                            String filePath, long address, long length,
                                            int headerRow, byte delim, byte quote,
                                            byte comment, String[] nullValues) throws CudfException;

    /**
     * Read in Parquet formatted data.
     * @param filterColumnNames name of the columns to read, or an empty array if we want to read all of them
     * @param filePath the path of the file to read, or null if no path should be read.
     * @param address the address of the buffer to read from or 0 if we should not.
     * @param length the length of the buffer to read from.
     */
    private static native long[] gdfReadParquet(String[] filterColumnNames,
                                                String filePath, long address, long length) throws CudfException;

    private static native long[] gdfOrderBy(long inputTable, long[] sortKeys, boolean[] isDescending,
                                          boolean areNullsSmallest) throws CudfException;

    private static native long[] gdfLeftJoin(long leftTable, int[] leftJoinCols, long rightTable,
                                             int[] rightJoinCols) throws CudfException;

    private static native long[] gdfInnerJoin(long leftTable, int[] leftJoinCols, long rightTable,
                                              int[] rightJoinCols) throws CudfException;


    /////////////////////////////////////////////////////////////////////////////
    // TABLE CREATION APIs
    /////////////////////////////////////////////////////////////////////////////

    public static Table readCSV(Schema schema, File path) {
        return readCSV(schema, CSVOptions.DEFAULT, path);
    }

    public static Table readCSV(Schema schema, CSVOptions opts, File path) {
        return new Table(
                gdfReadCSV(schema.getColumnNames(), schema.getTypesAsStrings(),
                        opts.getIncludeColumnNames(), path.getAbsolutePath(),
                        0, 0,
                        opts.getHeaderRow(),
                        opts.getDelim(),
                        opts.getQuote(),
                        opts.getComment(),
                        opts.getNullValues()));
    }

    public static Table readCSV(Schema schema, byte[] buffer) {
        return readCSV(schema, CSVOptions.DEFAULT, buffer, 0, buffer.length);
    }

    public static Table readCSV(Schema schema, CSVOptions options, byte[] buffer) {
        return readCSV(schema, options, buffer, 0, buffer.length);
    }

    public static Table readCSV(Schema schema, CSVOptions opts, byte[] buffer, long offset, long len) {
        if (len <= 0) {
            len = buffer.length - offset;
        }
        assert len > 0;
        assert len <= buffer.length - offset;
        assert offset >= 0 && offset < buffer.length;
        try (HostMemoryBuffer newBuf = HostMemoryBuffer.allocate(len)) {
            newBuf.setBytes(0, buffer, offset, len);
            return readCSV(schema, opts, newBuf, len);
        }
    }

    private static Table readCSV(Schema schema, CSVOptions opts, HostMemoryBuffer buffer, long len) {
        assert len > 0;
        assert len <= buffer.getLength();
        return new Table(gdfReadCSV(schema.getColumnNames(), schema.getTypesAsStrings(),
                opts.getIncludeColumnNames(), null,
                buffer.getAddress(), len,
                opts.getHeaderRow(),
                opts.getDelim(),
                opts.getQuote(),
                opts.getComment(),
                opts.getNullValues()));
    }

    public static Table readParquet(File path) {
        return readParquet(ParquetOptions.DEFAULT, path);
    }

    public static Table readParquet(ParquetOptions opts, File path) {
        return new Table(gdfReadParquet(opts.getIncludeColumnNames(),
                path.getAbsolutePath(), 0, 0));
    }

    public static Table readParquet(byte[] buffer) {
        return readParquet(ParquetOptions.DEFAULT, buffer, buffer.length);
    }

    public static Table readParquet(ParquetOptions opts, byte[] buffer) {
        return readParquet(opts, buffer, buffer.length);
    }

    public static Table readParquet(ParquetOptions opts, byte[] buffer, long len) {
        if (len <= 0) {
            len = buffer.length;
        }
        assert len > 0;
        assert len <= buffer.length;
        try (HostMemoryBuffer newBuf = HostMemoryBuffer.allocate(len)) {
            newBuf.setBytes(0, buffer, 0, len);
            return readParquet(opts, newBuf, len);
        }
    }

    static Table readParquet(ParquetOptions opts, HostMemoryBuffer buffer, long len) {
        return new Table(gdfReadParquet(opts.getIncludeColumnNames(),
                null, buffer.getAddress(), len));
    }

    /////////////////////////////////////////////////////////////////////////////
    // TABLE MANIPULATION APIs
    /////////////////////////////////////////////////////////////////////////////

    /**
     * Orders the table using the sortkeys returning a new allocated table. The caller is responsible for cleaning up
     * the {@link ColumnVector} returned as part of the output {@link Table}
     *
     * Example usage: orderBy(true, Table.asc(0), Table.desc(3)...);
     *
     * @param areNullsSmallest - represents if nulls are to be considered smaller than non-nulls.
     * @param args - Suppliers to initialize sortKeys.
     * @return Sorted Table
     */
    public Table orderBy(boolean areNullsSmallest, OrderByArg... args){
        assert args.length <= columns.length;
        long[] sortKeys = new long[args.length];
        boolean[] isDescending = new boolean[args.length];
        for (int i = 0 ; i < args.length ; i++) {
            int index = args[i].index;
            assert (index >= 0 && index < columns.length) :
                    "index is out of range 0 <= " + index + " < " + columns.length;
            isDescending[i] = args[i].isDescending;
            sortKeys[i] = columns[index].getNativeCudfColumnAddress();
        }

        return new Table(gdfOrderBy(nativeHandle, sortKeys, isDescending, areNullsSmallest));
    }

    public static OrderByArg asc(final int index) {
        return new OrderByArg(index, false);
    }

    public static OrderByArg desc(final int index) {
        return new OrderByArg(index, true);
    }

    public JoinColumns joinColumns(int... indices) {
        int[] joinIndicesArray = new int[indices.length];
        for (int i = 0 ; i < indices.length ; i++) {
            joinIndicesArray[i] = indices[i];
            assert joinIndicesArray[i] >= 0 && joinIndicesArray[i] < columns.length :
                    "join index is out of range 0 <= " + joinIndicesArray[i] + " < " + columns.length;
        }
        return new JoinColumns(this, joinIndicesArray);
    }

    /////////////////////////////////////////////////////////////////////////////
    // HELPER CLASSES
    /////////////////////////////////////////////////////////////////////////////

    public static final class OrderByArg {
        final int index;
        final boolean isDescending;

        OrderByArg(int index, boolean isDescending) {
            this.index = index;
            this.isDescending = isDescending;
        }
    }

    public static final class JoinColumns {
        private final int[] indices;
        private final Table table;

        JoinColumns(final Table table, final int... indices) {
            this.indices = indices;
            this.table = table;
        }

        /**
         * Joins two tables on the join columns that are passed in.
         * Usage:
         *      Table t1 ...
         *      Table t2 ...
         *      Table result = t1.joinColumns(0,1).leftJoin(t2.joinColumns(2,3));
         * @param rightJoinIndices - Indices of the right table to join on
         * @return Joined {@link Table}
         */
        public Table leftJoin(JoinColumns rightJoinIndices) {
            return new Table(gdfLeftJoin(this.table.nativeHandle, indices,
                    rightJoinIndices.table.nativeHandle, rightJoinIndices.indices));
        }

        /**
         * Joins two tables on the join columns that are passed in.
         * Usage:
         *      Table t1 ...
         *      Table t2 ...
         *      Table result = t1.joinColumns(0,1).innerJoin(t2.joinColumns(2,3));
         * @param rightJoinIndices - Indices of the right table to join on
         * @return Joined {@link Table}
         */
        public Table innerJoin(JoinColumns rightJoinIndices) {
            return new Table(gdfInnerJoin(this.table.nativeHandle, indices,
                    rightJoinIndices.table.nativeHandle, rightJoinIndices.indices));
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    // BUILDER
    /////////////////////////////////////////////////////////////////////////////

    /**
     * Create a table on the GPU with data from the CPU.  This is not fast and intended mostly for
     * tests.
     */
    public static final class TestBuilder {
        private final List<DType> types = new ArrayList<>();
        private final List<TimeUnit> units = new ArrayList<>();
        private final List<Object> typeErasedData = new ArrayList<>();

        public TestBuilder column(Byte... values) {
            types.add(DType.INT8);
            units.add(TimeUnit.NONE);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder column(Short... values) {
            types.add(DType.INT16);
            units.add(TimeUnit.NONE);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder column(Integer... values) {
            types.add(DType.INT32);
            units.add(TimeUnit.NONE);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder column(Long... values) {
            types.add(DType.INT64);
            units.add(TimeUnit.NONE);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder column(Float... values) {
            types.add(DType.FLOAT32);
            units.add(TimeUnit.NONE);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder column(Double... values) {
            types.add(DType.FLOAT64);
            units.add(TimeUnit.NONE);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder date32Column(Integer... values) {
            types.add(DType.DATE32);
            units.add(TimeUnit.NONE);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder date64Column(Long... values) {
            types.add(DType.DATE64);
            units.add(TimeUnit.NONE);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder timestampColumn(Long... values) {
            types.add(DType.TIMESTAMP);
            units.add(TimeUnit.NONE);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder timestampColumn(TimeUnit unit, Long... values) {
            types.add(DType.TIMESTAMP);
            units.add(unit);
            typeErasedData.add(values);
            return this;
        }

        private static ColumnVector from(DType type, TimeUnit unit, Object dataArray) {
            ColumnVector ret;
            switch(type) {
                case INT8:
                    ret = ColumnVector.fromBoxedBytes((Byte[]) dataArray);
                    break;
                case INT16:
                    ret = ColumnVector.fromBoxedShorts((Short[]) dataArray);
                    break;
                case INT32:
                    ret = ColumnVector.fromBoxedInts((Integer[]) dataArray);
                    break;
                case INT64:
                    ret = ColumnVector.fromBoxedLongs((Long[]) dataArray);
                    break;
                case DATE32:
                    ret = ColumnVector.datesFromBoxedInts((Integer[]) dataArray);
                    break;
                case DATE64:
                    ret = ColumnVector.datesFromBoxedLongs((Long[]) dataArray);
                    break;
                case TIMESTAMP:
                    ret = ColumnVector.timestampsFromBoxedLongs(unit, (Long[]) dataArray);
                    break;
                case FLOAT32:
                    ret = ColumnVector.fromBoxedFloats((Float[]) dataArray);
                    break;
                case FLOAT64:
                    ret = ColumnVector.fromBoxedDoubles((Double[]) dataArray);
                    break;
                default:
                    throw new IllegalArgumentException(type + " is not supported yet");
            }
            return ret;
        }

        public Table build() {
            List<ColumnVector> columns = new ArrayList<>(types.size());
            try {
                for (int i = 0; i < types.size(); i++) {
                    columns.add(from(types.get(i), units.get(i), typeErasedData.get(i)));
                }
                for (ColumnVector cv: columns) {
                    cv.ensureOnDevice();
                }
                return new Table(columns.toArray(new ColumnVector[columns.size()]));
            } finally {
                for (ColumnVector cv: columns) {
                    cv.close();
                }
            }
        }
    }
}

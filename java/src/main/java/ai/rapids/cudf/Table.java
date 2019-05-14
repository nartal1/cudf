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
    private final ColumnVector[] columnVectors;
    private final CudfTable cudfTable;
    private final long rows;

    /**
     * Table class makes a copy of the array of {@link ColumnVector}s passed to it. The class will decrease the refcount
     * on itself and all its contents when closed and free resources if refcount is zero
     * @param columnVectors - Array of ColumnVectors
     */
    public Table(ColumnVector[] columnVectors) {
        assert columnVectors != null : "ColumnVectors can't be null";
        final long rows = columnVectors[0].getRows();

        for (ColumnVector columnVector : columnVectors) {
            assert (null != columnVector) : "ColumnVectors can't be null";
            assert (rows == columnVector.getRows()) : "All columns should have the same number of rows";
        }
        this.columnVectors = new ColumnVector[columnVectors.length];
        // Since Arrays are mutable objects make a copy
        for (int i = 0 ; i < columnVectors.length ; i++) {
            this.columnVectors[i] = columnVectors[i];
            columnVectors[i].incRefCount();
        }
        CudfColumn[] cudfColumns = new CudfColumn[columnVectors.length];
        for (int i = 0 ; i < columnVectors.length ; i++) {
            cudfColumns[i] = columnVectors[i].getCudfColumn();
        }
        cudfTable = new CudfTable(cudfColumns);
        this.rows = rows;
    }

    Table(CudfColumn[] cudfColumns) {
        assert cudfColumns != null : "CudfColumns can't be null";

        this.columnVectors = new ColumnVector[cudfColumns.length];
        for (int i = 0 ; i < cudfColumns.length ; i++) {
            this.columnVectors[i] = ColumnVector.fromCudfColumn(cudfColumns[i]);
        }
        cudfTable = new CudfTable(cudfColumns);
        this.rows = cudfColumns[0].getSize();
    }

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
        assert args.length <= columnVectors.length;
        int[] sortKeysIndices = new int[args.length];
        boolean[] isDescending = new boolean[args.length];
        for (int i = 0 ; i < args.length ; i++) {
            sortKeysIndices[i] = args[i].index;
            assert (sortKeysIndices[i] >= 0 && sortKeysIndices[i] < columnVectors.length) :
                    "index is out of range 0 <= " + sortKeysIndices[i] + " < " + columnVectors.length;
            isDescending[i] = args[i].isDescending;}
        Table outputTable = Table.newOutputTable(this.columnVectors);
        cudfTable.gdfOrderBy(sortKeysIndices, isDescending, outputTable.cudfTable, areNullsSmallest);
        // We allocated the ColumnVectors in Java before the output was calculated therefore
        // we have to update them from native values
        for (ColumnVector columnVector : outputTable.columnVectors ) {
            columnVector.updateFromNative();
        }
        return outputTable;
    }

    private static Table newOutputTable(ColumnVector[] inputColumnVectors) {
        ColumnVector[] outputColumnVectors = new ColumnVector[inputColumnVectors.length];
        for (int i = 0 ; i < inputColumnVectors.length ; i++) {
            outputColumnVectors[i] = ColumnVector.newOutputVector(inputColumnVectors[i].getRows(),
                    inputColumnVectors[i].hasValidityVector(), inputColumnVectors[i].getType());
        }
        return new Table(outputColumnVectors);
    }

    public static Table readCSV(Schema schema, File path) {
        return readCSV(schema, CSVOptions.DEFAULT, path);
    }

    public static Table readCSV(Schema schema, CSVOptions opts, File path) {
        CudfColumn[] columns = CudfTable.readCSV(schema, opts, path.getAbsolutePath());
        return new Table(columns);
    }

    public static Table readCSV(Schema schema, byte[] buffer) {
        return readCSV(schema, CSVOptions.DEFAULT, buffer, buffer.length);
    }

    public static Table readCSV(Schema schema, CSVOptions opts, byte[] buffer, long len) {
        if (len <= 0) {
            len = buffer.length;
        }
        assert len > 0;
        assert len <= buffer.length;
        try (HostMemoryBuffer newBuf = HostMemoryBuffer.allocate(len)) {
            newBuf.setBytes(0, buffer, len);
            return readCSV(schema, opts, newBuf, len);
        }
    }

    static Table readCSV(Schema schema, CSVOptions opts, HostMemoryBuffer buffer, long len) {
        CudfColumn[] columns = CudfTable.readCSV(schema, opts, buffer, len);
        return new Table(columns);
    }

    public static Table readParquet(File path) {
        return readParquet(ParquetOptions.DEFAULT, path);
    }

    public static Table readParquet(ParquetOptions opts, File path) {
        CudfColumn[] columns = CudfTable.readParquet(opts, path);
        return new Table(columns);
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
            newBuf.setBytes(0, buffer, len);
            return readParquet(opts, newBuf, len);
        }
    }

    static Table readParquet(ParquetOptions opts, HostMemoryBuffer buffer, long len) {
        CudfColumn[] columns = CudfTable.readParquet(opts, buffer, len);
        return new Table(columns);
    }

    /**
     * Return the {@link ColumnVector} at the specified index. The caller is responsible to close it once done to free
     * resources
     */
    public ColumnVector getColumn(int index) {
        assert index < columnVectors.length;
        columnVectors[index].incRefCount();
        return columnVectors[index];
    }

    public final long getRows() {
        return rows;
    }

    public final int getNumberOfColumns() {
        return columnVectors.length;
    }

    @Override
    public void close() {
        if (cudfTable != null) {
            cudfTable.close();
        }
        for (int i = 0 ; i < columnVectors.length ; i++) {
            columnVectors[i].close();
            columnVectors[i] = null;
        }
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
            assert joinIndicesArray[i] >= 0 && joinIndicesArray[i] < columnVectors.length :
                    "join index is out of range 0 <= " + joinIndicesArray[i] + " < " + columnVectors.length;
        }
        return new JoinColumns(this, joinIndicesArray);
    }

    @Override
    public String toString() {
        return "Table{" +
                "columnVectors=" + Arrays.toString(columnVectors) +
                ", cudfTable=" + cudfTable +
                ", rows=" + rows +
                '}';
    }

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
            CudfColumn[] columns = CudfTable.leftJoin(this.table.cudfTable, indices, rightJoinIndices.table.cudfTable, rightJoinIndices.indices);
            return new Table(columns);
        }
    }

    /**
     * Create a table on the GPU with data from the CPU.  This is not fast and intended mostly for
     * tests.
     */
    public static final class TestBuilder {
        private final List<DType> types = new ArrayList<>();
        private final List<Object> typeErasedData = new ArrayList<>();

        public TestBuilder column(Byte... values) {
            types.add(DType.INT8);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder column(Short... values) {
            types.add(DType.INT16);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder column(Integer... values) {
            types.add(DType.INT32);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder column(Long... values) {
            types.add(DType.INT64);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder column(Float... values) {
            types.add(DType.FLOAT32);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder column(Double... values) {
            types.add(DType.FLOAT64);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder date32Column(Integer... values) {
            types.add(DType.DATE32);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder date64Column(Long... values) {
            types.add(DType.DATE64);
            typeErasedData.add(values);
            return this;
        }

        public TestBuilder timestampColumn(Long... values) {
            types.add(DType.TIMESTAMP);
            typeErasedData.add(values);
            return this;
        }

        private static ColumnVector from(DType type, Object dataArray) {
            ColumnVector ret;
            switch(type) {
                case INT8:
                    ret = ColumnVector.buildBoxed((Byte[]) dataArray);
                    break;
                case INT16:
                    ret = ShortColumnVector.buildBoxed((Short[]) dataArray);
                    break;
                case INT32:
                    ret = IntColumnVector.buildBoxed((Integer[]) dataArray);
                    break;
                case INT64:
                    ret = LongColumnVector.buildBoxed((Long[]) dataArray);
                    break;
                case DATE32:
                    ret = Date32ColumnVector.buildBoxed((Integer[]) dataArray);
                    break;
                case DATE64:
                    ret = Date64ColumnVector.buildBoxed((Long[]) dataArray);
                    break;
                case TIMESTAMP:
                    ret = TimestampColumnVector.buildBoxed((Long[]) dataArray);
                    break;
                case FLOAT32:
                    ret = FloatColumnVector.buildBoxed((Float[]) dataArray);
                    break;
                case FLOAT64:
                    ret = DoubleColumnVector.buildBoxed((Double[]) dataArray);
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
                    columns.add(from(types.get(i), typeErasedData.get(i)));
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
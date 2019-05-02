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

import java.util.Arrays;

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
        final long rows = columnVectors[0].rows;

        for (ColumnVector columnVector : columnVectors) {
            assert (null != columnVector) : "ColumnVectors can't be null";
            assert (rows == columnVector.rows) : "All columns should have the same number of rows";
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

    /**
     * Orders the table using the sortkeys returning a new allocated table. The caller is responsible for cleaning up
     * the {@link ColumnVector} returned as part of the output {@link Table}
     *
     * @param sortKeysIndices - indices of columnVectors to sort the table on.
     * @param isDescending - boolean array representing sortorder. (false = ascending)
     * @param areNullsSmallest - represents if nulls are to be considered smaller than non-nulls.
     * @return Sorted Table
     */
    public Table orderBy(int[] sortKeysIndices, boolean[] isDescending, boolean areNullsSmallest) {
        assert sortKeysIndices.length <= columnVectors.length;
        assert sortKeysIndices.length == isDescending.length;
        Table outputTable = Table.newOutputTable(this.columnVectors);
        cudfTable.gdfOrderBy(sortKeysIndices, isDescending, outputTable.cudfTable, areNullsSmallest);
        return outputTable;
    }

    private static Table newOutputTable(ColumnVector[] inputColumnVectors) {
        ColumnVector[] outputColumnVectors = new ColumnVector[inputColumnVectors.length];
        for (int i = 0 ; i < inputColumnVectors.length ; i++) {
            outputColumnVectors[i] = ColumnVector.newOutputVector(inputColumnVectors[i].rows,
                                                inputColumnVectors[i].hasValidityVector(), inputColumnVectors[i].type);
        }
        return new Table(outputColumnVectors);
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

    @Override
    public String toString() {
        return "Table{" +
                "columnVectors=" + Arrays.toString(columnVectors) +
                ", cudfTable=" + cudfTable +
                ", rows=" + rows +
                '}';
    }
}
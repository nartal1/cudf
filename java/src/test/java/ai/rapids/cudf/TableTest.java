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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TableTest {

    static {
        NativeDepsLoader.loadNativeDeps();
    }

    @Test
    void testOrderBy() {
        System.loadLibrary("cudfjni");

        try (
                IntColumnVector sortKeys = IntColumnVector.build(5, (IntColumnVector.Builder b) ->
                        {
                            b.append(5);
                            b.append(4);
                            b.append(3);
                            b.append(2);
                            b.append(1);
                        });
                IntColumnVector values = IntColumnVector.build(5, Range.appendInts(1, 10, 2))
        ) {
            sortKeys.toDeviceBuffer();
            values.toDeviceBuffer();
            try (Table table = new Table(new ColumnVector[]{sortKeys, values})) {
                Table sortedTable = table.orderBy(new int[]{0}, new boolean[]{false}, true);
                assertEquals(sortKeys.rows, sortedTable.getRows());
                IntColumnVector sortedKeys = (IntColumnVector) sortedTable.getColumn(0);
                IntColumnVector sortedValues = (IntColumnVector) sortedTable.getColumn(1);
                assertEquals(sortedKeys.rows, sortedValues.getRows());
                sortedKeys.toHostBuffer();
                sortedValues.toHostBuffer();
                for (int i = 0, sortedKey = 1, sortedValue = 9; i < sortedKeys.rows; i++, sortedKey++, sortedValue -= 2) {
                    assertEquals(sortedKey, sortedKeys.get(i));
                    assertEquals(sortedValue, sortedValues.get(i));
                }
            }
        }
    }

    @Test
    void testTableCreationIncreasesRefCount() {
        NativeDepsLoader.libraryLoaded();
        //tests the Table increases the refcount on column vectors
        assertThrows(IllegalStateException.class, () -> {
            try (IntColumnVector v1 = IntColumnVector.build(5, Range.appendInts(5));
                 IntColumnVector v2 = IntColumnVector.build(5, Range.appendInts(5))) {
                v1.toDeviceBuffer();
                v2.toDeviceBuffer();
                assertDoesNotThrow(() -> {
                    try (Table t = new Table(new ColumnVector[]{v1, v2})) {
                        v1.close();
                        v2.close();
                    }
                });
            }
        });
    }

    @Test
    void testGetColumnIncreasesRefCount() {
        NativeDepsLoader.libraryLoaded();
        assertDoesNotThrow(() -> {
            try (IntColumnVector v1 = IntColumnVector.build(5, Range.appendInts(5));
                 IntColumnVector v2 = IntColumnVector.build(5, Range.appendInts(5))) {
                v1.toDeviceBuffer();
                v2.toDeviceBuffer();
                try (Table t = new Table(new ColumnVector[]{v1, v2})) {
                    ColumnVector vector1 = t.getColumn(0);
                    ColumnVector vector2 = t.getColumn(1);
                    vector1.close();
                    vector2.close();
                }
            }
        });
    }

    @Test
    void testGetRows() {
        NativeDepsLoader.libraryLoaded();
        try (IntColumnVector v1 = IntColumnVector.build(5, Range.appendInts(5));
             IntColumnVector v2 = IntColumnVector.build(5, Range.appendInts(5))) {
                v1.toDeviceBuffer();
                v2.toDeviceBuffer();
                try (Table t = new Table(new ColumnVector[]{v1, v2})) {
                    assertEquals(5, t.getRows());
            }
        }
    }

    @Test
    void testSettingNullVectors() {
        assertThrows(AssertionError.class, () -> new Table(null));
    }

    @Test
    void testAllRowsSize() {
        NativeDepsLoader.libraryLoaded();
        try (IntColumnVector v1 = IntColumnVector.build(4, Range.appendInts(4));
             IntColumnVector v2 = IntColumnVector.build(5, Range.appendInts(5))) {
            v1.toDeviceBuffer();
            v2.toDeviceBuffer();
            assertThrows(AssertionError.class, () -> {try (Table t = new Table(new ColumnVector[]{v1, v2})) {}});
        }
    }

    @Test
    void testGetNumberOfColumns() {
        NativeDepsLoader.libraryLoaded();
        try (IntColumnVector v1 = IntColumnVector.build(5, Range.appendInts(5));
             IntColumnVector v2 = IntColumnVector.build(5, Range.appendInts(5))) {
            v1.toDeviceBuffer();
            v2.toDeviceBuffer();
            try (Table t = new Table(new ColumnVector[]{v1, v2})) {
                assertEquals(2, t.getNumberOfColumns());
            }
        }
    }
}

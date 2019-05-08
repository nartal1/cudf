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

import java.io.File;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class TableTest {

    @Test
    void testOrderBy() {
        assumeTrue(NativeDepsLoader.libraryLoaded());
        try (
                IntColumnVector sortKeys1 = IntColumnVector.build(5, (IntColumnVector.Builder b) ->
                {
                    b.append(5);
                    b.append(3);
                    b.append(3);
                    b.append(1);
                    b.append(1);
                });
                IntColumnVector sortKeys2 = IntColumnVector.build(5, (IntColumnVector.Builder b) ->
                {
                    b.append(5);
                    b.append(3);
                    b.append(4);
                    b.append(1);
                    b.append(2);
                });
                IntColumnVector values = IntColumnVector.build(5, Range.appendInts(1, 10, 2))
        ) {
            sortKeys1.toDeviceBuffer();
            sortKeys2.toDeviceBuffer();
            values.toDeviceBuffer();
            try (Table table = new Table(new ColumnVector[]{sortKeys1, sortKeys2, values});
                Table sortedTable = table.orderBy(true, Table.asc(0), Table.desc(1))) {
                assertEquals(sortKeys1.rows, sortedTable.getRows());
                IntColumnVector sortedKeys1 = (IntColumnVector) sortedTable.getColumn(0);
                IntColumnVector sortedKeys2 = (IntColumnVector) sortedTable.getColumn(1);
                IntColumnVector sortedValues = (IntColumnVector) sortedTable.getColumn(2);
                assertEquals(sortedKeys2.rows, sortedValues.getRows());
                sortedKeys2.toHostBuffer();
                sortedValues.toHostBuffer();
                sortedKeys1.toHostBuffer();
                int[] expectedSortedKeys1 = {1,1,3,3,5};
                int[] expectedSortedKeys2 = {2,1,4,3,5};
                int[] expectedValues = {9,7,5,3,1};
                for (int i = 0 ; i < sortedKeys2.rows; i++) {
                    assertEquals(expectedSortedKeys1[i], sortedKeys1.get(i));
                    assertEquals(expectedSortedKeys2[i], sortedKeys2.get(i));
                    assertEquals(expectedValues[i], sortedValues.get(i));
                }
            }

            try (Table table = new Table(new ColumnVector[]{sortKeys1, sortKeys2, values});
                Table sortedTable = table.orderBy(true, Table.desc(0), Table.desc(1))) {
                assertEquals(sortKeys1.rows, sortedTable.getRows());
                IntColumnVector sortedKeys1 = (IntColumnVector) sortedTable.getColumn(0);
                IntColumnVector sortedKeys2 = (IntColumnVector) sortedTable.getColumn(1);
                IntColumnVector sortedValues = (IntColumnVector) sortedTable.getColumn(2);
                assertEquals(sortedKeys2.rows, sortedValues.getRows());
                sortedKeys2.toHostBuffer();
                sortedValues.toHostBuffer();
                sortedKeys1.toHostBuffer();
                int[] expectedSortedKeys1 = {5,3,3,1,1};
                int[] expectedSortedKeys2 = {5,4,3,2,1};
                int[] expectedValues = {1,5,3,9,7};
                for (int i = 0 ; i < sortedKeys2.rows; i++) {
                    assertEquals(expectedSortedKeys1[i], sortedKeys1.get(i));
                    assertEquals(expectedSortedKeys2[i], sortedKeys2.get(i));
                    assertEquals(expectedValues[i], sortedValues.get(i));
                }
            }
        }
    }

    @Test
    void testTableCreationIncreasesRefCount() {
        assumeTrue(NativeDepsLoader.libraryLoaded());
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
        assumeTrue(NativeDepsLoader.libraryLoaded());
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
        assumeTrue(NativeDepsLoader.libraryLoaded());
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
        ColumnVector[] columnVectors = null;
        assertThrows(AssertionError.class, () -> new Table(columnVectors));
    }

    @Test
    void testAllRowsSize() {
        assumeTrue(NativeDepsLoader.libraryLoaded());
        try (IntColumnVector v1 = IntColumnVector.build(4, Range.appendInts(4));
             IntColumnVector v2 = IntColumnVector.build(5, Range.appendInts(5))) {
            v1.toDeviceBuffer();
            v2.toDeviceBuffer();
            assertThrows(AssertionError.class, () -> {try (Table t = new Table(new ColumnVector[]{v1, v2})) {}});
        }
    }

    @Test
    void testGetNumberOfColumns() {
        assumeTrue(NativeDepsLoader.libraryLoaded());
        try (IntColumnVector v1 = IntColumnVector.build(5, Range.appendInts(5));
             IntColumnVector v2 = IntColumnVector.build(5, Range.appendInts(5))) {
            v1.toDeviceBuffer();
            v2.toDeviceBuffer();
            try (Table t = new Table(new ColumnVector[]{v1, v2})) {
                assertEquals(2, t.getNumberOfColumns());
            }
        }
    }

    @Test
    void testReadCSVPrune() {
        Schema schema = Schema.builder()
                .column(DType.INT32, "A")
                .column(DType.FLOAT64, "B")
                .column(DType.INT64, "C")
                .build();
        CSVOptions opts = CSVOptions.builder()
                .includeColumn("A")
                .includeColumn("B")
                .build();
        try (Table table = Table.readCSV(schema, opts, new File("./src/test/resources/simple.csv"))) {
            long rows = table.getRows();
            assertEquals(10, rows);
            int len = table.getNumberOfColumns();
            assertEquals(2, len);

            double[] doubleData = new double[] {110.0,111.0,112.0,113.0,114.0,115.0,116.0,117.0,118.2,119.8};
            int[] intData = new int[] {0,1,2,3,4,5,6,7,8,9};
            try (IntColumnVector intOutput = (IntColumnVector) table.getColumn(0);
                 DoubleColumnVector doubleOutput = (DoubleColumnVector) table.getColumn(1)) {
                intOutput.toHostBuffer();
                doubleOutput.toHostBuffer();
                for (int i = 0; i < rows; i++) {
                    assertEquals(intData[i], intOutput.get(i));
                    assertEquals(doubleData[i], doubleOutput.get(i));
                }
            }
        }
    }

    @Test
    void testReadCSVBuffer() {
        Schema schema = Schema.builder()
                .column(DType.INT32, "A")
                .column(DType.FLOAT64, "B")
                .column(DType.INT64, "C")
                .build();
        CSVOptions opts = CSVOptions.builder()
                .includeColumn("A")
                .includeColumn("B")
                .build();
        byte [] data =
                ("0,110.0,120\n" +
                        "1,111.0,121\n" +
                        "2,112.0,122\n" +
                        "3,113.0,123\n" +
                        "4,114.0,124\n" +
                        "5,115.0,125\n" +
                        "6,116.0,126\n" +
                        "7,117.0,127\n" +
                        "8,118.2,128\n" +
                        "9,119.8,129").getBytes(StandardCharsets.UTF_8);
        try (Table table = Table.readCSV(schema, opts, data, data.length)) {
            long rows = table.getRows();
            assertEquals(10, rows);
            int len = table.getNumberOfColumns();
            assertEquals(2, len);

            double[] doubleData = new double[] {110.0,111.0,112.0,113.0,114.0,115.0,116.0,117.0,118.2,119.8};
            int[] intData = new int[] {0,1,2,3,4,5,6,7,8,9};
            try (IntColumnVector intOutput = (IntColumnVector) table.getColumn(0);
                 DoubleColumnVector doubleOutput = (DoubleColumnVector) table.getColumn(1)) {
                intOutput.toHostBuffer();
                doubleOutput.toHostBuffer();
                for (int i = 0; i < rows; i++) {
                    assertEquals(intData[i], intOutput.get(i));
                    assertEquals(doubleData[i], doubleOutput.get(i));
                }
            }
        }
    }

    @Test
    void testReadCSV() {
        Schema schema = Schema.builder()
                .column(DType.INT32, "A")
                .column(DType.FLOAT64, "B")
                .column(DType.INT64, "C")
                .build();
        try (Table table = Table.readCSV(schema, new File("./src/test/resources/simple.csv"))) {
            long rows = table.getRows();
            assertEquals(10, rows);
            int len = table.getNumberOfColumns();
            assertEquals(3, len);

            int[] intData = new int[] {0,1,2,3,4,5,6,7,8,9};
            double[] doubleData = new double[] {110.0,111.0,112.0,113.0,114.0,115.0,116.0,117.0,118.2,119.8};
            int[] LongData = new int[] {120,121,122,123,124,125,126,127,128,129};
            try (IntColumnVector intOutput = (IntColumnVector) table.getColumn(0);
                 DoubleColumnVector doubleOutput = (DoubleColumnVector) table.getColumn(1);
                 LongColumnVector longOutput = (LongColumnVector) table.getColumn(2)) {
                intOutput.toHostBuffer();
                doubleOutput.toHostBuffer();
                longOutput.toHostBuffer();
                for (int i = 0; i < rows; i++) {
                    assertEquals(intData[i], intOutput.get(i));
                    assertEquals(doubleData[i], doubleOutput.get(i), 0.1);
                    assertEquals(LongData[i], longOutput.get(i));
                }
            }
        }
    }

    @Test
    void testLeftJoin() {
        int length = 10;
        try (IntColumnVector l0 = IntColumnVector.build(length, (b)->b.append(360).append(326).append(254).append(306)
                                                                    .append(109).append(361).append(251).append(335)
                                                                    .append(301).append(317));
             IntColumnVector l1 = IntColumnVector.build(length, (b) -> b.append(323).append(172).append(11).append(243)
                                                                    .append(57).append(143).append(305).append(95)
                                                                    .append(147).append(58));
             IntColumnVector r0 = IntColumnVector.build(length, (b)-> b.append(306).append(301).append(360).append(109)
                                                                    .append(335).append(254).append(317).append(361)
                                                                    .append(251).append(326));
             IntColumnVector r1 = IntColumnVector.build(length, (b)-> b.append(84).append(257).append(80).append(93)
                                                                    .append(231).append(193).append(22).append(12)
                                                                    .append(186).append(184))) {
            l0.toDeviceBuffer();
            l1.toDeviceBuffer();
            r0.toDeviceBuffer();
            r1.toDeviceBuffer();
            try (Table leftTable = new Table(new ColumnVector[]{l0, l1});
                 Table rightTable = new Table(new ColumnVector[]{r0, r1})) {

                try (Table joinedTable = leftTable.joinColumns(0).leftJoin(rightTable.joinColumns(new int[]{0}));
                     Table orderedJoinedTable = joinedTable.orderBy(true, Table.asc(1))) {
                    long rows = orderedJoinedTable.getRows();
                    int cols = orderedJoinedTable.getNumberOfColumns();
                    assertEquals(3, cols);
                    IntColumnVector out0 = (IntColumnVector) orderedJoinedTable.getColumn(0);
                    IntColumnVector out1 = (IntColumnVector) orderedJoinedTable.getColumn(1);
                    IntColumnVector out2 = (IntColumnVector) orderedJoinedTable.getColumn(2);
                    out0.toHostBuffer();
                    out1.toHostBuffer();
                    out2.toHostBuffer();

                    int[] expectedOut0 = new int[]{57,305,11,147,243,58,172,95,323,143};
                    int[] expectedOut1 = new int[]{109,251,254,301,306,317,326,335,360,361};
                    int[] expectedOut2 = new int[]{93,186,193,257,84,22,184,231,80,12};
                    for (int i = 0; i < rows; i++) {
                        assertEquals(expectedOut0[i], out0.get(i));
                        assertEquals(expectedOut1[i], out1.get(i));
                        assertEquals(expectedOut2[i], out2.get(i));
                    }
                }
            }
        }
    }

    @Test
    void testLeftJoinWithNulls() {
        int length = 10;
        try (IntColumnVector l0 = IntColumnVector.build(length, (b)-> b.append(2).append(3).append(9).append(0).append(1)
                                                                    .append(7).append(4).append(6).append(5).append(8));
             IntColumnVector l1 = IntColumnVector.build(length, (b)-> b.append(102).append(103).append(19).append(100).append(101)
                                                                    .append(4).append(104).append(1).append(3).append(1));
             IntColumnVector r0 = IntColumnVector.build(length, (b)-> b.append(6).append(5).append(9).append(8).append(10).append(32));
             IntColumnVector r1 = IntColumnVector.build(length, (b)-> b.append(199).append(211).append(321).append(1233).append(33).append(392))) {
            l0.toDeviceBuffer();
            l1.toDeviceBuffer();
            r0.toDeviceBuffer();
            r1.toDeviceBuffer();
            try (Table leftTable = new Table(new ColumnVector[]{l0, l1});
                 Table rightTable = new Table(new ColumnVector[]{r0, r1})) {

                try (Table joinedTable = leftTable.joinColumns(0).leftJoin(rightTable.joinColumns(0));
                    Table orderedJoinedTable = joinedTable.orderBy(true, Table.asc(1))) {
                    int cols = orderedJoinedTable.getNumberOfColumns();
                    assertEquals(3, cols);
                    IntColumnVector out0 = (IntColumnVector) orderedJoinedTable.getColumn(0);
                    IntColumnVector out1 = (IntColumnVector) orderedJoinedTable.getColumn(1);
                    IntColumnVector out2 = (IntColumnVector) orderedJoinedTable.getColumn(2);
                    out0.toHostBuffer();
                    out1.toHostBuffer();
                    out2.toHostBuffer();
                    long rows = orderedJoinedTable.getRows();
                    int[] expectedOut0 = new int[]{100, 101, 102, 103, 104, 3, 1, 4, 1, 19};
                    int[] expectedOut2 = new int[]{0,0,0,0,0, 211, 199, 0, 1233, 321};
                    for (int i = 0; i < rows; i++) {
                        assertEquals(expectedOut0[i], out0.get(i));
                        assertEquals(i, out1.get(i));
                        assertEquals(expectedOut2[i], out2.get(i));
                    }
                }
            }
        }
    }
}
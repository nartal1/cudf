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
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class TableTest {
    private static final File TEST_PARQUET_FILE = new File("src/test/resources/acq.parquet");

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
            try (Table table = new Table(new ColumnVector[]{sortKeys1, sortKeys2, values})) {
                Table sortedTable = table.orderBy(true, Table.asc(0), Table.desc(1));
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

            try (Table table = new Table(new ColumnVector[]{sortKeys1, sortKeys2, values})) {
                Table sortedTable = table.orderBy(true, Table.desc(0), Table.desc(1));
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
    void testReadParquet() {
        ParquetOptions opts = ParquetOptions.builder()
                .includeColumn("loan_id")
                .includeColumn("zip")
                .includeColumn("num_units")
                .build();
        try (Table table = Table.readParquet(opts, TEST_PARQUET_FILE)) {
            long rows = table.getRows();
            assertEquals(1000, rows);
            int len = table.getNumberOfColumns();
            assertEquals(3, len);

            try (LongColumnVector loadId = (LongColumnVector) table.getColumn(0);
                 IntColumnVector zip = (IntColumnVector)table.getColumn(1);
                 IntColumnVector numUnits = (IntColumnVector)table.getColumn(2)) {
                // Empty
            }
        }
    }

    @Test
    void testReadParquetBuffer() throws IOException {
        ParquetOptions opts = ParquetOptions.builder()
                .includeColumn("loan_id")
                .includeColumn("coborrow_credit_score")
                .includeColumn("borrower_credit_score")
                .build();

        byte [] buffer = new byte[(int)TEST_PARQUET_FILE.length() + 1024];
        int bufferLen = 0;
        try (FileInputStream in = new FileInputStream(TEST_PARQUET_FILE)) {
            bufferLen = in.read(buffer);
        }
        try (Table table = Table.readParquet(opts, buffer, bufferLen)) {
            long rows = table.getRows();
            assertEquals(1000, rows);
            int len = table.getNumberOfColumns();
            assertEquals(3, len);

            try (LongColumnVector loadId = (LongColumnVector) table.getColumn(0);
                 DoubleColumnVector cocs = (DoubleColumnVector)table.getColumn(1);
                 DoubleColumnVector bcs = (DoubleColumnVector)table.getColumn(2)) {
                // Empty
            }
        }
    }

    @Test
    void testReadParquetFull() {
        try (Table table = Table.readParquet(TEST_PARQUET_FILE)) {
            long rows = table.getRows();
            assertEquals(1000, rows);
            int len = table.getNumberOfColumns();
            assertEquals(26, len);

            DType [] expectedTypes = new DType[] {
                    DType.INT64, // loan_id
                    DType.INT32, // orig_channel
                    DType.FLOAT64, // orig_interest_rate
                    DType.INT32, // orig_upb
                    DType.INT32, // orig_loan_term
                    DType.DATE32, // orig_date
                    DType.DATE32, // first_pay_date
                    DType.FLOAT64, // orig_ltv
                    DType.FLOAT64, // orig_cltv
                    DType.FLOAT64, // num_borrowers
                    DType.FLOAT64, // dti
                    DType.FLOAT64, // borrower_credit_score
                    DType.INT32, // first_home_buyer
                    DType.INT32, // loan_purpose
                    DType.INT32, // property_type
                    DType.INT32, // num_units
                    DType.INT32, // occupancy_status
                    DType.INT32, // property_state
                    DType.INT32, // zip
                    DType.FLOAT64, // mortgage_insurance_percent
                    DType.INT32, // product_type
                    DType.FLOAT64, // coborrow_credit_score
                    DType.FLOAT64, // mortgage_insurance_type
                    DType.INT32, // relocation_mortgage_indicator
                    DType.INT32, // quarter
                    DType.INT32 // seller_id
            };

            for (int i = 0; i < len; i++) {
                try (ColumnVector vec = table.getColumn(i)) {
                    assertEquals(expectedTypes[i], vec.getType(), "Types don't match at " + i);
                }
            }
        }
    }
}
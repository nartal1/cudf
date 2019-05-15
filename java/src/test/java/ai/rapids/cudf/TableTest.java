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
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class TableTest {
    private static final File TEST_PARQUET_FILE = new File("src/test/resources/acq.parquet");

    public static void assertTablesAreEqual(Table expected, Table table) {
        assertEquals(expected.getNumberOfColumns(), table.getNumberOfColumns());
        assertEquals(expected.getRows(), table.getRows());
        for (int col = 0; col < expected.getNumberOfColumns(); col++) {
            try (ColumnVector expect = expected.getColumn(col);
                 ColumnVector cv = table.getColumn(col)) {
                assertEquals(expect.getType(), cv.getType(), "Column " + col);
                assertEquals(expect.getRows(), cv.getRows(), "Column " + col); // Yes this might be redundant
                assertEquals(expect.getNullCount(), cv.getNullCount(), "Column " + col);
                expect.ensureOnHost();
                cv.ensureOnHost();
                DType type = expect.getType();
                for (long row = 0; row < expect.getRows(); row++) {
                    assertEquals(expect.isNull(row), cv.isNull(row), "Column " + col + " Row " + row);
                    if (!expect.isNull(row)) {
                        switch(type) {
                            case INT8:
                                assertEquals(expect.getByte(row), cv.getByte(row),
                                        "Column " + col + " Row " + row);
                                break;
                            case INT16:
                                assertEquals(((ShortColumnVector)expect).get(row),
                                        ((ShortColumnVector)cv).get(row),
                                        "Column " + col + " Row " + row);
                                break;
                            case INT32:
                                assertEquals(expect.getInt(row), cv.getInt(row),
                                        "Column " + col + " Row " + row);
                                break;
                            case INT64:
                                assertEquals(((LongColumnVector)expect).get(row),
                                        ((LongColumnVector)cv).get(row),
                                        "Column " + col + " Row " + row);
                                break;
                            case FLOAT32:
                                assertEquals(expect.getFloat(row), cv.getFloat(row), 0.0001,
                                        "Column " + col + " Row " + row);
                                break;
                            case FLOAT64:
                                assertEquals(expect.getDouble(row), cv.getDouble(row), 0.0001,
                                        "Column " + col + " Row " + row);
                                break;
                            case DATE32:
                                assertEquals(expect.getInt(row), cv.getInt(row),
                                        "Column " + col + " Row " + row);
                                break;
                            case DATE64:
                                assertEquals(expect.getLong(row), cv.getLong(row),
                                        "Column " + col + " Row " + row);
                                break;
                            case TIMESTAMP:
                                assertEquals(((TimestampColumnVector)expect).get(row),
                                        ((TimestampColumnVector)cv).get(row),
                                        "Column " + col + " Row " + row);
                                break;
                            default:
                                throw new IllegalArgumentException(type + " is not supported yet");
                        }
                    }
                }
            }
        }
    }


    public static void assertTableTypes(DType [] expectedTypes, Table t) {
        int len = t.getNumberOfColumns();
        assertEquals(expectedTypes.length, len);
        for (int i = 0; i < len; i++) {
            try (ColumnVector vec = t.getColumn(i)) {
                DType type = vec.getType();
                assertEquals(expectedTypes[i], type, "Types don't match at " + i);
                // TODO when done delete this...
                Class c = ColumnVector.class;
                switch(type) {
                    case INT8:
                    case INT32:
                    case FLOAT32:
                    case FLOAT64:
                    case DATE32:
                    case DATE64:
                        // Ignored
                        break;
                    case INT16:
                        c = ShortColumnVector.class;
                        break;
                    case INT64:
                        c = LongColumnVector.class;
                        break;
                    case TIMESTAMP:
                        c = TimestampColumnVector.class;
                        break;
                    default:
                        throw new IllegalArgumentException(type + " is not supported yet");
                }
                assertTrue(c.isAssignableFrom(vec.getClass()), "Expected type " + c + " but found " + vec.getClass());
            }
        }
    }

    @Test
    void testOrderByAD() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (Table table = new Table.TestBuilder()
                .column(    5,    3,    3,    1,    1)
                .column(    5,    3,    4,    1,    2)
                .column(    1,    3,    5,    7,    9)
                .build();
             Table expected = new Table.TestBuilder()
                     .column(   1,    1,    3,    3,    5)
                     .column(   2,    1,    4,    3,    5)
                     .column(   9,    7,    5,    3,    1)
                     .build();
             Table sortedTable = table.orderBy(false, Table.asc(0), Table.desc(1))) {
            assertTablesAreEqual(expected, sortedTable);
        }
    }

    @Test
    void testOrderByDD() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (Table table = new Table.TestBuilder()
                .column(    5,    3,    3,    1,    1)
                .column(    5,    3,    4,    1,    2)
                .column(    1,    3,    5,    7,    9)
                .build();
             Table expected = new Table.TestBuilder()
                     .column(   5,    3,    3,    1,    1)
                     .column(   5,    4,    3,    2,    1)
                     .column(   1,    5,    3,    9,    7)
                     .build();
             Table sortedTable = table.orderBy(false, Table.desc(0), Table.desc(1))) {
            assertTablesAreEqual(expected, sortedTable);
        }
    }

    @Test
    void testOrderByWithNulls() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (Table table = new Table.TestBuilder()
                .column(    5, null,    3,    1,    1)
                .column(    5,    3,    4, null, null)
                .column(    1,    3,    5,    7,    9)
                .build();
             Table expected = new Table.TestBuilder()
                 .column(   1,    1,    3,    5, null)
                 .column(null, null,    4,    5,    3)
                 .column(   7,    9,    5,    1,    3)
                 .build();
            Table sortedTable = table.orderBy(false, Table.asc(0), Table.desc(1))) {
            assertTablesAreEqual(expected, sortedTable);
        }
    }

    @Test
    void testTableCreationIncreasesRefCount() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        //tests the Table increases the refcount on column vectors
        assertThrows(IllegalStateException.class, () -> {
            try (ColumnVector v1 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
                 ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5))) {
                v1.ensureOnDevice();
                v2.ensureOnDevice();
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
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        assertDoesNotThrow(() -> {
            try (ColumnVector v1 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
                 ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5))) {
                v1.ensureOnDevice();
                v2.ensureOnDevice();
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
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (ColumnVector v1 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
             ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5))) {
            v1.ensureOnDevice();
            v2.ensureOnDevice();
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
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (ColumnVector v1 = ColumnVector.build(DType.INT32, 4, Range.appendInts(4));
             ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5))) {
            v1.ensureOnDevice();
            v2.ensureOnDevice();
            assertThrows(AssertionError.class, () -> {
                try (Table t = new Table(new ColumnVector[]{v1, v2})) {
                }
            });
        }
    }

    @Test
    void testGetNumberOfColumns() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (ColumnVector v1 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
             ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5))) {
            v1.ensureOnDevice();
            v2.ensureOnDevice();
            try (Table t = new Table(new ColumnVector[]{v1, v2})) {
                assertEquals(2, t.getNumberOfColumns());
            }
        }
    }

    @Test
    void testReadCSVPrune() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        Schema schema = Schema.builder()
                .column(DType.INT32, "A")
                .column(DType.FLOAT64, "B")
                .column(DType.INT64, "C")
                .build();
        CSVOptions opts = CSVOptions.builder()
                .includeColumn("A")
                .includeColumn("B")
                .build();
        try (Table expected = new Table.TestBuilder()
                .column(    0,     1,     2,     3,     4,     5,     6,     7,     8,     9)
                .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.2, 119.8)
                .build();
                Table table = Table.readCSV(schema, opts, new File("./src/test/resources/simple.csv"))) {
            assertTablesAreEqual(expected, table);
        }
    }

    @Test
    void testReadCSVBufferInferred() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        CSVOptions opts = CSVOptions.builder()
                .includeColumn("A")
                .includeColumn("B")
                .hasHeader()
                .withComment('#')
                .build();
        byte[] data = ("A,B,C\n" +
                "0,110.0,120'\n" +
                "#0.5,1.0,200\n" +
                "1,111.0,121\n" +
                "2,112.0,122\n" +
                "3,113.0,123\n" +
                "4,114.0,124\n" +
                "5,115.0,125\n" +
                "6,116.0,126\n" +
                "7,117.0,127\n" +
                "8,118.2,128\n" +
                "9,119.8,129").getBytes(StandardCharsets.UTF_8);
        try (Table expected = new Table.TestBuilder()
                .column(   0L,    1L,    2L,    3L,    4L,    5L,    6L,    7L,    8L,    9L)
                .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.2, 119.8)
                .build();
             Table table = Table.readCSV(Schema.INFERRED, opts, data, data.length)) {
            assertTablesAreEqual(expected, table);
        }
    }

    @Test
    void testReadCSVBuffer() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        Schema schema = Schema.builder()
                .column(DType.INT32, "A")
                .column(DType.FLOAT64, "B")
                .column(DType.INT64, "C")
                .build();
        CSVOptions opts = CSVOptions.builder()
                .includeColumn("A")
                .includeColumn("B")
                .hasHeader()
                .withDelim('|')
                .withQuote('\'')
                .withNullValue("NULL")
                .build();
        byte[] data = ("A|B|C\n" +
                "'0'|'110.0'|'120'\n" +
                "1|111.0|121\n" +
                "2|112.0|122\n" +
                "3|113.0|123\n" +
                "4|114.0|124\n" +
                "5|115.0|125\n" +
                "6|116.0|126\n" +
                "7|NULL|127\n" +
                "8|118.2|128\n" +
                "9|119.8|129").getBytes(StandardCharsets.UTF_8);
        try (Table expected = new Table.TestBuilder()
                .column(    0,     1,     2,     3,     4,     5,     6,     7,     8,     9)
                .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0,  null, 118.2, 119.8)
                .build();
             Table table = Table.readCSV(schema, opts, data, data.length)) {
            assertTablesAreEqual(expected, table);
        }
    }

    @Test
    void testReadCSV() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        Schema schema = Schema.builder()
                .column(DType.INT32, "A")
                .column(DType.FLOAT64, "B")
                .column(DType.INT64, "C")
                .build();
        try (Table expected = new Table.TestBuilder()
                .column(    0,     1,     2,     3,     4,     5,     6,     7,     8,     9)
                .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.2, 119.8)
                .column( 120L,  121L,  122L,  123L,  124L,  125L,  126L,  127L,  128L,  129L)
                .build();
                Table table = Table.readCSV(schema, new File("./src/test/resources/simple.csv"))) {
            assertTablesAreEqual(expected, table);
        }
    }

    @Test
    void testReadParquet() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        ParquetOptions opts = ParquetOptions.builder()
                .includeColumn("loan_id")
                .includeColumn("zip")
                .includeColumn("num_units")
                .build();
        try (Table table = Table.readParquet(opts, TEST_PARQUET_FILE)) {
            long rows = table.getRows();
            assertEquals(1000, rows);
            assertTableTypes(new DType[] {DType.INT64, DType.INT32, DType.INT32}, table);
        }
    }

    @Test
    void testReadParquetBuffer() throws IOException {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        ParquetOptions opts = ParquetOptions.builder()
                .includeColumn("loan_id")
                .includeColumn("coborrow_credit_score")
                .includeColumn("borrower_credit_score")
                .build();

        byte[] buffer = new byte[(int) TEST_PARQUET_FILE.length() + 1024];
        int bufferLen = 0;
        try (FileInputStream in = new FileInputStream(TEST_PARQUET_FILE)) {
            bufferLen = in.read(buffer);
        }
        try (Table table = Table.readParquet(opts, buffer, bufferLen)) {
            long rows = table.getRows();
            assertEquals(1000, rows);
            assertTableTypes(new DType[] {DType.INT64, DType.FLOAT64, DType.FLOAT64}, table);
        }
    }

    @Test
    void testReadParquetFull() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (Table table = Table.readParquet(TEST_PARQUET_FILE)) {
            long rows = table.getRows();
            assertEquals(1000, rows);

            DType[] expectedTypes = new DType[]{
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

            assertTableTypes(expectedTypes, table);
        }
    }

    @Test
    void testLeftJoinWithNulls() {
        try (Table leftTable = new Table.TestBuilder()
                      .column(  2,    3,    9,    0,    1,    7,    4,    6,    5,    8)
                      .column(102,  103,   19,  100,  101,    4,  104,    1,    3,    1)
                      .build();
             Table rightTable = new Table.TestBuilder()
                     .column(   6,    5,    9,    8,   10,   32)
                     .column( 199,  211,  321, 1233,   33,  392)
                     .build();
             Table expected = new Table.TestBuilder()
                     .column( 100,  101,  102,  103,  104,    3,    1,    4,    1,   19)
                     .column(   0,    1,    2,    3,    4,    5,    6,    7,    8,    9)
                     .column(null, null, null, null, null,  211,  199, null, 1233,  321)
                     .build();
             Table joinedTable = leftTable.joinColumns(0).leftJoin(rightTable.joinColumns(0));
             Table orderedJoinedTable = joinedTable.orderBy(true, Table.asc(1))) {
            assertTablesAreEqual(expected, orderedJoinedTable);
        }
    }

    @Test
    void testLeftJoin() {
        try (Table leftTable = new Table.TestBuilder()
                     .column(360, 326, 254, 306, 109, 361, 251, 335, 301, 317)
                     .column(323, 172,  11, 243,  57, 143, 305,  95, 147,  58)
                     .build();
             Table rightTable = new Table.TestBuilder()
                     .column(306, 301, 360, 109, 335, 254, 317, 361, 251, 326)
                     .column( 84, 257,  80,  93, 231, 193,  22,  12, 186, 184)
                     .build();
             Table joinedTable = leftTable.joinColumns(0).leftJoin(rightTable.joinColumns(new int[]{0}));
             Table orderedJoinedTable = joinedTable.orderBy(true, Table.asc(1));
             Table expected = new Table.TestBuilder()
                     .column( 57, 305,  11, 147, 243,  58, 172,  95, 323, 143)
                     .column(109, 251, 254, 301, 306, 317, 326, 335, 360, 361)
                     .column( 93, 186, 193, 257,  84,  22, 184, 231,  80,  12)
                     .build()) {
            assertTablesAreEqual(expected, orderedJoinedTable);
        }
    }
}
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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class TimestampColumnVectorTest {
    static final long[] TIMES = {-131968727238L,   //'1965-10-26 14:01:12.762'
            1530705600000L,   //'2018-07-04 12:00:00.000'
            1674631932929L};  //'2023-01-25 07:32:12.929'

    @Test
    public void getYear() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());

        try (TimestampColumnVector timestampColumnVector = TimestampColumnVector.build(TIMES)) {
            timestampColumnVector.ensureOnDevice();
            ShortColumnVector result= timestampColumnVector.year();
            result.ensureOnHost();
            assertEquals(1965,result.get(0));
            assertEquals(2018,result.get(1));
            assertEquals(2023,result.get(2));
        }
    }

    @Test
    public void getMonth() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (TimestampColumnVector timestampColumnVector = TimestampColumnVector.build(TIMES)) {
            timestampColumnVector.ensureOnDevice();
            ShortColumnVector result= timestampColumnVector.month();
            result.ensureOnHost();
            assertEquals(10,result.get(0));
            assertEquals(7,result.get(1));
            assertEquals(1,result.get(2));
        }
    }

    @Test
    public void getDay() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (TimestampColumnVector timestampColumnVector = TimestampColumnVector.build(TIMES)) {
            timestampColumnVector.ensureOnDevice();
            try (ShortColumnVector result= timestampColumnVector.day()) {
                result.ensureOnHost();
                assertEquals(26, result.get(0));
                assertEquals(4, result.get(1));
                assertEquals(25, result.get(2));
            }
        }
    }

    @Test
    public void getHour() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (TimestampColumnVector timestampColumnVector = TimestampColumnVector.build(TIMES)) {
            timestampColumnVector.ensureOnDevice();
            try (ShortColumnVector result= timestampColumnVector.hour()) {
                result.ensureOnHost();
                assertEquals(14, result.get(0));
                assertEquals(12, result.get(1));
                assertEquals(7, result.get(2));
            }
        }
    }

    @Test
    public void getMinute() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (TimestampColumnVector timestampColumnVector = TimestampColumnVector.build(TIMES)) {
            timestampColumnVector.ensureOnDevice();
            try (ShortColumnVector result= timestampColumnVector.minute()) {
                result.ensureOnHost();
                assertEquals(1, result.get(0));
                assertEquals(0, result.get(1));
                assertEquals(32, result.get(2));
            }
        }
    }

    @Test
    public void getSecond() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (TimestampColumnVector timestampColumnVector = TimestampColumnVector.build(TIMES)) {
            timestampColumnVector.ensureOnDevice();
            try (ShortColumnVector result= timestampColumnVector.second()) {
                result.ensureOnHost();
                assertEquals(12, result.get(0));
                assertEquals(0, result.get(1));
                assertEquals(12, result.get(2));
            }
        }
    }
}
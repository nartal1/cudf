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
import org.mockito.Mockito;


import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.Mockito.mock;

public class ColumnVectorTest {

    @Test
    void testCudfColumnSize() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        DeviceMemoryBuffer mockDataBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
        DeviceMemoryBuffer mockValidBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);

        try (ColumnVector v0 = new ColumnVector(mockDataBuffer, mockValidBuffer, 0, DType.INT32) {
        }) {
            v0.getCudfColumn();
        }

        try (ColumnVector v1 = new ColumnVector(mockDataBuffer, mockValidBuffer, Long.MAX_VALUE, DType.INT32) {
        }) {
            assertThrows(AssertionError.class, () -> v1.getCudfColumn());
        }
    }

    @Test
    void testCudfColumnFromHostVector() {
        HostMemoryBuffer mockDataBuffer = mock(HostMemoryBuffer.class);
        try (ColumnVector v = new ColumnVector(mockDataBuffer, null, 10, DType.INT32, 0) {}) {
            assertThrows(IllegalStateException.class, () -> v.getCudfColumn());
        }
    }

    @Test
    void testGetNativeAddressFromHostVector() {
        HostMemoryBuffer mockDataBuffer = mock(HostMemoryBuffer.class);
        try (ColumnVector v = new ColumnVector(mockDataBuffer, null, 10, DType.INT32, 0) {}) {
            assertThrows(IllegalStateException.class, () -> v.getNativeCudfColumnAddress());
        }
    }

    @Test
    void testRefCount() {
        DeviceMemoryBuffer mockDataBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
        DeviceMemoryBuffer mockValidBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);

        assertThrows(IllegalStateException.class, () -> {
            try (ColumnVector v2 = new ColumnVector(mockDataBuffer, mockValidBuffer, Long.MAX_VALUE, DType.INT32) {}) {
                v2.close();
            }
        });
    }

    @Test
    void testRefCountLeak() throws InterruptedException {
        assumeTrue(Boolean.getBoolean("ai.rapids.cudf.flaky-tests-enabled"));
        long expectedLeakCount = ColumnVectorCleaner.leakCount.get() + 1;
        DeviceMemoryBuffer mockDataBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
        DeviceMemoryBuffer mockValidBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
        new ColumnVector(mockDataBuffer, mockValidBuffer, Long.MAX_VALUE, DType.INT32) {};
        long maxTime = System.currentTimeMillis() + 10_000;
        long leakNow;
        do {
            System.gc();
            Thread.sleep(50);
            leakNow = ColumnVectorCleaner.leakCount.get();
        } while(leakNow != expectedLeakCount && System.currentTimeMillis() < maxTime);
        assertEquals(expectedLeakCount, ColumnVectorCleaner.leakCount.get());
    }
}

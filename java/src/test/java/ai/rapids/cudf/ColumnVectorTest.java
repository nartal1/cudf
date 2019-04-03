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

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class ColumnVectorTest {

    @Test
    void testCudfColumnSize() {
        DeviceMemoryBuffer mockDataBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
        DeviceMemoryBuffer mockValidBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
        when(mockDataBuffer.getLength()).thenReturn(Long.MAX_VALUE);

        ColumnVector v0 = new ColumnVector(mockDataBuffer, mockValidBuffer, 0, Optional.empty()) {};
        assertThrows(AssertionError.class, () -> v0.getCudfColumn(DType.CUDF_INT32));

        ColumnVector v1 = new ColumnVector(mockDataBuffer, mockValidBuffer, Long.MAX_VALUE, Optional.empty()) {};
        assertThrows(AssertionError.class, () -> v1.getCudfColumn(DType.CUDF_INT32));
    }
}

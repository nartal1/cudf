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

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;

public class ColumnVectorTest {

    @Test
    void testCudfColumnSize() {
        HostMemoryBuffer mockDataBuffer = mock(HostMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
        HostMemoryBuffer mockValidBuffer = mock(HostMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);

        IntColumnVector v2 = IntColumnVector.builderTest(Integer.MAX_VALUE + 1, mockDataBuffer, mockValidBuffer).build();
        try (IntColumnVector v1 = IntColumnVector.build(Integer.MAX_VALUE + 1, (v) -> v.append(3, Integer.MAX_VALUE + 1))) {
            assertThrows(IndexOutOfBoundsException.class, () -> v1.getCudfColumn(DType.CUDF_INT32));
        }
    }

}

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

import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostMemoryBuffer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;


public class HostMemoryBufferImplTest {

    static HostMemoryBuffer hostMemoryBuffer;

    @BeforeEach
    public void clear(){
        hostMemoryBuffer = HostMemoryBuffer.allocate(16);
    }

    @AfterEach
    void free() {
        hostMemoryBuffer.close();
    }

    @Test
    public void testGetInt() {
        long offset = 1;
        hostMemoryBuffer.setInt(offset * DType.CUDF_INT32.sizeInBytes, 2);
        assertEquals(2, hostMemoryBuffer.getInt(offset * DType.CUDF_INT32.sizeInBytes));
    }

    @Test
    public void testGetByte() {
        long offset = 1;
        hostMemoryBuffer.setByte(offset * DType.CUDF_INT8.sizeInBytes, (byte) 2);
        assertEquals((byte) 2, hostMemoryBuffer.getByte(offset * DType.CUDF_INT8.sizeInBytes));

    }

    @Test
    public void testGetLong() {
        long offset = 1;
        hostMemoryBuffer.setLong(offset * DType.CUDF_INT64.sizeInBytes, 3);
        assertEquals(3, hostMemoryBuffer.getLong(offset * DType.CUDF_INT64.sizeInBytes));
    }

    @Test
    public void testGetLength() {
        long length = hostMemoryBuffer.getLength();
        assertEquals(16, length);
    }

    @Test
    public void testCopyFromDeviceBuffer() {
        assertFalse(true);
    }
}

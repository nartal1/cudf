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

package ai.rapids.memory;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

public class UnsafeMemoryAccessorTest {

    static UnsafeMemoryAccessor memoryAccessor;

    static long address;

    @BeforeAll
    static public void setup() {
        memoryAccessor = new UnsafeMemoryAccessor();
    }

    @AfterEach
    void freeAllocation() {
        memoryAccessor.free(address);
    }

    @Test
    public void testAllocate() {
        address = memoryAccessor.allocate(3);
        assertNotEquals(0, address);
    }

    @Test
    public void setByteAndGetByte() {
        address = memoryAccessor.allocate(2);
        memoryAccessor.setByte(address, (byte) 34);
        memoryAccessor.setByte(address + 1, (byte) 63);
        Byte b = memoryAccessor.getByte(address);
        assertEquals((byte) 34, b);
        b = memoryAccessor.getByte(address + 1);
        assertEquals((byte) 63, b);
    }

    @Test
    public void setIntAndGetInt() {
        address = memoryAccessor.allocate(2 * 4);
        memoryAccessor.setInt(address, 2);
        memoryAccessor.setInt(address + 4, 4);
        int v = memoryAccessor.getInt(address);
        assertEquals(2, v);
        v = memoryAccessor.getInt(address + 4);
        assertEquals(4, v);
    }

    @Test
    public void setMemoryValue() {
        address = memoryAccessor.allocate(4);
        memoryAccessor.setMemory(address, 4, (byte) 1);
        int v = memoryAccessor.getInt(address);
        assertEquals(16843009, v);
    }
}

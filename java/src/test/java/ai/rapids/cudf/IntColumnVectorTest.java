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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.Mockito.spy;

public class IntColumnVectorTest {

    private static Logger log = LoggerFactory.getLogger(IntColumnVector.class);

    @Test
    public void testCreateColumnVectorBuilder() {
        try (IntColumnVector intColumnVector = IntColumnVector.build(3, (b) -> b.append(1))) {
            assertFalse(intColumnVector.hasNulls());
        }
    }

    @Test
    public void testArrayAllocation() {
        try (IntColumnVector intColumnVector = IntColumnVector.build(3,
                (b) -> b.append(2).append(3).append(5))) {
            assertFalse(intColumnVector.hasNulls());
            assertEquals(intColumnVector.get(0), 2);
            assertEquals(intColumnVector.get(1), 3);
            assertEquals(intColumnVector.get(2), 5);
        }
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        try (IntColumnVector intColumnVector = IntColumnVector.build(3,
                (b) -> b.append(2).append(3).append(5))) {
            assertThrows(AssertionError.class, () -> intColumnVector.get(3));
            assertFalse(intColumnVector.hasNulls());
        }
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        try (IntColumnVector intColumnVector = IntColumnVector.build(3,
                (b) -> b.append(2).append(3).append(5))) {
            assertFalse(intColumnVector.hasNulls());
            assertThrows(AssertionError.class, () -> intColumnVector.get(-1));
        }
    }

    @Test
    public void testAddingNullValues() {
        try (IntColumnVector intColumnVector = IntColumnVector.build(72,
                (b) -> {
                    for (int i = 0; i < 70; i += 2) {
                        b.append(2).append(5);
                    }
                    b.append(2).appendNull();
                })) {
            for (int i = 0; i < 71; i++) {
                log.debug("{}", intColumnVector.get(i));
                assertFalse(intColumnVector.isNull(i));
            }
            assertTrue(intColumnVector.isNull(71));
            assertTrue(intColumnVector.hasNulls());
        }
    }

    @Test
    public void testOverrunningTheBuffer() {
        try (IntColumnVector.Builder builder = IntColumnVector.builder(3)) {
            assertThrows(AssertionError.class, () -> builder.append(2).appendNull().append(5).append(4).build());
        }
    }

    @Test
    public void testCopyVector() {
        try (IntColumnVector vector1 = IntColumnVector.build(9,
                (b) -> b.append(3, 7));
             IntColumnVector vector2 = IntColumnVector.build(8,
                     (b) -> {
                         b.append(1);
                         b.append(vector1);
                     })) {
            assertEquals(1, vector2.get(0));
            for (int i = 1; i < 8; i++) {
                assertEquals(vector1.get(i - 1), vector2.get(i));
            }
        }
    }

    @Test
    void testClose() {
        try (HostMemoryBuffer mockDataBuffer = spy(HostMemoryBuffer.allocate(4 * 4));
             HostMemoryBuffer mockValidBuffer = spy(HostMemoryBuffer.allocate(8))){
            try (IntColumnVector.Builder builder = IntColumnVector.builderTest(4, mockDataBuffer, mockValidBuffer)) {
                builder.append(2).append(3).append(5).appendNull();
            }
            Mockito.verify(mockDataBuffer).doClose();
            Mockito.verify(mockValidBuffer).doClose();
        }
    }

    @Test
    public void testAdd() {
        assumeTrue(NativeDepsLoader.libraryLoaded());
        try (IntColumnVector intColumnVector1 = IntColumnVector.build(4, Range.appendInts(1, 5));
             IntColumnVector intColumnVector2 = IntColumnVector.build(4, Range.appendInts(10, 50, 10))) {

            intColumnVector1.toDeviceBuffer();
            intColumnVector2.toDeviceBuffer();

            try (IntColumnVector intColumnVector3 = intColumnVector1.add(intColumnVector2)) {
                intColumnVector3.toHostBuffer();
                assertEquals(4, intColumnVector3.getRows());
                assertEquals(0, intColumnVector3.getNullCount());
                for (int i = 0; i < 4; i++) {
                    long v1 = intColumnVector1.get(i);
                    long v2 = intColumnVector2.get(i);
                    long v3 = intColumnVector3.get(i);
                    assertEquals(v1 + v2, v3);
                    log.debug("{}", v3);
                }
            }
        }
    }
}

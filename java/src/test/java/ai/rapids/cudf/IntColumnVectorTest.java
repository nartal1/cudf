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

import java.util.Random;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.Mockito.spy;

public class IntColumnVectorTest {

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
    void testAppendVector() {
        Random random = new Random(192312989128L);
        for (int dstSize = 1; dstSize <= 100 ; dstSize++) {
            for (int dstPrefilledSize = 0 ; dstPrefilledSize < dstSize ; dstPrefilledSize++) {
                final int srcSize = dstSize - dstPrefilledSize;
                for (int  sizeOfDataNotToAdd = 0 ; sizeOfDataNotToAdd <= dstPrefilledSize ; sizeOfDataNotToAdd++) {
                    try (IntColumnVector.Builder dst = IntColumnVector.builder(dstSize);
                        IntColumnVector src = IntColumnVector.build(srcSize, (b) -> {
                            for (int i = 0 ; i < srcSize ; i++) {
                                if (random.nextBoolean()) {
                                    b.appendNull();
                                } else {
                                    b.append(random.nextInt());
                                }
                            }
                        });
                        IntColumnVector.Builder gtBuilder = IntColumnVector.builder(dstPrefilledSize)) {
                        assertEquals(dstSize, srcSize + dstPrefilledSize);
                        //add the first half of the prefilled list
                        for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd ; i++) {
                            if (random.nextBoolean()) {
                                dst.appendNull();
                                gtBuilder.appendNull();
                            } else {
                                int a = random.nextInt();
                                dst.append(a);
                                gtBuilder.append(a);
                            }
                        }
                        // append the src vector
                        dst.append(src);
                        try (IntColumnVector dstVector = dst.build();
                             IntColumnVector gt = gtBuilder.build()) {
                            for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd ; i++) {
                                assertEquals(gt.isNull(i), dstVector.isNull(i));
                                if (!gt.isNull(i)) {
                                    assertEquals(gt.get(i), dstVector.get(i));
                                }
                            }
                            for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                                assertEquals(src.isNull(j), dstVector.isNull(i));
                                if (!src.isNull(j)) {
                                    assertEquals(src.get(j), dstVector.get(i));
                                }
                            }
                            if (dstVector.offHeap.hostData.valid != null) {
                                for (int i = dstSize - sizeOfDataNotToAdd ; i < BitVectorHelper.getValidityAllocationSizeInBytes(dstVector.offHeap.hostData.valid.length); i++) {
                                    assertFalse(BitVectorHelper.isNull(dstVector.offHeap.hostData.valid, i));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void testClose() {
        try (HostMemoryBuffer mockDataBuffer = spy(HostMemoryBuffer.allocate(4 * 4));
             HostMemoryBuffer mockValidBuffer = spy(HostMemoryBuffer.allocate(8))){
            try (IntColumnVector.Builder builder = IntColumnVector.builder(4, mockDataBuffer, mockValidBuffer)) {
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
                }
            }
        }
    }
}


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

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.Mockito.spy;

public class ByteColumnVectorTest {

    @Test
    public void testCreateColumnVectorBuilder() {
        try (ByteColumnVector shortColumnVector = ByteColumnVector.build(3, (b) -> b.append((byte)1))) {
            assertFalse(shortColumnVector.hasNulls());
        }
    }

    @Test
    public void testArrayAllocation() {
        try (ByteColumnVector byteColumnVector = ByteColumnVector.build(3,
                (b) -> b.append((byte)2).append((byte)3).append((byte)5))) {
            assertFalse(byteColumnVector.hasNulls());
            assertEquals(byteColumnVector.get(0), 2);
            assertEquals(byteColumnVector.get(1), 3);
            assertEquals(byteColumnVector.get(2), 5);
        }
    }

    @Test
    public void testAppendRepeatingValues() {
        try (ByteColumnVector byteColumnVector = ByteColumnVector.build(3,
                (b) -> b.append((byte)2,(long)3))) {
            assertFalse(byteColumnVector.hasNulls());
            assertEquals(byteColumnVector.get(0), 2);
            assertEquals(byteColumnVector.get(1), 2);
            assertEquals(byteColumnVector.get(2), 2);
        }
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        try (ByteColumnVector byteColumnVector = ByteColumnVector.build(3,
                (b) -> b.append((byte)2).append((byte) 3).append(((byte)5)))) {
            assertThrows(AssertionError.class, () -> byteColumnVector.get(3));
            assertFalse(byteColumnVector.hasNulls());
        }
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        try (ByteColumnVector byteColumnVector = ByteColumnVector.build(3,
                (b) -> b.append((byte)2).append((byte)3).append((byte)5))) {
            assertFalse(byteColumnVector.hasNulls());
            assertThrows(AssertionError.class, () -> byteColumnVector.get(-1));
        }
    }

    @Test
    public void testAddingNullValues() {
        try (ByteColumnVector byteColumnVector = ByteColumnVector.build(72,
                (b) -> {
                    for (int i = 0; i < 70; i += 2) {
                        b.append((byte)2).append((byte)5);
                    }
                    b.append((byte)2).appendNull();
                })) {
            for (int i = 0; i < 71; i++) {
                assertFalse(byteColumnVector.isNull(i));
            }
            assertTrue(byteColumnVector.isNull(71));
            assertTrue(byteColumnVector.hasNulls());
        }
    }

    @Test
    public void testOverrunningTheBuffer() {
        try (ByteColumnVector.Builder builder = ByteColumnVector.builder(3)) {
            assertThrows(AssertionError.class, () -> builder.append((byte)2).appendNull().append((byte)5).append((byte)4).build());
        }
    }

    @Test
    void testAppendVector() {
        Random random = new Random(192312989128L);
        for (int dstSize = 1 ; dstSize <= 100 ; dstSize++) {
            for (int dstPrefilledSize = 0 ; dstPrefilledSize < dstSize ; dstPrefilledSize++) {
                final int srcSize = dstSize - dstPrefilledSize;
                for (int  sizeOfDataNotToAdd = 0 ; sizeOfDataNotToAdd <= dstPrefilledSize ; sizeOfDataNotToAdd++) {
                    try (ByteColumnVector.Builder dst = ByteColumnVector.builder(dstSize);
                         ByteColumnVector src = ByteColumnVector.build(srcSize, (b) -> {
                             for (int i = 0 ; i < srcSize ; i++) {
                                 if (random.nextBoolean()) {
                                     b.appendNull();
                                 } else {
                                     b.append((byte)random.nextInt());
                                 }
                             }
                         });
                         ByteColumnVector.Builder gtBuilder = ByteColumnVector.builder(dstPrefilledSize)) {
                        assertEquals(dstSize, srcSize + dstPrefilledSize);
                        //add the first half of the prefilled list
                        for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd ; i++) {
                            if (random.nextBoolean()) {
                                dst.appendNull();
                                gtBuilder.appendNull();
                            } else {
                                byte a = (byte)random.nextInt();
                                dst.append(a);
                                gtBuilder.append(a);
                            }
                        }
                        // append the src vector
                        dst.append(src);
                        try (ByteColumnVector dstVector = dst.build();
                             ByteColumnVector gt = gtBuilder.build()) {
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
        try (HostMemoryBuffer mockDataBuffer = spy(HostMemoryBuffer.allocate(4 * 8));
             HostMemoryBuffer mockValidBuffer = spy(HostMemoryBuffer.allocate(8))){
            try (ByteColumnVector.Builder builder = ByteColumnVector.builder(4, mockDataBuffer, mockValidBuffer)) {
                builder.append((byte)2).append((byte)3).append((byte)5).appendNull();
            }
            Mockito.verify(mockDataBuffer).doClose();
            Mockito.verify(mockValidBuffer).doClose();
        }
    }
}
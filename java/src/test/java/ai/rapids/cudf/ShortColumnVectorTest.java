
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

public class ShortColumnVectorTest {

    @Test
    public void testCreateColumnVectorBuilder() {
        try (ShortColumnVector shortColumnVector = ShortColumnVector.build(3, (b) -> b.append((short)1))) {
            assertFalse(shortColumnVector.hasNulls());
        }
    }

    @Test
    public void testArrayAllocation() {
        try (ShortColumnVector shortColumnVector = ShortColumnVector.build((short)2, (short)3, (short)5)) {
            assertFalse(shortColumnVector.hasNulls());
            assertEquals(shortColumnVector.get(0), 2);
            assertEquals(shortColumnVector.get(1), 3);
            assertEquals(shortColumnVector.get(2), 5);
        }
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        try (ShortColumnVector shortColumnVector = ShortColumnVector.build((short)2, (short)3, (short)5)) {
            assertThrows(AssertionError.class, () -> shortColumnVector.get(3));
            assertFalse(shortColumnVector.hasNulls());
        }
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        try (ShortColumnVector shortColumnVector = ShortColumnVector.build((short)2, (short)3, (short)5)) {
            assertFalse(shortColumnVector.hasNulls());
            assertThrows(AssertionError.class, () -> shortColumnVector.get(-1));
        }
    }

    @Test
    public void testAddingNullValues() {
        try (ShortColumnVector cv =
                     ShortColumnVector.buildBoxed(new Short[] {2,3,4,5,6,7,null,null})) {
            assertTrue(cv.hasNulls());
            assertEquals(2, cv.getNullCount());
            for (int i = 0; i < 6; i++) {
                assertFalse(cv.isNull(i));
            }
            assertTrue(cv.isNull(6));
            assertTrue(cv.isNull(7));
        }
    }

    @Test
    public void testOverrunningTheBuffer() {
        try (ShortColumnVector.Builder builder = ShortColumnVector.builder(3)) {
            assertThrows(AssertionError.class, () -> builder.append((short)2).appendNull().appendArray((short)5, (short)4).build());
        }
    }

    @Test
    void testAppendVector() {
        Random random = new Random(192312989128L);
        for (int dstSize = 1 ; dstSize <= 100 ; dstSize++) {
            for (int dstPrefilledSize = 0 ; dstPrefilledSize < dstSize ; dstPrefilledSize++) {
                final int srcSize = dstSize - dstPrefilledSize;
                for (int  sizeOfDataNotToAdd = 0 ; sizeOfDataNotToAdd <= dstPrefilledSize ; sizeOfDataNotToAdd++) {
                    try (ShortColumnVector.Builder dst = ShortColumnVector.builder(dstSize);
                         ShortColumnVector src = ShortColumnVector.build(srcSize, (b) -> {
                             for (int i = 0 ; i < srcSize ; i++) {
                                 if (random.nextBoolean()) {
                                     b.appendNull();
                                 } else {
                                     b.append((short)random.nextInt());
                                 }
                             }
                         });
                         ShortColumnVector.Builder gtBuilder = ShortColumnVector.builder(dstPrefilledSize)) {
                         assertEquals(dstSize, srcSize + dstPrefilledSize);
                         //add the first half of the prefilled list
                         for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd ; i++) {
                             if (random.nextBoolean()) {
                                 dst.appendNull();
                                 gtBuilder.appendNull();
                             } else {
                                 short a = (short)random.nextInt();
                                 dst.append(a);
                                 gtBuilder.append(a);
                             }
                         }
                         // append the src vector
                         dst.append(src);
                         try (ShortColumnVector dstVector = dst.build();
                              ShortColumnVector gt = gtBuilder.build()) {
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
            try (ShortColumnVector.Builder builder = ShortColumnVector.builder(4, mockDataBuffer, mockValidBuffer)) {
                builder.appendArray((short)2, (short)3, (short)5).appendNull();
            }
            Mockito.verify(mockDataBuffer).doClose();
            Mockito.verify(mockValidBuffer).doClose();
        }
    }
}
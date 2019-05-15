
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

public class LongColumnVectorTest {

    @Test
    public void testCreateColumnVectorBuilder() {
        try (LongColumnVector longColumnVector = LongColumnVector.build(3, (b) -> b.append(1L))) {
            assertFalse(longColumnVector.hasNulls());
        }
    }

    @Test
    public void testArrayAllocation() {
        try (LongColumnVector longColumnVector = LongColumnVector.build(2L, 3L, 5L)) {
            assertFalse(longColumnVector.hasNulls());
            assertEquals(longColumnVector.get(0), 2);
            assertEquals(longColumnVector.get(1), 3);
            assertEquals(longColumnVector.get(2), 5);
        }
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        try (LongColumnVector longColumnVector = LongColumnVector.build(2L, 3L, 5L)) {
            assertThrows(AssertionError.class, () -> longColumnVector.get(3));
            assertFalse(longColumnVector.hasNulls());
        }
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        try (LongColumnVector longColumnVector = LongColumnVector.build(2L, 3L, 5L)) {
            assertFalse(longColumnVector.hasNulls());
            assertThrows(AssertionError.class, () -> longColumnVector.get(-1));
        }
    }

    @Test
    public void testAddingNullValues() {
        try (LongColumnVector cv = LongColumnVector.buildBoxed(2L,3L,4L,5L,6L,7L, null,null)) {
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
        try (LongColumnVector.Builder builder = LongColumnVector.builder(3)) {
            assertThrows(AssertionError.class, () -> builder.append(2).appendNull().append(5).append(4).build());
        }
    }

    @Test
    void testAppendVector() {
        Random random = new Random(192312989128L);
        for (int dstSize = 1 ; dstSize <= 100 ; dstSize++) {
            for (int dstPrefilledSize = 0 ; dstPrefilledSize < dstSize ; dstPrefilledSize++) {
                final int srcSize = dstSize - dstPrefilledSize;
                for (int  sizeOfDataNotToAdd = 0 ; sizeOfDataNotToAdd <= dstPrefilledSize ; sizeOfDataNotToAdd++) {
                    try (LongColumnVector.Builder dst = LongColumnVector.builder(dstSize);
                         LongColumnVector src = LongColumnVector.build(srcSize, (b) -> {
                             for (int i = 0 ; i < srcSize ; i++) {
                                 if (random.nextBoolean()) {
                                     b.appendNull();
                                 } else {
                                     b.append(random.nextLong());
                                 }
                             }
                         });
                         LongColumnVector.Builder gtBuilder = LongColumnVector.builder(dstPrefilledSize)) {
                         assertEquals(dstSize, srcSize + dstPrefilledSize);
                         //add the first half of the prefilled list
                         for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd ; i++) {
                             if (random.nextBoolean()) {
                                 dst.appendNull();
                                 gtBuilder.appendNull();
                             } else {
                                 long a = random.nextLong();
                                 dst.append(a);
                                 gtBuilder.append(a);
                             }
                         }
                         // append the src vector
                         dst.append(src);
                         try (LongColumnVector dstVector = dst.build();
                              LongColumnVector gt = gtBuilder.build()) {
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
            try (LongColumnVector.Builder builder = LongColumnVector.builder(4, mockDataBuffer, mockValidBuffer)) {
                builder.appendArray(2, 3, 5).appendNull();
            }
            Mockito.verify(mockDataBuffer).doClose();
            Mockito.verify(mockValidBuffer).doClose();
        }
    }

    @Test
    public void testAdd() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (LongColumnVector longColumnVector1 = LongColumnVector.build(4, Range.appendLongs(1,5));
             LongColumnVector longColumnVector2 = LongColumnVector.build(4, Range.appendLongs(10,  50, 10))) {

            longColumnVector1.ensureOnDevice();
            longColumnVector2.ensureOnDevice();

            try (LongColumnVector longColumnVector3 = longColumnVector1.add(longColumnVector2)) {
                longColumnVector3.ensureOnHost();
                assertEquals(4, longColumnVector3.getRows());
                assertEquals(0, longColumnVector3.getNullCount());
                for (int i = 0; i < 4; i++) {
                    long v1 = longColumnVector1.get(i);
                    long v2 = longColumnVector2.get(i);
                    long v3 = longColumnVector3.get(i);
                    assertEquals(v1 + v2, v3);
                }
            }
        }
    }
}
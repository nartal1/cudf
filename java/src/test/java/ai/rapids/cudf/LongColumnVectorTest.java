
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
        try (ColumnVector longColumnVector = ColumnVector.build(DType.INT64, 3, (b) -> b.appendLong(1L))) {
            assertFalse(longColumnVector.hasNulls());
        }
    }

    @Test
    public void testArrayAllocation() {
        try (ColumnVector longColumnVector = ColumnVector.build(2L, 3L, 5L)) {
            assertFalse(longColumnVector.hasNulls());
            assertEquals(longColumnVector.getLong(0), 2);
            assertEquals(longColumnVector.getLong(1), 3);
            assertEquals(longColumnVector.getLong(2), 5);
        }
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        try (ColumnVector longColumnVector = ColumnVector.build(2L, 3L, 5L)) {
            assertThrows(AssertionError.class, () -> longColumnVector.getLong(3));
            assertFalse(longColumnVector.hasNulls());
        }
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        try (ColumnVector longColumnVector = ColumnVector.build(2L, 3L, 5L)) {
            assertFalse(longColumnVector.hasNulls());
            assertThrows(AssertionError.class, () -> longColumnVector.getLong(-1));
        }
    }

    @Test
    public void testAddingNullValues() {
        try (ColumnVector cv = ColumnVector.buildBoxed(2L,3L,4L,5L,6L,7L, null,null)) {
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
        try (ColumnVector.Builder builder = ColumnVector.builder(DType.INT64,3)) {
            assertThrows(AssertionError.class, () -> builder.appendLong(2).appendNull().appendLong(5).appendLong(4).build());
        }
    }

    @Test
    void testAppendVector() {
        Random random = new Random(192312989128L);
        for (int dstSize = 1 ; dstSize <= 100 ; dstSize++) {
            for (int dstPrefilledSize = 0 ; dstPrefilledSize < dstSize ; dstPrefilledSize++) {
                final int srcSize = dstSize - dstPrefilledSize;
                for (int  sizeOfDataNotToAdd = 0 ; sizeOfDataNotToAdd <= dstPrefilledSize ; sizeOfDataNotToAdd++) {
                    try (ColumnVector.Builder dst = ColumnVector.builder(DType.INT64, dstSize);
                         ColumnVector src = ColumnVector.build(DType.INT64, srcSize, (b) -> {
                             for (int i = 0 ; i < srcSize ; i++) {
                                 if (random.nextBoolean()) {
                                     b.appendNull();
                                 } else {
                                     b.appendLong(random.nextLong());
                                 }
                             }
                         });
                         ColumnVector.Builder gtBuilder = ColumnVector.builder(DType.INT64, dstPrefilledSize)) {
                         assertEquals(dstSize, srcSize + dstPrefilledSize);
                         //add the first half of the prefilled list
                         for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd ; i++) {
                             if (random.nextBoolean()) {
                                 dst.appendNull();
                                 gtBuilder.appendNull();
                             } else {
                                 long a = random.nextLong();
                                 dst.appendLong(a);
                                 gtBuilder.appendLong(a);
                             }
                         }
                         // append the src vector
                         dst.append(src);
                         try (ColumnVector dstVector = dst.build();
                              ColumnVector gt = gtBuilder.build()) {
                             for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd ; i++) {
                                 assertEquals(gt.isNull(i), dstVector.isNull(i));
                                 if (!gt.isNull(i)) {
                                     assertEquals(gt.getLong(i), dstVector.getLong(i));
                                 }
                             }
                             for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                                 assertEquals(src.isNull(j), dstVector.isNull(i));
                                 if (!src.isNull(j)) {
                                     assertEquals(src.getLong(j), dstVector.getLong(i));
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
            try (ColumnVector.Builder builder = ColumnVector.builder(DType.INT64, 4, mockDataBuffer, mockValidBuffer)) {
                builder.appendLongs(new long[] {2, 3, 5}).appendNull();
            }
            Mockito.verify(mockDataBuffer).doClose();
            Mockito.verify(mockValidBuffer).doClose();
        }
    }

    @Test
    public void testAdd() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (ColumnVector longColumnVector1 = ColumnVector.build(DType.INT64, 4, Range.appendLongs(1,5));
             ColumnVector longColumnVector2 = ColumnVector.build(DType.INT64, 4, Range.appendLongs(10,  50, 10))) {

            longColumnVector1.ensureOnDevice();
            longColumnVector2.ensureOnDevice();

            try (ColumnVector longColumnVector3 = longColumnVector1.add(longColumnVector2)) {
                longColumnVector3.ensureOnHost();
                assertEquals(4, longColumnVector3.getRows());
                assertEquals(0, longColumnVector3.getNullCount());
                for (int i = 0; i < 4; i++) {
                    long v1 = longColumnVector1.getLong(i);
                    long v2 = longColumnVector2.getLong(i);
                    long v3 = longColumnVector3.getLong(i);
                    assertEquals(v1 + v2, v3);
                }
            }
        }
    }
}
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

public class IntColumnVectorTest {

    @Test
    public void testCreateColumnVectorBuilder() {
        try (ColumnVector intColumnVector = ColumnVector.build(DType.INT32, 3, (b) -> b.append(1))) {
            assertFalse(intColumnVector.hasNulls());
        }
    }

    @Test
    public void testArrayAllocation() {
        try (ColumnVector intColumnVector = ColumnVector.fromInts(2, 3, 5)) {
            assertFalse(intColumnVector.hasNulls());
            assertEquals(intColumnVector.getInt(0), 2);
            assertEquals(intColumnVector.getInt(1), 3);
            assertEquals(intColumnVector.getInt(2), 5);
        }
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        try (ColumnVector intColumnVector = ColumnVector.fromInts(2, 3, 5)) {
            assertThrows(AssertionError.class, () -> intColumnVector.getInt(3));
            assertFalse(intColumnVector.hasNulls());
        }
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        try (ColumnVector intColumnVector = ColumnVector.fromInts(2, 3, 5)) {
            assertFalse(intColumnVector.hasNulls());
            assertThrows(AssertionError.class, () -> intColumnVector.getInt(-1));
        }
    }

    @Test
    public void testAddingNullValues() {
        try (ColumnVector cv = ColumnVector.fromBoxedInts(2,3,4,5,6,7,null,null)) {
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
        try (ColumnVector.Builder builder = ColumnVector.builder(DType.INT32, 3)) {
            assertThrows(AssertionError.class, () -> builder.append(2).appendNull().appendArray(new int[]{5, 4}).build());
        }
    }

    @Test
    void testAppendVector() {
        Random random = new Random(192312989128L);
        for (int dstSize = 1; dstSize <= 100 ; dstSize++) {
            for (int dstPrefilledSize = 0 ; dstPrefilledSize < dstSize ; dstPrefilledSize++) {
                final int srcSize = dstSize - dstPrefilledSize;
                for (int  sizeOfDataNotToAdd = 0 ; sizeOfDataNotToAdd <= dstPrefilledSize ; sizeOfDataNotToAdd++) {
                    try (ColumnVector.Builder dst = ColumnVector.builder(DType.INT32, dstSize);
                        ColumnVector src = ColumnVector.build(DType.INT32, srcSize, (b) -> {
                            for (int i = 0 ; i < srcSize ; i++) {
                                if (random.nextBoolean()) {
                                    b.appendNull();
                                } else {
                                    b.append(random.nextInt());
                                }
                            }
                        });
                        ColumnVector.Builder gtBuilder = ColumnVector.builder(DType.INT32, dstPrefilledSize)) {
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
                        try (ColumnVector dstVector = dst.build();
                             ColumnVector gt = gtBuilder.build()) {
                            for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd ; i++) {
                                assertEquals(gt.isNull(i), dstVector.isNull(i));
                                if (!gt.isNull(i)) {
                                    assertEquals(gt.getInt(i), dstVector.getInt(i));
                                }
                            }
                            for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                                assertEquals(src.isNull(j), dstVector.isNull(i));
                                if (!src.isNull(j)) {
                                    assertEquals(src.getInt(j), dstVector.getInt(i));
                                }
                            }
                            if (dstVector.hasValidityVector()) {
                                long maxIndex = BitVectorHelper.getValidityAllocationSizeInBytes(dstVector.getRowCount()) * 8;
                                for (long i = dstSize - sizeOfDataNotToAdd; i < maxIndex; i++) {
                                    assertFalse(dstVector.isNullExtendedRange(i));
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
            try (ColumnVector.Builder builder = new ColumnVector.Builder(DType.INT32, 4, mockDataBuffer, mockValidBuffer)) {
                builder.appendArray(new int[]{2, 3, 5}).appendNull();
            }
            Mockito.verify(mockDataBuffer).doClose();
            Mockito.verify(mockValidBuffer).doClose();
        }
    }

    @Test
    public void testAdd() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (ColumnVector intColumnVector1 = ColumnVector.build(DType.INT32, 4, Range.appendInts(1, 5));
             ColumnVector intColumnVector2 = ColumnVector.build(DType.INT32, 4, Range.appendInts(10, 50, 10))) {

            intColumnVector1.ensureOnDevice();
            intColumnVector2.ensureOnDevice();

            try (ColumnVector intColumnVector3 = intColumnVector1.add(intColumnVector2)) {
                intColumnVector3.ensureOnHost();
                assertEquals(4, intColumnVector3.getRowCount());
                assertEquals(0, intColumnVector3.getNullCount());
                for (int i = 0; i < 4; i++) {
                    long v1 = intColumnVector1.getInt(i);
                    long v2 = intColumnVector2.getInt(i);
                    long v3 = intColumnVector3.getInt(i);
                    assertEquals(v1 + v2, v3);
                }
            }
        }
    }
}

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

public class DoubleColumnVectorTest {

    @Test
    public void testCreateColumnVectorBuilder() {
        try (ColumnVector doubleColumnVector = ColumnVector.build(DType.FLOAT64, 3, (b) -> b.append(1.0))) {
            assertFalse(doubleColumnVector.hasNulls());
        }
    }

    @Test
    public void testArrayAllocation() {
        try (ColumnVector doubleColumnVector = ColumnVector.build(2.1, 3.02, 5.003)) {
            assertFalse(doubleColumnVector.hasNulls());
            assertEquals(doubleColumnVector.getDouble(0), 2.1, 0.01);
            assertEquals(doubleColumnVector.getDouble(1), 3.02,0.01);
            assertEquals(doubleColumnVector.getDouble(2), 5.003,0.001);
        }
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        try (ColumnVector doubleColumnVector = ColumnVector.build(2.1, 3.02, 5.003)) {
            assertThrows(AssertionError.class, () -> doubleColumnVector.getDouble(3));
            assertFalse(doubleColumnVector.hasNulls());
        }
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        try (ColumnVector doubleColumnVector = ColumnVector.build(2.1, 3.02, 5.003)) {
            assertFalse(doubleColumnVector.hasNulls());
            assertThrows(AssertionError.class, () -> doubleColumnVector.getDouble(-1));
        }
    }

    @Test
    public void testAddingNullValues() {
        try (ColumnVector cv =
                     ColumnVector.buildBoxed(2.0,3.0,4.0,5.0,6.0,7.0,null,null)) {
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
        try (ColumnVector.Builder builder = ColumnVector.builder(DType.FLOAT64, 3)) {
            assertThrows(AssertionError.class, () -> builder.append(2.1).appendNull().appendArray(new double[]{5.003, 4.0}).build());
        }
    }

    @Test
    void testAppendVector() {
        Random random = new Random(192312989128L);
        for (int dstSize = 1 ; dstSize <= 100 ; dstSize++) {
            for (int dstPrefilledSize = 0 ; dstPrefilledSize < dstSize ; dstPrefilledSize++) {
                final int srcSize = dstSize - dstPrefilledSize;
                for (int  sizeOfDataNotToAdd = 0 ; sizeOfDataNotToAdd <= dstPrefilledSize ; sizeOfDataNotToAdd++) {
                    try (ColumnVector.Builder dst = ColumnVector.builder(DType.FLOAT64, dstSize);
                         ColumnVector src = ColumnVector.build(DType.FLOAT64, srcSize, (b) -> {
                             for (int i = 0 ; i < srcSize ; i++) {
                                 if (random.nextBoolean()) {
                                     b.appendNull();
                                 } else {
                                     b.append(random.nextDouble());
                                 }
                             }
                         });
                         ColumnVector.Builder gtBuilder = ColumnVector.builder(DType.FLOAT64, dstPrefilledSize)) {
                         assertEquals(dstSize, srcSize + dstPrefilledSize);
                         //add the first half of the prefilled list
                         for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd ; i++) {
                             if (random.nextBoolean()) {
                                 dst.appendNull();
                                 gtBuilder.appendNull();
                             } else {
                                 double a = random.nextDouble();
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
                                     assertEquals(gt.getDouble(i), dstVector.getDouble(i));
                                 }
                             }
                             for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                                 assertEquals(src.isNull(j), dstVector.isNull(i));
                                 if (!src.isNull(j)) {
                                     assertEquals(src.getDouble(j), dstVector.getDouble(i));
                                 }
                             }
                             // TODO do we really need this?
//                             if (dstVector.offHeap.hostData.valid != null) {
//                                 for (int i = dstSize - sizeOfDataNotToAdd ; i < BitVectorHelper.getValidityAllocationSizeInBytes(dstVector.offHeap.hostData.valid.length); i++) {
//                                     assertFalse(BitVectorHelper.isNull(dstVector.offHeap.hostData.valid, i));
//                                 }
//                             }
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
            try (ColumnVector.Builder builder = ColumnVector.builder(DType.FLOAT64, 4, mockDataBuffer, mockValidBuffer)) {
                builder.appendArray(new double[] {2.1, 3.02, 5.004}).appendNull();
            }
            Mockito.verify(mockDataBuffer).doClose();
            Mockito.verify(mockValidBuffer).doClose();
        }
    }

    @Test
    public void testAdd() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (ColumnVector doubleColumnVector1 = ColumnVector.build(DType.FLOAT64, 5, Range.appendDoubles(1.1, 5.5));
             ColumnVector doubleColumnVector2 = ColumnVector.build(DType.FLOAT64, 5, Range.appendDoubles(10,  60, 10))) {

            doubleColumnVector1.ensureOnDevice();
            doubleColumnVector2.ensureOnDevice();

            try (ColumnVector doubleColumnVector3 = doubleColumnVector1.add(doubleColumnVector2)) {
                doubleColumnVector3.ensureOnHost();
                assertEquals(5, doubleColumnVector3.getRows());
                assertEquals(0, doubleColumnVector3.getNullCount());
                for (int i = 0; i < 5; i++) {
                    double v1 = doubleColumnVector1.getDouble(i);
                    double v2 = doubleColumnVector2.getDouble(i);
                    double v3 = doubleColumnVector3.getDouble(i);
                    assertEquals(v1 + v2, v3,0.001);
                }
            }
        }
    }
}
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

public class FloatColumnVectorTest {

    @Test
    public void testCreateColumnVectorBuilder() {
        try (ColumnVector floatColumnVector = ColumnVector.build(DType.FLOAT32, 3, (b) -> b.append(1.0f))) {
            assertFalse(floatColumnVector.hasNulls());
        }
    }

    @Test
    public void testArrayAllocation() {
        try (ColumnVector floatColumnVector = ColumnVector.fromFloats(2.1f, 3.02f, 5.003f)) {
            assertFalse(floatColumnVector.hasNulls());
            assertEquals(floatColumnVector.getFloat(0), 2.1, 0.01);
            assertEquals(floatColumnVector.getFloat(1), 3.02,0.01);
            assertEquals(floatColumnVector.getFloat(2), 5.003,0.001);
        }
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        try (ColumnVector floatColumnVector = ColumnVector.fromFloats(2.1f, 3.02f, 5.003f)) {
            assertThrows(AssertionError.class, () -> floatColumnVector.getFloat(3));
            assertFalse(floatColumnVector.hasNulls());
        }
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        try (ColumnVector floatColumnVector = ColumnVector.fromFloats(2.1f, 3.02f, 5.003f)) {
            assertFalse(floatColumnVector.hasNulls());
            assertThrows(AssertionError.class, () -> floatColumnVector.getFloat(-1));
        }
    }

    @Test
    public void testAddingNullValues() {
        try (ColumnVector cv = ColumnVector.fromBoxedFloats(
                new Float[] {2f,3f,4f,5f,6f,7f,null,null})) {
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
        try (ColumnVector.Builder builder = ColumnVector.builder(DType.FLOAT32, 3)) {
            assertThrows(AssertionError.class, () -> builder.append(2.1f).appendNull().appendArray(5.003f, 4.0f).build());
        }
    }

    @Test
    void testAppendVector() {
        Random random = new Random(192312989128L);
        for (int dstSize = 1 ; dstSize <= 100 ; dstSize++) {
            for (int dstPrefilledSize = 0 ; dstPrefilledSize < dstSize ; dstPrefilledSize++) {
                final int srcSize = dstSize - dstPrefilledSize;
                for (int  sizeOfDataNotToAdd = 0 ; sizeOfDataNotToAdd <= dstPrefilledSize ; sizeOfDataNotToAdd++) {
                    try (ColumnVector.Builder dst = ColumnVector.builder(DType.FLOAT32, dstSize);
                         ColumnVector src = ColumnVector.build(DType.FLOAT32, srcSize, (b) -> {
                             for (int i = 0 ; i < srcSize ; i++) {
                                 if (random.nextBoolean()) {
                                     b.appendNull();
                                 } else {
                                     b.append(random.nextFloat());
                                 }
                             }
                         });
                         ColumnVector.Builder gtBuilder = ColumnVector.builder(DType.FLOAT32, dstPrefilledSize)) {
                         assertEquals(dstSize, srcSize + dstPrefilledSize);
                         //add the first half of the prefilled list
                         for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd ; i++) {
                             if (random.nextBoolean()) {
                                 dst.appendNull();
                                 gtBuilder.appendNull();
                             } else {
                                 float a = random.nextFloat();
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
                                     assertEquals(gt.getFloat(i), dstVector.getFloat(i));
                                 }
                             }
                             for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                                 assertEquals(src.isNull(j), dstVector.isNull(i));
                                 if (!src.isNull(j)) {
                                     assertEquals(src.getFloat(j), dstVector.getFloat(i));
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
        try (HostMemoryBuffer mockDataBuffer = spy(HostMemoryBuffer.allocate(4 * 8));
             HostMemoryBuffer mockValidBuffer = spy(HostMemoryBuffer.allocate(8))){
            try (ColumnVector.Builder builder = new ColumnVector.Builder(DType.FLOAT32, 4, mockDataBuffer, mockValidBuffer)) {
                builder.appendArray(2.1f, 3.02f, 5.004f).appendNull();
            }
            Mockito.verify(mockDataBuffer).doClose();
            Mockito.verify(mockValidBuffer).doClose();
        }
    }

    @Test
    public void testAdd() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (ColumnVector floatColumnVector1 = ColumnVector.build(DType.FLOAT32, 5, Range.appendFloats(1.1f, 5.5f));
             ColumnVector floatColumnVector2 = ColumnVector.build(DType.FLOAT32, 5, Range.appendFloats(10,  60, 10))) {

            floatColumnVector1.ensureOnDevice();
            floatColumnVector2.ensureOnDevice();

            try (ColumnVector floatColumnVector3 = floatColumnVector1.add(floatColumnVector2)) {
                floatColumnVector3.ensureOnHost();
                assertEquals(5, floatColumnVector3.getRowCount());
                assertEquals(0, floatColumnVector3.getNullCount());
                for (int i = 0; i < 5; i++) {
                    float v1 = floatColumnVector1.getFloat(i);
                    float v2 = floatColumnVector2.getFloat(i);
                    float v3 = floatColumnVector3.getFloat(i);
                    assertEquals(v1 + v2, v3,0.001);
                }
            }
        }
    }
}
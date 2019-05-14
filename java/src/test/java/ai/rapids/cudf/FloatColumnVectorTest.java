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
        try (FloatColumnVector floatColumnVector = FloatColumnVector.build(3, (b) -> b.append(1))) {
            assertFalse(floatColumnVector.hasNulls());
        }
    }

    @Test
    public void testArrayAllocation() {
        try (FloatColumnVector floatColumnVector = FloatColumnVector.build((float)2.1, (float)3.02, (float)5.003)) {
            assertFalse(floatColumnVector.hasNulls());
            assertEquals(floatColumnVector.get(0), 2.1, 0.01);
            assertEquals(floatColumnVector.get(1), 3.02,0.01);
            assertEquals(floatColumnVector.get(2), 5.003,0.001);
        }
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        try (FloatColumnVector floatColumnVector = FloatColumnVector.build((float)2.1, (float)3.02, (float)5.003)) {
            assertThrows(AssertionError.class, () -> floatColumnVector.get(3));
            assertFalse(floatColumnVector.hasNulls());
        }
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        try (FloatColumnVector floatColumnVector = FloatColumnVector.build((float)2.1, (float)3.02, (float)5.003)) {
            assertFalse(floatColumnVector.hasNulls());
            assertThrows(AssertionError.class, () -> floatColumnVector.get(-1));
        }
    }

    @Test
    public void testAddingNullValues() {
        try (FloatColumnVector cv = FloatColumnVector.buildBoxed(
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
        try (FloatColumnVector.Builder builder = FloatColumnVector.builder(3)) {
            assertThrows(AssertionError.class, () -> builder.append((float)2.1).appendNull().appendArray((float)5.003, (float)4.0).build());
        }
    }

    @Test
    void testAppendVector() {
        Random random = new Random(192312989128L);
        for (int dstSize = 1 ; dstSize <= 100 ; dstSize++) {
            for (int dstPrefilledSize = 0 ; dstPrefilledSize < dstSize ; dstPrefilledSize++) {
                final int srcSize = dstSize - dstPrefilledSize;
                for (int  sizeOfDataNotToAdd = 0 ; sizeOfDataNotToAdd <= dstPrefilledSize ; sizeOfDataNotToAdd++) {
                    try (FloatColumnVector.Builder dst = FloatColumnVector.builder(dstSize);
                         FloatColumnVector src = FloatColumnVector.build(srcSize, (b) -> {
                             for (int i = 0 ; i < srcSize ; i++) {
                                 if (random.nextBoolean()) {
                                     b.appendNull();
                                 } else {
                                     b.append(random.nextFloat());
                                 }
                             }
                         });
                         FloatColumnVector.Builder gtBuilder = FloatColumnVector.builder(dstPrefilledSize)) {
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
                         try (FloatColumnVector dstVector = dst.build();
                              FloatColumnVector gt = gtBuilder.build()) {
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
            try (FloatColumnVector.Builder builder = FloatColumnVector.builder(4, mockDataBuffer, mockValidBuffer)) {
                builder.appendArray((float)2.1, (float)3.02, (float)5.004).appendNull();
            }
            Mockito.verify(mockDataBuffer).doClose();
            Mockito.verify(mockValidBuffer).doClose();
        }
    }

    @Test
    public void testAdd() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (FloatColumnVector floatColumnVector1 = FloatColumnVector.build(5, Range.appendFloats((float)1.1,(float)5.5));
             FloatColumnVector floatColumnVector2 = FloatColumnVector.build(5, Range.appendFloats(10,  60, 10))) {

            floatColumnVector1.ensureOnDevice();
            floatColumnVector2.ensureOnDevice();

            try (FloatColumnVector floatColumnVector3 = floatColumnVector1.add(floatColumnVector2)) {
                floatColumnVector3.ensureOnHost();
                assertEquals(5, floatColumnVector3.getRows());
                assertEquals(0, floatColumnVector3.getNullCount());
                for (int i = 0; i < 5; i++) {
                    float v1 = floatColumnVector1.get(i);
                    float v2 = floatColumnVector2.get(i);
                    float v3 = floatColumnVector3.get(i);
                    assertEquals(v1 + v2, v3,0.001);
                }
            }
        }
    }
}
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
        try (DoubleColumnVector doubleColumnVector = DoubleColumnVector.build(3, (b) -> b.append(1))) {
            assertFalse(doubleColumnVector.hasNulls());
        }
    }

    @Test
    public void testArrayAllocation() {
        try (DoubleColumnVector doubleColumnVector = DoubleColumnVector.build(3,
                (b) -> b.append(2.1).append(3.02).append(5.003))) {
            assertFalse(doubleColumnVector.hasNulls());
            assertEquals(doubleColumnVector.get(0), 2.1, 0.01);
            assertEquals(doubleColumnVector.get(1), 3.02,0.01);
            assertEquals(doubleColumnVector.get(2), 5.003,0.001);
        }
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        try (DoubleColumnVector doubleColumnVector = DoubleColumnVector.build(3,
                (b) -> b.append(2.1).append(3.02).append(5.003))) {
            assertThrows(AssertionError.class, () -> doubleColumnVector.get(3));
            assertFalse(doubleColumnVector.hasNulls());
        }
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        try (DoubleColumnVector doubleColumnVector = DoubleColumnVector.build(3,
                (b) -> b.append(2.1).append(3.02).append(5.003))) {
            assertFalse(doubleColumnVector.hasNulls());
            assertThrows(AssertionError.class, () -> doubleColumnVector.get(-1));
        }
    }

    @Test
    public void testAddingNullValues() {
        try (DoubleColumnVector doubleColumnVector = DoubleColumnVector.build(72,
                (b) -> {
                    for (int i = 0; i < 70; i += 2) {
                        b.append(2.1).append(5.003);
                    }
                    b.append(2.1).appendNull();
                })) {
            for (int i = 0; i < 71; i++) {
                assertFalse(doubleColumnVector.isNull(i));
            }
            assertTrue(doubleColumnVector.isNull(71));
            assertTrue(doubleColumnVector.hasNulls());
        }
    }

    @Test
    public void testOverrunningTheBuffer() {
        try (DoubleColumnVector.Builder builder = DoubleColumnVector.builder(3)) {
            assertThrows(AssertionError.class, () -> builder.append(2.1).appendNull().append(5.003).append(4.0).build());
        }
    }

    @Test
    void testAppendVector() {
        Random random = new Random(192312989128L);
        for (int dstSize = 1 ; dstSize <= 100 ; dstSize++) {
            for (int dstPrefilledSize = 0 ; dstPrefilledSize < dstSize ; dstPrefilledSize++) {
                final int srcSize = dstSize - dstPrefilledSize;
                for (int  sizeOfDataNotToAdd = 0 ; sizeOfDataNotToAdd <= dstPrefilledSize ; sizeOfDataNotToAdd++) {
                    try (DoubleColumnVector.Builder dst = DoubleColumnVector.builder(dstSize);
                         DoubleColumnVector src = DoubleColumnVector.build(srcSize, (b) -> {
                             for (int i = 0 ; i < srcSize ; i++) {
                                 if (random.nextBoolean()) {
                                     b.appendNull();
                                 } else {
                                     b.append(random.nextDouble());
                                 }
                             }
                         });
                         DoubleColumnVector.Builder gtBuilder = DoubleColumnVector.builder(dstPrefilledSize)) {
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
                         try (DoubleColumnVector dstVector = dst.build();
                              DoubleColumnVector gt = gtBuilder.build()) {
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
            try (DoubleColumnVector.Builder builder = DoubleColumnVector.builder(4, mockDataBuffer, mockValidBuffer)) {
                builder.append(2.1).append(3.02).append(5.004).appendNull();
            }
            Mockito.verify(mockDataBuffer).doClose();
            Mockito.verify(mockValidBuffer).doClose();
        }
    }

    @Test
    public void testAdd() {
        assumeTrue(Cuda.isEnvCompatibleForTesting());
        try (DoubleColumnVector doubleColumnVector1 = DoubleColumnVector.build(5, Range.appendDoubles(1.1,5.5));
             DoubleColumnVector doubleColumnVector2 = DoubleColumnVector.build(5, Range.appendDoubles(10,  60, 10))) {

            doubleColumnVector1.toDeviceBuffer();
            doubleColumnVector2.toDeviceBuffer();

            try (DoubleColumnVector doubleColumnVector3 = doubleColumnVector1.add(doubleColumnVector2)) {
                doubleColumnVector3.toHostBuffer();
                assertEquals(5, doubleColumnVector3.getRows());
                assertEquals(0, doubleColumnVector3.getNullCount());
                for (int i = 0; i < 5; i++) {
                    double v1 = doubleColumnVector1.get(i);
                    double v2 = doubleColumnVector2.get(i);
                    double v3 = doubleColumnVector3.get(i);
                    assertEquals(v1 + v2, v3,0.001);
                }
            }
        }
    }
}
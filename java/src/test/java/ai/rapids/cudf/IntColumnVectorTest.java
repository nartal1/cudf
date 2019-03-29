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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.Mockito.spy;

public class IntColumnVectorTest {

    private static Logger log = LoggerFactory.getLogger(IntColumnVector.class);

    @Test
    public void testCreateColumnVectorBuilder() {
        IntColumnVector.IntColumnVectorBuilder builder = new IntColumnVector.IntColumnVectorBuilder(3);
        IntColumnVector intColumnVector = builder.build();
        assertEquals(intColumnVector.getClass(), IntColumnVector.class);
        assertFalse(intColumnVector.hasNulls());
    }

    @Test
    public void testArrayAllocation() {
        IntColumnVector.IntColumnVectorBuilder builder = new IntColumnVector.IntColumnVectorBuilder(3);
        IntColumnVector intColumnVector = builder.append(2).append(3).append(5).build();
        assertFalse(intColumnVector.hasNulls());
        assertEquals(intColumnVector.getValue(0), 2);
        assertEquals(intColumnVector.getValue(1), 3);
        assertEquals(intColumnVector.getValue(2), 5);
    }

    @Test
    public void testUpperIndexOutOfBoundsException() {
        IntColumnVector.IntColumnVectorBuilder builder = new IntColumnVector.IntColumnVectorBuilder(3);
        IntColumnVector intColumnVector = builder.append(2).append(3).append(5).build();
        assertThrows(IndexOutOfBoundsException.class, () -> intColumnVector.getValue(3));
        assertFalse(intColumnVector.hasNulls());
    }

    @Test
    public void testLowerIndexOutOfBoundsException() {
        IntColumnVector.IntColumnVectorBuilder builder = new IntColumnVector.IntColumnVectorBuilder(3);
        IntColumnVector intColumnVector = builder.append(2).append(3).append(5).build();
        assertFalse(intColumnVector.hasNulls());
        assertThrows(IndexOutOfBoundsException.class, () -> intColumnVector.getValue(-1));
    }

    @Test
    public void testAddingNullValues() {
        IntColumnVector.IntColumnVectorBuilder builder = new IntColumnVector.IntColumnVectorBuilder(72);

        for (int i = 0; i < 70; i += 2) {
            builder.append(2).append(5);
        }
        builder.append(2).appendNull();
        IntColumnVector intColumnVector = builder.build();

        for (int i = 0 ; i < 71 ; i++) {
            log.debug("{}", intColumnVector.getValue(i));
            assertFalse(intColumnVector.isNull(i));
        }
        assertTrue(intColumnVector.isNull(71));
        assertTrue(intColumnVector.hasNulls());
    }

    @Test
    public void testOverrunningTheBuffer() {
        IntColumnVector.IntColumnVectorBuilder builder = new IntColumnVector.IntColumnVectorBuilder(3);
        assertThrows(IndexOutOfBoundsException.class, () -> builder.append(2).appendNull().append(5).append(4).build());
    }

    @Test
    public void testCopyVector() {
        IntColumnVector.IntColumnVectorBuilder builder = new IntColumnVector.IntColumnVectorBuilder(7);
        for (int i = 0; i < 7; i++) {
            builder.append(3);
        }

        IntColumnVector vector1 = builder.build();

        builder = new IntColumnVector.IntColumnVectorBuilder(8);

        IntColumnVector vector2 = builder.append(1).append(vector1).build();

        assertEquals(1, vector2.getValue(0));
        for (int i = 1; i < 8; i++) {
            assertEquals(vector1.getValue(i - 1), vector2.getValue(i));
        }
    }

    @Test
    void testClose() {

        IntColumnVector.IntColumnVectorBuilder builder = new IntColumnVector.IntColumnVectorBuilder(4);

        HostMemoryBuffer mockDataBuffer = spy(builder.getDataBuffer());

        builder.setDataBuffer(mockDataBuffer);
        builder.append(2).append(3).append(5).appendNull();
        HostMemoryBuffer mockValidBuffer = spy(builder.getValidityBuffer());
        builder.setValidBuffer(mockValidBuffer);

        builder.close();
        Mockito.verify(mockDataBuffer).close();
        Mockito.verify(mockValidBuffer).close();
    }

    @Test
    public void testAdd() {

        assumeTrue(CommonApi.libraryLoaded());
        IntColumnVector.IntColumnVectorBuilder columnVectorBuilder1 = new IntColumnVector.IntColumnVectorBuilder(4);
        IntColumnVector intColumnVector1 = columnVectorBuilder1.append(1).append(2).append(3).append(4).build();
        IntColumnVector.IntColumnVectorBuilder columnVectorBuilder2 = new IntColumnVector.IntColumnVectorBuilder(4);
        IntColumnVector intColumnVector2 = columnVectorBuilder2.append(10).append(20).append(30).append(40).build();

        intColumnVector1.toDeviceBuffer();
        intColumnVector2.toDeviceBuffer();

        IntColumnVector intColumnVector3 = intColumnVector1.add(intColumnVector2);

        intColumnVector3.toHostBuffer();
        for (int i = 0; i < 4; i++){
            long v1=intColumnVector1.getValue(i);
            long v2=intColumnVector2.getValue(i);
            long v3=intColumnVector3.getValue(i);
            assertEquals( v1+v2, v3);
            log.debug("{}",v3);
        }

    }

}

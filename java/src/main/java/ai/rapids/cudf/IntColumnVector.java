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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class IntColumnVector extends ColumnVector {

    private static Logger log = LoggerFactory.getLogger(IntColumnVector.class);

    /**
     * Private constructor to use the BuilderPattern
     */
    private IntColumnVector(HostMemoryBuffer data, HostMemoryBuffer bitmask, long nullCount, long rows) {
        super(data, bitmask, nullCount, rows);
    }

    /**
     * Private constructor for creating a device vector
     */
    private IntColumnVector(DeviceMemoryBuffer data, DeviceMemoryBuffer bitmask, long nullCount, long rows) {
        super(data, bitmask, nullCount, rows);
    }

    /**
     * Get the value pointed by the index
     * @param index
     * @return
     * @throws IndexOutOfBoundsException
     * @throws {@link NullPointerException}
     */
    public final int getValue(long index) throws IndexOutOfBoundsException, NullPointerException {
        checkBeforeGetValue(index);
        return hostData.data.getInt(index * DType.CUDF_INT32.sizeInBytes);

    }

    private void checkBeforeGetValue(long index) {
        if (index < 0 || index >= rows) {
            throw new IndexOutOfBoundsException(String.valueOf(index));
        }
        if (hasNulls() && isNull(index)) {
            throw new NullPointerException("value at index: " + index + " is null");
        }
        if (hostData == null) {
            throw new UnsupportedOperationException("Cannot access data from device buffer, transfer to host first");
        }
    }

    /**
     * This is a factory method to create a vector on the GPU with the intention that the caller will populate it
     * @param rows
     * @param bitmask
     * @return
     */
    static IntColumnVector newOutputVector(long rows, boolean bitmask) {
        DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(rows * DType.CUDF_INT32.sizeInBytes);
        DeviceMemoryBuffer valid = null;
        if (bitmask) {
            valid = DeviceMemoryBuffer.allocate(getValidityBufferSize(rows));
        }
        return new IntColumnVector(data, valid, 0L, rows);
    }

    /**
     * A method to add two vectors
     * Preconditions - vectors have to be the same size
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     * Example:
     *          IntColumnVector v1 = IntColumnVector.IntColumnVectorBuilder().append(1).append(5)...build();
     *          IntColumnVector v2 = IntColumnVector.IntColumnVectorBuilder().append(5).append(13)...build();
     *
     *          IntColumnVector v3 = v1.add(v2);
     *          ...
     *          v3.close();
     *
     * @param v1 - vector to be added to this vector.
     * @return - A new vector allocated on the GPU.
     */
    public IntColumnVector add(IntColumnVector v1) {
        if (v1.getSize() != getSize()) {
            throw new IllegalStateException("Vectors size mismatch");
        }

        IntColumnVector result = IntColumnVector.newOutputVector(v1.getSize(), hasNulls() || v1.hasNulls());
        Cudf.gdfAddI32(getCudfColumn(DType.CUDF_INT32), v1.getCudfColumn(DType.CUDF_INT32),
                                                                        result.getCudfColumn(DType.CUDF_INT32));
        return result;
    }

    private static long getValidityBufferSize(long rows) {
        long numBytes = Math.max(rows / 8, 1);
        return ((numBytes + 7) / 8) * 8;
    }

    /**
     * Builder for IntColumnVector to make it immutable
     */
    public static final class IntColumnVectorBuilder implements AutoCloseable {
        private HostMemoryBuffer data;
        private HostMemoryBuffer valid;
        private long currentIndex;
        private long nullCount;
        private long rows;
        private boolean built;

        /**
         * Getter method for testing
         * @return @HostMemoryBuffer that holds the data
         */
        final HostMemoryBuffer getDataBuffer() {
            return data;
        }

        /**
         * Setter method for testing
         * @param data - Set the data buffer
         */
        final void setDataBuffer(HostMemoryBuffer data) {
            this.data = data;
        }

        /**
         * Getter method for testing
         * @return @HostMemoryBuffer that holds the validity mask
         */
        final HostMemoryBuffer getValidityBuffer() {
            return valid;
        }

        /**
         * Setter method for testing
         * @param valid  @HostMemoryBuffer that holds the validity mask
         */
        final void setValidBuffer(HostMemoryBuffer valid) {
            this.valid = valid;
        }

        /**
         * Create a builder with a buffer of size rows
         * @param rows
         */
        public IntColumnVectorBuilder(long rows) {
            this.rows = rows;
            this.data = HostMemoryBuffer.allocate(rows * DType.CUDF_INT32.sizeInBytes);
            this.currentIndex = 0;
        }

        /**
         * Build the immutable @IntColumnVector
         * @return  - The IntColumnVector based on this builder values
         */
        public final IntColumnVector build() {
            //do the magic
            built = true;
            return new IntColumnVector(data, valid, nullCount, rows);
        }

        /**
         * Append this vector to the end of this vector
         * @param intColumnVector - Vector to be added
         * @return  - The IntColumnVector based on this builder values
         */
        public final IntColumnVectorBuilder append(IntColumnVector intColumnVector) {
            if (intColumnVector.rows > rows - currentIndex) {
                throw new IndexOutOfBoundsException("Not enough space to copy column vector");
            }

            data.copyRange(currentIndex * DType.CUDF_INT32.sizeInBytes, intColumnVector.hostData.data,
                                                            0L, intColumnVector.hostData.data.getLength());

            if (intColumnVector.hasNulls()) {
                valid.copyRange(currentIndex * DType.CUDF_INT32.sizeInBytes, intColumnVector.hostData.valid,
                                                            0L, intColumnVector.hostData.data.getLength());
            }
            currentIndex += intColumnVector.rows;
            return this;
        }

        /**
         * Append value to the next open index
         * @param value - value to be appended
         * @return - IntColumnVector
         * @throws - {@link IndexOutOfBoundsException}
         */
        public final IntColumnVectorBuilder append(int value) throws IndexOutOfBoundsException {
            if (currentIndex >= rows) {
                throw new IndexOutOfBoundsException("vector allocated for: " + rows + " is full");
            }
            //add value to the array
            setValues(currentIndex, value, true);
            currentIndex++;
            return this;
        }

        /**
         * Append null value to the next open index
         * @return
         */
        public final IntColumnVectorBuilder appendNull() {
            if (currentIndex >= rows) {
                throw new IndexOutOfBoundsException("vector allocated for: " + rows + " is full");
            }
            // add null
            if (this.valid == null) {
                long bitmaskSize = getValidityBufferSize(rows);
                this.valid = HostMemoryBuffer.allocate(bitmaskSize);
                valid.setMemory(0, bitmaskSize, (byte) 0xFF);
            }
            setValues(currentIndex, 0, false);
            currentIndex++;
            nullCount++;
            return this;
        }

        private void setValues(long index, int value, boolean isValid) {
            if (isValid) {
                data.setInt(index * DType.CUDF_INT32.sizeInBytes, value);
            } else {
                long bucket = index / 8;
                byte currentByte = valid.getByte(bucket);
                int bitmask = ~(1 << (index % 8));
                currentByte &= bitmask;
                valid.setByte(bucket, currentByte);
            }
        }

        /**
         * Close this builder and free memory if the IntColumnVector wasn't generated
         */
        @Override
        public void close() {
            if (!built) {
                data.close();
                if (valid != null) {
                    valid.close();
                }
            }
        }
    }
}

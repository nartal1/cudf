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

import java.util.function.Consumer;

public final class ByteColumnVector extends ColumnVector {

    /**
     * Private constructor to use the BuilderPattern.
     */
    private ByteColumnVector(HostMemoryBuffer data, HostMemoryBuffer validity, long rows, long nullCount) {
        super(data, validity, rows, DType.INT8, nullCount);
    }

    private ByteColumnVector(DeviceMemoryBuffer data, DeviceMemoryBuffer validity, long rows) {
        super(data, validity, rows, DType.INT8);
    }

    protected ByteColumnVector(CudfColumn cudfColumn) {
        super(cudfColumn);
        assert cudfColumn.getDtype() == DType.INT8;
    }

    /**
     * Get the value at index.
     */
    public final byte get(long index) {
        assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
        assert offHeap.hostData != null : "data is not on the host";
        assert !isNull(index) : " value at " + index + " is null";
        return offHeap.hostData.data.getByte(index * DType.INT8.sizeInBytes);
    }

    /**
     * This is a factory method to create a vector on the GPU with the intention that the caller will populate it. The
     * nullCount will be set lazily in cases when both given vectors (v1 and v2) have a validity vector
     * @param v1 - vector 1
     * @param v2 - vector 2
     * @return ByteColumnVector big enough to store the result
     */
    static ByteColumnVector newOutputVector(ByteColumnVector v1, ByteColumnVector v2) {
        assert v1.rows == v2.rows;
        return newOutputVector(v1.rows, v1.hasValidityVector() || v2.hasValidityVector());
    }

    /**
     * This is a factory method to create a vector on the GPU with the intention that the
     * caller will populate it.
     */
    static ByteColumnVector newOutputVector(long rows, boolean hasValidityVector) {
        DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(rows * DType.INT8.sizeInBytes);
        DeviceMemoryBuffer valid = null;
        if (hasValidityVector) {
            valid = DeviceMemoryBuffer.allocate(BitVectorHelper.getValidityAllocationSizeInBytes(rows));
        }
        return new ByteColumnVector(data, valid, rows);
    }

    /**
     * Create a new Builder to hold the specified number of rows.  Be sure to close the builder when done with it.
     * Please try to use {@see #build(int, Consumer)} instead to avoid needing to close the builder.
     * @param rows the number of rows this builder can hold
     * @return the builder to use.
     */
    public static Builder builder(int rows) { return new Builder(rows); }

    /**
     * Create a builder but with some things possibly replaced for testing.
     */
    static Builder builder(int rows, HostMemoryBuffer testData, HostMemoryBuffer testValid) {
        return new Builder(rows, testData, testValid);
    }

    /**
     * Create a new vector.
     * @param rows maximum number of rows that the vector can hold.
     * @param init what will initialize the vector.
     * @return the created vector.
     */
    public static ByteColumnVector build(int rows, Consumer<Builder> init) {
        try (Builder builder = builder(rows)) {
            init.accept(builder);
            return builder.build();
        }
    }

    /**
     * Create a new vector from the given values.
     */
    public static ByteColumnVector build(byte ... values) {
        return build(values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ByteColumnVector buildBoxed(Byte ... values) {
        return build(values.length, (b) -> b.appendBoxed(values));
    }

    public static final class Builder implements AutoCloseable {

        private final ColumnVector.Builder builder;

        /**
         * Create a builder with a buffer of size rows
         * @param rows number of rows to allocate.
         */
        private Builder(long rows) { builder = new ColumnVector.Builder(DType.INT8, rows); }

        /**
         * Create a builder with a buffer of size rows (for testing ONLY).
         * @param rows number of rows to allocate.
         * @param testData a buffer to hold the data (should be large enough to hold rows entries).
         * @param testValid a buffer to hold the validity vector (should be large enough to hold
         *                 rows entries or is null).
         */
        Builder(long rows, HostMemoryBuffer testData, HostMemoryBuffer testValid) {
            builder = new ColumnVector.Builder(DType.INT8, rows, testData, testValid);
        }

        /**
         * Build the immutable @ByteColumnVector. The rows of the vector will be equal to the appended values i.e. If a
         * larger buffer was allocated then the extra space will not be considered as part of the rows.
         * @return  - The ByteColumnVector based on this builder values
         */
        public final ByteColumnVector build() {
            //do the magic
            builder.built = true;
            return new ByteColumnVector(builder.data, builder.valid, builder.currentIndex, builder.nullCount);
        }

        /**
         * Append this vector to the end of this vector
         * @param byteColumnVector - Vector to be added
         * @return The Builder
         */
        public final Builder append(ByteColumnVector byteColumnVector) {
            builder.append(byteColumnVector);
            return this;
        }

        /**
         * Append multiple of the same value.
         * @param value what to append.
         * @param count how many to append.
         * @return this for chaining.
         */
        public final Builder append(byte value, long count) {
            builder.appendBytes(value, count);
            return this;
        }

        /**
         * Append value to the next open index
         * @param value - value to be appended
         * @return  this for chaining.
         * @throws  {@link IndexOutOfBoundsException}
         */
        public final Builder append(byte value) throws IndexOutOfBoundsException {
            builder.appendByte(value);
            return this;
        }

        /**
         * Append multiple values
         * @param values the values to append
         * @return  this for chaining.
         * @throws  {@link IndexOutOfBoundsException}
         */
        public final Builder appendArray(byte ... values) throws IndexOutOfBoundsException {
            builder.appendBytes(values);
            return this;
        }

        /**
         * Append multiple values.  This is very slow and should really only be used for tests.
         * @param values the values to append, including nulls.
         * @return  this for chaining.
         * @throws  {@link IndexOutOfBoundsException}
         */
        public final Builder appendBoxed(Byte ... values) throws IndexOutOfBoundsException {
            for (Byte b: values) {
                if (b == null) {
                    builder.appendNull();
                } else {
                    builder.appendByte(b);
                }
            }
            return this;
        }

        /**
         * Append null value.
         */
        public final Builder appendNull(){
            builder.appendNull();
            return this;
        }

        /**
         * Set a specific value to null
         * @param index the index to set it at.
         * @return this
         */
        public final Builder setNullAt(long index) {
            builder.setNullAt(index);
            return this;
        }

        /**
         * Close this builder and free memory if the ColumnVector wasn't generated
         */
        @Override
        public void close() { builder.close(); }
    }
}
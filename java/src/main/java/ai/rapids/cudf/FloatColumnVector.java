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

public final class FloatColumnVector extends ColumnVector {

    /**
     * Private constructor to use the BuilderPattern.
     */
    private FloatColumnVector(HostMemoryBuffer data, HostMemoryBuffer validity, long rows, long nullCount) {
        super(data, validity, rows, DType.CUDF_FLOAT32, nullCount);
    }

    private FloatColumnVector(DeviceMemoryBuffer data, DeviceMemoryBuffer validity, long rows) {
        super(data, validity, rows, DType.CUDF_FLOAT32);
    }

    /**
     * Get the value at index.
     */
    public final float get(long index) {
        assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
        assert hostData != null : "data is not on the host";
        assert !isNull(index) : " value at " + index + " is null";
        return hostData.data.getFloat(index * DType.CUDF_FLOAT32.sizeInBytes);
    }

    /**
     * This is a factory method to create a vector on the GPU with the intention that the caller will populate it. The
     * nullCount will be set lazily in cases when both given vectors (v1 and v2) have a validity vector
     * @param v1 - vector 1
     * @param v2 - vector 2
     * @return FloatColumnVector big enough to store the result
     */
    static FloatColumnVector newOutputVector(FloatColumnVector v1, FloatColumnVector v2) {
        assert v1.rows == v2.rows;
        return newOutputVector(v1.rows, v1.hasValidityVector() || v2.hasValidityVector());
    }

    /**
     * This is a factory method to create a vector on the GPU with the intention that the
     * caller will populate it.
     */
    static FloatColumnVector newOutputVector(long rows, boolean hasValidityVector) {
        DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(rows * DType.CUDF_FLOAT32.sizeInBytes);
        DeviceMemoryBuffer valid = null;
        if (hasValidityVector) {
            valid = DeviceMemoryBuffer.allocate(BitVectorHelper.getValidityAllocationSizeInBytes(rows));
        }
        return new FloatColumnVector(data, valid, rows);
    }

    /**
     * Add two vectors.
     * Preconditions - vectors have to be the same size
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     * Example:
     *          try (FloatColumnVector v1 = FloatColumnVector.build(5, (b) -> b.append(1).append(5)...);
     *               FloatColumnVector v2 = FloatColumnVector.build(5, (b) -> b.append(5).append(13)...);
     *               FloatColumnVector v3 = v1.add(v2);_ {}
     *            ...
     *          }
     *
     * @param v1 - vector to be added to this vector.
     * @return - A new vector allocated on the GPU.
     */
    public FloatColumnVector add(FloatColumnVector v1) {
        assert v1.getRows() == getRows(); // cudf will check this too.
        assert v1.getNullCount() == 0; // cudf add does not currently update nulls at all
        assert getNullCount() == 0;

        FloatColumnVector result = FloatColumnVector.newOutputVector(v1, this);
        Cudf.gdfAddF32(getCudfColumn(), v1.getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
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
    public static FloatColumnVector build(int rows, Consumer<Builder> init) {
        try (Builder builder = builder(rows)) {
            init.accept(builder);
            return builder.build();
        }
    }

    public static final class Builder implements AutoCloseable {

        private final ColumnVector.Builder builder;

        /**
         * Create a builder with a buffer of size rows
         * @param rows number of rows to allocate.
         */
        private Builder(long rows) {
            builder = new ColumnVector.Builder(DType.CUDF_FLOAT32, rows);
        }

        /**
         * Create a builder with a buffer of size rows (for testing ONLY).
         * @param rows number of rows to allocate.
         * @param testData a buffer to hold the data (should be large enough to hold rows entries).
         * @param testValid a buffer to hold the validity vector (should be large enough to hold
         *                 rows entries or is null).
         */
        Builder(long rows, HostMemoryBuffer testData, HostMemoryBuffer testValid) {
            builder = new ColumnVector.Builder(DType.CUDF_FLOAT32, rows, testData, testValid);
        }

        /**
         * Build the immutable @FloatColumnVector. The rows of the vector will be equal to the appended values i.e. If a
         * larger buffer was allocated then the extra space will not be considered as part of the rows.
         * @return  - The FloatColumnVector based on this builder values
         */
        public final FloatColumnVector build() {
            //do the magic
            builder.built = true;
            return new FloatColumnVector(builder.data, builder.valid, builder.currentIndex, builder.nullCount);
        }

        /**
         * Append this vector to the end of this vector
         * @param floatColumnVector - Vector to be added
         * @return  - The FloatColumnVector based on this builder values
         */
        public final Builder append(FloatColumnVector floatColumnVector) {
            builder.append(floatColumnVector);
            return this;
        }

        /**
         * Append multiple of the same value.
         * @param value what to append.
         * @param count how many to append.
         * @return this for chaining.
         */
        public final Builder append(float value, long count) {
            assert (count + builder.currentIndex) <= builder.rows;
            // If we are going to do this a lot we need a good way to memset more than a repeating byte.
            for (long i = 0; i < count; i++) {
                builder.appendFloat(value);
            }
            return this;
        }

        /**
         * Append value to the next open index
         * @param value - value to be appended
         * @return - FloatColumnVector
         * @throws - {@link IndexOutOfBoundsException}
         */
        public final Builder append(float value) throws IndexOutOfBoundsException {
            builder.appendFloat(value);
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
         * Close this builder and free memory if the ColumnVector wasn't generated
         */
        @Override
        public void close() {
            builder.close();
        }
    }
}
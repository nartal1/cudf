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

public final class TimestampColumnVector extends ColumnVector {

    /**
     * Private constructor to use the BuilderPattern.
     */
    private TimestampColumnVector(HostMemoryBuffer data, HostMemoryBuffer validity, long rows, long nullCount) {
        super(data, validity, rows, DType.TIMESTAMP, nullCount);
    }

    private TimestampColumnVector(DeviceMemoryBuffer data, DeviceMemoryBuffer validity, long rows) {
        super(data, validity, rows, DType.TIMESTAMP);
    }

    protected TimestampColumnVector(CudfColumn cudfColumn) {
        super(cudfColumn);
    }

    /**
     * Get the value at index.
     */
    public final long get(long index) {
        assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
        assert offHeap.hostData != null : "data is not on the host";
        assert !isNull(index) : " value at " + index + " is null";
        return offHeap.hostData.data.getLong(index * DType.TIMESTAMP.sizeInBytes);
    }

    /**
     * This is a factory method to create a vector on the GPU with the intention that the
     * caller will populate it.
     */
    static TimestampColumnVector newOutputVector(long rows, boolean hasValidityVector) {
        DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(rows * DType.TIMESTAMP.sizeInBytes);
        DeviceMemoryBuffer valid = null;
        if (hasValidityVector) {
            valid = DeviceMemoryBuffer.allocate(BitVectorHelper.getValidityAllocationSizeInBytes(rows));
        }
        return new TimestampColumnVector(data, valid, rows);
    }

    /**
     * Get year from Timestamp
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector year() {
        ColumnVector result = ColumnVector.newOutputVector(this.rows, this.hasValidityVector(), DType.INT16);
        Cudf.gdfExtractDatetimeYear(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Get month from Timestamp
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector month() {
        ColumnVector result = ColumnVector.newOutputVector(this.rows, this.hasValidityVector(), DType.INT16);
        Cudf.gdfExtractDatetimeMonth(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Get day from Timestamp
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector day() {
        ColumnVector result = ColumnVector.newOutputVector(this.rows, this.hasValidityVector(), DType.INT16);
        Cudf.gdfExtractDatetimeDay(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Get hour from Timestamp
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector hour() {
        ColumnVector result = ColumnVector.newOutputVector(this.rows, this.hasValidityVector(), DType.INT16);
        Cudf.gdfExtractDatetimeHour(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Get minute from Timestamp
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector minute() {
        ColumnVector result = ColumnVector.newOutputVector(this.rows, this.hasValidityVector(), DType.INT16);
        Cudf.gdfExtractDatetimeMinute(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Get second from Timestamp
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector second() {
        ColumnVector result = ColumnVector.newOutputVector(this.rows, this.hasValidityVector(), DType.INT16);
        Cudf.gdfExtractDatetimeSecond(getCudfColumn(), result.getCudfColumn());
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
    public static TimestampColumnVector build(int rows, Consumer<Builder> init) {
        try (Builder builder = builder(rows)) {
            init.accept(builder);
            return builder.build();
        }
    }

    /**
     * Create a new vector from the given values.
     */
    public static TimestampColumnVector build(long ... values) {
        return build(values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static TimestampColumnVector buildBoxed(Long ... values) {
        return build(values.length, (b) -> b.appendBoxed(values));
    }

    public static final class Builder implements AutoCloseable {

        private final ColumnVector.Builder builder;

        /**
         * Create a builder with a buffer of size rows
         * @param rows number of rows to allocate.
         */
        private Builder(long rows) {
            builder = new ColumnVector.Builder(DType.TIMESTAMP, rows);
        }

        /**
         * Create a builder with a buffer of size rows (for testing ONLY).
         * @param rows number of rows to allocate.
         * @param testData a buffer to hold the data (should be large enough to hold rows entries).
         * @param testValid a buffer to hold the validity vector (should be large enough to hold
         *                 rows entries or is null).
         */
        Builder(long rows, HostMemoryBuffer testData, HostMemoryBuffer testValid) {
            builder = new ColumnVector.Builder(DType.TIMESTAMP, rows, testData, testValid);
        }

        /**
         * Build the immutable @TimestampColumnVector. The rows of the vector will be equal to the appended values i.e. If a
         * larger buffer was allocated then the extra space will not be considered as part of the rows.
         * @return  - The TimestampColumnVector based on this builder values
         */
        public final TimestampColumnVector build() {
            //do the magic
            builder.built = true;
            return new TimestampColumnVector(builder.data, builder.valid, builder.currentIndex, builder.nullCount);
        }

        /**
         * Append this vector to the end of this vector
         * @param timestampColumnVector - Vector to be added
         * @return  - The Builder
         */
        public final Builder append(TimestampColumnVector timestampColumnVector) {
            builder.append(timestampColumnVector);
            return this;
        }

        /**
         * Append multiple of the same value.
         * @param value what to append.
         * @param count how many to append.
         * @return this for chaining.
         */
        public final Builder append(long value, long count) {
            assert (count + builder.currentIndex) <= builder.rows;
            for (long i = 0; i < count; i++) {
                builder.appendLong(value);
            }
            return this;
        }

        /**
         * Append value to the next open index
         * @param value  value to be appended
         * @return  this for chaining
         * @throws  {@link IndexOutOfBoundsException}
         */
        public final Builder append(long value) throws IndexOutOfBoundsException {
            builder.appendLong(value);
            return this;
        }

        /**
         * Append multiple values
         * @param values to append
         * @return - this for chaining
         * @throws - {@link IndexOutOfBoundsException}
         */
        public final Builder appendArray(long ... values) throws IndexOutOfBoundsException {
            builder.appendLongs(values);
            return this;
        }

        /**
         * Append multiple values.  This is very slow and should really only be used for tests.
         * @param values the values to append, including nulls.
         * @return  this for chaining.
         * @throws  {@link IndexOutOfBoundsException}
         */
        public final Builder appendBoxed(Long ... values) throws IndexOutOfBoundsException {
            for (Long b: values) {
                if (b == null) {
                    builder.appendNull();
                } else {
                    builder.appendLong(b);
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
        public void close() {
            builder.close();
        }
    }
}

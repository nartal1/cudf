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

/**
 * Abstract class depicting a Column Vector. This class represents the immutable vector created by the Builders from
 * each respective ColumnVector subclasses.  This class holds references to off heap memory and is
 * reference counted to know when to release it.  Call close to decrement the reference count when
 * you are done with the column, and call inRefCount to increment the reference count.
 */
public abstract class ColumnVector implements AutoCloseable {
    private static Logger log = LoggerFactory.getLogger(ColumnVector.class);

    protected long rows;
    protected final DType type;
    protected BufferEncapsulator<HostMemoryBuffer> hostData;
    protected BufferEncapsulator<DeviceMemoryBuffer> deviceData;
    protected long nullCount;
    private CudfColumn cudfColumn;
    private int refCount;


    protected ColumnVector(HostMemoryBuffer hostDataBuffer,
                           HostMemoryBuffer hostValidityBuffer, long rows, DType type, long nullCount) {
        if (nullCount > 0 && hostValidityBuffer == null) {
            throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
        }
        this.hostData = new BufferEncapsulator(hostDataBuffer, hostValidityBuffer);
        this.deviceData = null;
        this.rows = rows;
        this.nullCount = nullCount;
        this.type = type;
        refCount = 1;
    }

    protected ColumnVector(DeviceMemoryBuffer dataBuffer,
                           DeviceMemoryBuffer validityBuffer, long rows, DType type) {
        this.deviceData = new BufferEncapsulator(dataBuffer, validityBuffer);
        this.hostData = null;
        this.rows = rows;
        // This should be overwritten, as this constructor is just for output
        this.nullCount = 0;
        this.type = type;
        refCount = 1;
    }

    /**
     * Increment the reference count for this column.  You need to call close on this
     * to decrement the reference count again.
     */
    public void incRefCount() {
        refCount++;
    }

    /**
     * Returns the number of rows in this vector.
     */
    public final long getRows() {
        return rows;
    }

    /**
     * Returns the type of this vector.
     */
    public final DType getType() {
        return type;
    }

    /**
     * Close this Vector and free memory allocated for HostMemoryBuffer and DeviceMemoryBuffer
     */
    public final void close() {
        refCount --;
        if (refCount == 0) {
            if (hostData != null) {
                hostData.close();
                hostData = null;
            }
            if (deviceData != null) {
                deviceData.close();
                deviceData = null;
            }
            if (cudfColumn != null) {
                cudfColumn.close();
                cudfColumn = null;
            }
        } else if (refCount < 0) {
            log.error("Close called too many times on %s", this);
        }
    }

    /**
     * Returns the number of nulls in the data.
     */
    public final long getNullCount(){
        return nullCount;
    }

    private void checkDeviceData() {
        if (deviceData == null) {
            throw new IllegalStateException("Vector not on Device");
        }
    }

    private void checkHostData() {
        if (hostData == null) {
            throw new IllegalStateException("Vector not on Host");
        }
    }

    /**
     * Returns if the vector has a validity vector allocated or not.
     */
    public final boolean hasValidityVector() {
        boolean ret;
        if (hostData != null) {
            ret = (hostData.valid != null);
        } else {
            ret = (deviceData.valid != null);
        }
        return ret;
    }

    /**
     * Returns if the vector has nulls
     */
    public final boolean hasNulls() {
        return getNullCount() > 0;
    }

    public final boolean isNull(long index) {
        checkHostData();
        if (hasNulls()) {
            return BitVectorHelper.isNull(hostData.valid, index);
        }
        return false;
    }

    /**
     * Copies the HostBuffer data to DeviceBuffer
     */
    public final void toDeviceBuffer() {
        checkHostData();
        if (deviceData == null) {
            DeviceMemoryBuffer deviceDataBuffer = DeviceMemoryBuffer.allocate(hostData.data.getLength());
            DeviceMemoryBuffer deviceValidityBuffer = null;
            boolean needsCleanup = true;
            try {
                if (hasNulls()) {
                    deviceValidityBuffer = DeviceMemoryBuffer.allocate(hostData.valid.getLength());
                }
                deviceData = new BufferEncapsulator(deviceDataBuffer, deviceValidityBuffer);
                needsCleanup = false;
            } finally {
                if (needsCleanup) {
                    if (deviceDataBuffer != null) {
                        deviceDataBuffer.close();
                    }
                    if (deviceValidityBuffer != null) {
                        deviceValidityBuffer.close();
                    }
                }
            }
        }
        deviceData.data.copyFromHostBuffer(hostData.data);
        if (deviceData.valid != null) {
            deviceData.valid.copyFromHostBuffer(hostData.valid);
        }
    }

    /**
     * Copies the DeviceBuffer data to HostBuffer
     *
     */
    public final void toHostBuffer() {
        checkDeviceData();
        if (hostData == null) {
            HostMemoryBuffer hostDataBuffer = HostMemoryBuffer.allocate(deviceData.data.getLength());
            HostMemoryBuffer hostValidityBuffer = null;
            boolean needsCleanup = true;
            try {
                if (deviceData.valid != null) {
                    hostValidityBuffer = HostMemoryBuffer.allocate(deviceData.valid.getLength());
                }
                hostData = new BufferEncapsulator(hostDataBuffer, hostValidityBuffer);
                needsCleanup = false;
            } finally {
                if (needsCleanup) {
                    if (hostDataBuffer != null) {
                        hostDataBuffer.close();
                    }
                    if (hostValidityBuffer != null) {
                        hostValidityBuffer.close();
                    }
                }
            }
            hostData.data.copyFromDeviceBuffer(deviceData.data);
            if (hostData.valid != null) {
                hostData.valid.copyFromDeviceBuffer(deviceData.valid);
            }
        }
    }

    @Override
    public String toString() {
        return "ColumnVector{" +
                "rows=" + rows +
                ", type=" + type +
                ", hostData=" + hostData +
                ", deviceData=" + deviceData +
                ", nullCount=" + nullCount +
                ", cudfColumn=" + cudfColumn +
                '}';
    }

    protected final CudfColumn getCudfColumn() {
        if (cudfColumn == null) {
            assert rows <= Integer.MAX_VALUE;
            assert getNullCount() <= Integer.MAX_VALUE;
            cudfColumn = new CudfColumn(deviceData.data.getAddress(),
                    (deviceData.valid == null ? 0 : deviceData.valid.getAddress()),
                    (int)rows,
                    type, (int) getNullCount(), CudfTimeUnit.NONE);
        }
        return cudfColumn;
    }

    /**
     * Update any internal accounting from what is in the Native Code
     */
    protected void updateFromNative() {
        assert cudfColumn != null;
        this.nullCount = cudfColumn.getNullCount();
        this.rows = cudfColumn.getSize();
    }

    /**
     * Encapsulator class to hold the two buffers and nullcount as a cohesive object
     */
    protected static final class BufferEncapsulator<T extends MemoryBuffer> implements AutoCloseable {
        public final T data;
        public final T valid;

        BufferEncapsulator(T data, T valid) {
            this.data = data;
            this.valid = valid;
        }

        @Override
        public String toString() {
            return "BufferEncapsulator{type= " + data.getClass().getSimpleName()
                    + ", data= " + data
                    + ", valid= " + valid +"}";
        }

        @Override
        public void close() {
            data.close();
            if (valid != null) {
                valid.close();
            }
        }
    }


    /**
     * Base class for Builder
     */
    static final class Builder implements AutoCloseable {
        HostMemoryBuffer data;
        HostMemoryBuffer valid;
        long currentIndex = 0;
        long nullCount;
        final long rows;
        boolean built;
        final DType type;


        /**
         * Create a builder with a buffer of size rows
         * @param type datatype
         * @param rows number of rows to allocate.
         */
        Builder(DType type, long rows) {
            this.type=type;
            this.rows = rows;
            this.data=HostMemoryBuffer.allocate(rows * type.sizeInBytes);
        }

        /**
         * Create a builder with a buffer of size rows (for testing ONLY).
         * @param type datatype
         * @param rows number of rows to allocate.
         * @param testData a buffer to hold the data (should be large enough to hold rows entries).
         * @param testValid a buffer to hold the validity vector (should be large enough to hold
         *                 rows entries or is null).
         */
        Builder(DType type, long rows, HostMemoryBuffer testData, HostMemoryBuffer testValid) {
            this.type = type;
            this.rows = rows;
            this.data = testData;
            this.valid = testValid;
        }

        final void appendShort(short value) {
            assert type == DType.CUDF_INT16;
            assert currentIndex < rows;
            data.setShort(currentIndex *  type.sizeInBytes, value);
            currentIndex++;
        }

        final void appendInt(int value) {
            assert (type == DType.CUDF_INT32 || type == DType.CUDF_DATE32);
            assert currentIndex < rows;
            data.setInt(currentIndex *  type.sizeInBytes, value);
            currentIndex++;
        }

        final void appendLong(long value) {
            assert type == DType.CUDF_INT64;
            assert currentIndex < rows;
            data.setLong(currentIndex * type.sizeInBytes, value);
            currentIndex++;
        }

        final void appendFloat(float value) {
            assert type == DType.CUDF_FLOAT32;
            assert currentIndex < rows;
            data.setFloat(currentIndex * type.sizeInBytes, value);
            currentIndex++;
        }

        final void appendDouble(double value) {
            assert type == DType.CUDF_FLOAT64;
            assert currentIndex < rows;
            data.setDouble(currentIndex * type.sizeInBytes, value);
            currentIndex++;
        }

        void allocateBitmaskAndSetDefaultValues() {
            long bitmaskSize = BitVectorHelper.getValidityAllocationSizeInBytes(rows);
            valid = HostMemoryBuffer.allocate(bitmaskSize);
            valid.setMemory(0, bitmaskSize, (byte) 0xFF);
        }

        /**
         * Append this vector to the end of this vector
         * @param columnVector - Vector to be added
         * @return  - The ColumnVector based on this builder values
         */
        final Builder append(ColumnVector columnVector) {
            assert columnVector.rows <= (rows - currentIndex);
            assert columnVector.type == type;
            assert columnVector.hostData != null;

            data.copyRange(currentIndex * type.sizeInBytes, columnVector.hostData.data,
                    0L,
                    columnVector.getRows() * type.sizeInBytes);

            if (columnVector.nullCount != 0) {
                if (valid == null) {
                    allocateBitmaskAndSetDefaultValues();
                }
                //copy values from intColumnVector to this
                BitVectorHelper.append(columnVector.hostData.valid, valid, currentIndex, columnVector.rows);
                nullCount += columnVector.nullCount;
            }
            currentIndex += columnVector.rows;
            return this;
        }

        /**
         * Append null value.
         */
        void appendNull() {
            assert currentIndex < rows;

            // add null
            if (this.valid == null) {
                allocateBitmaskAndSetDefaultValues();
            }
            BitVectorHelper.appendNull(valid,currentIndex);
            currentIndex++;
            nullCount++;
        }

        /**
         * Close this builder and free memory if the ColumnVector wasn't generated
         */
        @Override
        public void close() {
            if (!built) {
                data.close();
                data = null;
                if (valid != null) {
                    valid.close();
                    valid = null;
                }
                built = true;
            }
        }

        @Override
        public String toString() {
            return "Builder{" +
                    "data=" + data +
                    "type=" + type +
                    ", valid=" + valid +
                    ", currentIndex=" + currentIndex +
                    ", nullCount=" + nullCount +
                    ", rows=" + rows +
                    ", built=" + built +
                    '}';
        }
    }
}

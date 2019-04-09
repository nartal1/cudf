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

/**
 * Abstract class depicting a Column Vector. This class represents the immutable vector created by the Builders from
 * each respective ColumnVector subclasses
 */
public abstract class ColumnVector implements AutoCloseable {

    protected long rows;
    protected final DType type;
    protected BufferEncapsulator<HostMemoryBuffer> hostData;
    protected BufferEncapsulator<DeviceMemoryBuffer> deviceData;
    protected long nullCount;
    private CudfColumn cudfColumn;


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

    }

    protected ColumnVector(DeviceMemoryBuffer dataBuffer,
                           DeviceMemoryBuffer validityBuffer, long rows, DType type) {
        this.deviceData = new BufferEncapsulator(dataBuffer, validityBuffer);
        this.hostData = null;
        this.rows = rows;
        // This should be overwritten, as this constructor is just for output
        this.nullCount = 0;
        this.type = type;
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
    public void close() {
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
            int b = hostData.valid.getByte(index / 8);
            int i = b & (1 << (index % 8));
            return i == 0;
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
        return "ColumnVector{type= " + this.getClass().getSimpleName()
                + " hostData=" + hostData
                + ", deviceData=" + deviceData
                + ", rows=" + rows + "}";
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

}

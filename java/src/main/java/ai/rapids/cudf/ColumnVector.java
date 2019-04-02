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
 * each respective ColumnVector subclasses
 */
public abstract class ColumnVector implements AutoCloseable {

    protected final long rows;
    protected BufferEncapsulator<HostMemoryBuffer> hostData;
    protected BufferEncapsulator<DeviceMemoryBuffer> deviceData;
    protected final long nullCount;
    private CudfColumn cudfColumn;


    protected ColumnVector(HostMemoryBuffer hostDataBuffer,
                           HostMemoryBuffer hostValidityBuffer, long nullCount, long rows) {
        if (nullCount > 0 && hostValidityBuffer == null) {
            throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
        }
        this.hostData = new BufferEncapsulator(hostDataBuffer, hostValidityBuffer);
        this.deviceData = null;
        this.rows = rows;
        this.nullCount = nullCount;
    }

    protected ColumnVector(DeviceMemoryBuffer deviceDataBuffer, DeviceMemoryBuffer deviceValidityBuffer,
                           long nullCount, long rows) {
        if (nullCount > 0 && deviceValidityBuffer == null) {
            throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
        }
        this.deviceData = new BufferEncapsulator(deviceDataBuffer, deviceValidityBuffer);
        this.rows = rows;
        this.hostData = null;
        this.nullCount = nullCount;
    }

    /**
     * Return the allocated size of this vector based on the type -
     * e.g. IntColumnVector v = {1,2,3}
     *      v.getSize() = 3 and not 12 which is the allocated size
     * @return
     */
    public final long getSize() {
        return rows;
    }

    /**
     * Close this Vector and free memory allocated for HostMemoryBuffer and DeviceMemoryBuffer
     */
    public void close() {
        if (hostData != null) {
            hostData.data.close();
            if (hostData.valid !=null) {
                hostData.valid.close();
            }
            hostData = null;
        }
        if (deviceData != null) {
            deviceData.data.close();
            if (deviceData.valid != null ) {
                deviceData.valid.close();
            }
            deviceData = null;
        }
        if (cudfColumn != null) {
            cudfColumn.close();
            cudfColumn = null;
        }
    }

    /**
     * Returns the number of null data pointers
     * @return
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
     * Returns if the vector has nulls
     * @return - true, if it has nulls, else, false
     */
    public final boolean hasNulls() {
        return nullCount > 0;
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
            boolean validityBufferAlloc = false;
            try {
                if (hasNulls()) {
                    deviceValidityBuffer = DeviceMemoryBuffer.allocate(hostData.valid.getLength());
                }
                deviceData = new BufferEncapsulator(deviceDataBuffer, deviceValidityBuffer);
                validityBufferAlloc = true;
            } finally {
                if (!validityBufferAlloc) {
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
                boolean validityBufferAlloc = false;
                try {
                    if (hasNulls()) {
                        hostValidityBuffer = HostMemoryBuffer.allocate(deviceData.valid.getLength());
                    }
                    hostData = new BufferEncapsulator(hostDataBuffer, hostValidityBuffer);
                    validityBufferAlloc = true;
                } finally {
                    if (!validityBufferAlloc) {
                        if (hostDataBuffer != null) {
                            hostDataBuffer.close();
                        }
                        if (hostValidityBuffer != null) {
                            hostValidityBuffer.close();
                        }
                    }
                }
            }
            hostData.data.copyFromDeviceBuffer(deviceData.data);
            if (hostData.valid != null) {
                hostData.valid.copyFromDeviceBuffer(deviceData.valid);
            }
    }

    @Override
    public String toString() {
        return "ColumnVector{type= " + this.getClass().getSimpleName()
                + " hostData=" + hostData
                + ", deviceData=" + deviceData
                + ", rows=" + rows + "}";
    }

    protected final CudfColumn getCudfColumn(DType type) {
        if (cudfColumn == null) {
            cudfColumn = new CudfColumn(deviceData.data.getAddress(),
                    (deviceData.valid == null ? 0 : deviceData.valid.getAddress()),
                    (int) deviceData.data.getLength(),
                    type, (int) getNullCount(), CudfTimeUnit.NONE);
        }
        return cudfColumn;
    }

    /**
     * Encapsulator class to hold the two buffers and nullcount as a cohesive object
     */
    protected final class BufferEncapsulator<T extends MemoryBuffer> {
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
    }

}

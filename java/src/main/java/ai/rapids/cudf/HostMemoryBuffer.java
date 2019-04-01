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
 * This class represents a Address held in the host memory
 */
class HostMemoryBuffer extends MemoryBuffer {

    private static Logger log = LoggerFactory.getLogger(HostMemoryBuffer.class);

    // protected

    private HostMemoryBuffer(long address, long length) {
        super(address, length);
    }

    /**
     * Method to copy from a DeviceMemoryBuffer to a HostMemoryBuffer
     * @param deviceMemoryBuffer - Buffer to copy data from
     */
    public void copyFromDeviceBuffer(DeviceMemoryBuffer deviceMemoryBuffer) {
        Cuda.memcpy(address, deviceMemoryBuffer.address, deviceMemoryBuffer.length,
                                                                  CudaMemcpyKind.DEVICE_TO_HOST);
    }

    /**
     * Factory method to create this buffer
     * @param bytes - size in bytes to allocate
     * @return - return this newly created buffer
     */
    public static HostMemoryBuffer allocate(long bytes) {
        return new HostMemoryBuffer(UnsafeMemoryAccessor.allocate(bytes), bytes);
    }

    private void addressOutOfBoundsCheck(long address) throws IndexOutOfBoundsException {
        if (address < this.address || address >= this.address + length) {
            throw new IndexOutOfBoundsException(String.valueOf(address));
        }
    }

    private void checkUpperAndLowerBounds(long address, DType type) {
        addressOutOfBoundsCheck(address);
        addressOutOfBoundsCheck(type.sizeInBytes - 1 + address);
    }

    /**
     * Returns the Integer value at that offset
     * @param offset - offset from the address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public final int getInt(long offset) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_INT32);
        return UnsafeMemoryAccessor.getInt(requestedAddress);
    }

    /**
     * Sets the Integer value at that offset
     * @param offset - offset from the address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public final void setInt(long offset, int value) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_INT32);
        UnsafeMemoryAccessor.setInt(requestedAddress, value);
    }

    /**
     * Returns the Byte value at that offset
     * @param offset - offset from the address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public final byte getByte(long offset) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_INT8);
        return UnsafeMemoryAccessor.getByte(requestedAddress);
    }

    /**
     * Sets the Long value at that offset
     * @param offset - offset from the address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public final void setLong(long offset, long value) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_INT64);
        UnsafeMemoryAccessor.setLong(requestedAddress, value);
    }

    /**
     * Sets the Byte value at that offset
     * @param offset - offset from the address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public final void setByte(long offset, byte value) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_INT8);
        UnsafeMemoryAccessor.setByte(requestedAddress, value);
    }

    /**
     * Sets the values in this buffer repeatedly
     * @param offset - offset from the address
     * @param length - number of bytes to set
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public final void setMemory(long offset, long length, byte value) throws IndexOutOfBoundsException {
        addressOutOfBoundsCheck(address + offset + length - 1);
        UnsafeMemoryAccessor.setMemory(address + offset, length, value);
    }

    public final void copyMemory(long fromAddress, long len) {
        addressOutOfBoundsCheck(address + len - 1);
        UnsafeMemoryAccessor.copyMemory(null, fromAddress, null, address, len);
    }
    /**
     * Returns the Long value at that offset
     * @param offset - offset from the address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public final long getLong(long offset) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_INT64);
        return UnsafeMemoryAccessor.getLong(requestedAddress);
    }

    /**
     * Sets the Short value at that offset
     * @param offset - offset from the address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public final void setShort(long offset, short value) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_INT16);
        UnsafeMemoryAccessor.setShort(requestedAddress, value);
    }

    /**
     * Returns the Short value at that offset
     * @param offset - offset from the address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public final short getShort(long offset) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_INT16);
        return UnsafeMemoryAccessor.getShort(requestedAddress);
    }

    /**
     * Sets the Double value at that offset
     * @param offset - offset from the address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public final void setDouble(long offset, double value) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_FLOAT64);
        UnsafeMemoryAccessor.setDouble(requestedAddress, value);
    }

    /**
     * Returns the Double value at that offset
     * @param offset - offset from the address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public final double getDouble(long offset) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_FLOAT64);
        return UnsafeMemoryAccessor.getDouble(requestedAddress);
    }

    /**
     * Sets the Float value at that offset
     * @param offset - offset from the address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public final void setFloat(long offset, float value) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_FLOAT32);
        UnsafeMemoryAccessor.setFloat(requestedAddress, value);
    }

    /**
     * Returns the Float value at that offset
     * @param offset - offset from the address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public final float getFloat(long offset) throws IndexOutOfBoundsException {
        long requestedAddress = this.address + offset;
        checkUpperAndLowerBounds(requestedAddress, DType.CUDF_FLOAT32);
        return UnsafeMemoryAccessor.getFloat(requestedAddress);
    }

    @Override
    public void close() {
        UnsafeMemoryAccessor.free(address);
    }

    /**
     * Append the contents of the given buffer to this buffer
     * @param currentOffset
     * @param hostData - Buffer to be copied from
     * @param startIndex
     * @param length
     */
    public void copyRange(long currentOffset, HostMemoryBuffer hostData, long startIndex, long length) {
        addressOutOfBoundsCheck(address + length - 1);
        UnsafeMemoryAccessor.copyMemory(null, hostData.address + startIndex, null,
                                                            address + currentOffset, hostData.getLength());
    }
}

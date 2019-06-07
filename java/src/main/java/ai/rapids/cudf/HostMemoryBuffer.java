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
 * This class represents an address held in the host memory.
 */
class HostMemoryBuffer extends MemoryBuffer {
  HostMemoryBuffer(long address, long length) {
    super(address, length);
  }

  /**
   * Factory method to create this buffer
   * @param bytes - size in bytes to allocate
   * @return - return this newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes) {
    return new HostMemoryBuffer(UnsafeMemoryAccessor.allocate(bytes), bytes);
  }

  /**
   * Returns the Byte value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final byte getByte(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 1, "getByte");
    return UnsafeMemoryAccessor.getByte(requestedAddress);
  }

  /**
   * Copy a set of bytes to an array from the buffer starting at offset.
   * @param dstOffset the offset from the address to start copying to
   * @param dst       the data to be copied.
   */
  public final void getBytes(byte[] dst, long dstOffset, long srcOffset, long len) {
    assert len > 0;
    assert len <= dst.length - dstOffset;
    assert srcOffset >= 0;
    long requestedAddress = this.address + srcOffset;
    addressOutOfBoundsCheck(requestedAddress, len, "getBytes");
    UnsafeMemoryAccessor.getBytes(dst, dstOffset, requestedAddress, len);
  }

  /**
   * Sets the Byte value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setByte(long offset, byte value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 1, "setByte");
    UnsafeMemoryAccessor.setByte(requestedAddress, value);
  }

  /**
   * Copy a set of bytes from an array into the buffer at offset.
   * @param offset the offset from the address to start copying to
   * @param data   the data to be copied.
   */
  public final void setBytes(long offset, byte[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    assert srcOffset >= 0;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len, "setBytes");
    UnsafeMemoryAccessor.setBytes(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Short value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final short getShort(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 2, "getShort");
    return UnsafeMemoryAccessor.getShort(requestedAddress);
  }

  /**
   * Sets the Short value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setShort(long offset, short value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 2, "setShort");
    UnsafeMemoryAccessor.setShort(requestedAddress, value);
  }

  /**
   * Copy a set of shorts from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index in data to start at.
   */
  public final void setShorts(long offset, short[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 2, "setShorts");
    UnsafeMemoryAccessor.setShorts(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Integer value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final int getInt(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "getInt");
    return UnsafeMemoryAccessor.getInt(requestedAddress);
  }

  /**
   * Sets the Integer value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setInt(long offset, int value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "setInt");
    UnsafeMemoryAccessor.setInt(requestedAddress, value);
  }

  /**
   * Copy a set of ints from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at
   */
  public final void setInts(long offset, int[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 4, "setInts");
    UnsafeMemoryAccessor.setInts(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Long value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final long getLong(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "setLong");
    return UnsafeMemoryAccessor.getLong(requestedAddress);
  }

  /**
   * Sets the Long value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setLong(long offset, long value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "getLong");
    UnsafeMemoryAccessor.setLong(requestedAddress, value);
  }

  /**
   * Copy a set of longs from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at.
   */
  public final void setLongs(long offset, long[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 8, "setLongs");
    UnsafeMemoryAccessor.setLongs(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Float value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final float getFloat(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "getFloat");
    return UnsafeMemoryAccessor.getFloat(requestedAddress);
  }

  /**
   * Sets the Float value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setFloat(long offset, float value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "setFloat");
    UnsafeMemoryAccessor.setFloat(requestedAddress, value);
  }

  /**
   * Copy a set of floats from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at
   */
  public final void setFloats(long offset, float[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 4, "setFloats");
    UnsafeMemoryAccessor.setFloats(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Double value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final double getDouble(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "getDouble");
    return UnsafeMemoryAccessor.getDouble(requestedAddress);
  }

  /**
   * Sets the Double value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setDouble(long offset, double value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "setDouble");
    UnsafeMemoryAccessor.setDouble(requestedAddress, value);
  }

  /**
   * Copy a set of doubles from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at
   */
  public final void setDoubles(long offset, double[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 8, "setDoubles");
    UnsafeMemoryAccessor.setDoubles(requestedAddress, data, srcOffset, len);
  }

  /**
   * Sets the values in this buffer repeatedly
   * @param offset - offset from the address
   * @param length - number of bytes to set
   * @param value  - value to be set
   */
  public final void setMemory(long offset, long length, byte value) {
    addressOutOfBoundsCheck(address + offset, length, "set memory");
    UnsafeMemoryAccessor.setMemory(address + offset, length, value);
  }

  public final void copyFromMemory(long fromAddress, long len) {
    addressOutOfBoundsCheck(address, len, "copy from memory");
    UnsafeMemoryAccessor.copyMemory(null, fromAddress, null, address, len);
  }

  /**
   * Append the contents of the given buffer to this buffer
   * @param destOffset offset in bytes in this buffer to start copying to
   * @param srcData    Buffer to be copied from
   * @param srcOffset  offset in bytes to start copying from in srcData
   * @param length     number of bytes to copy
   */
  public final void copyFromHostBuffer(long destOffset, HostMemoryBuffer srcData, long srcOffset,
                                       long length) {
    addressOutOfBoundsCheck(address + destOffset, length, "copy from dest");
    srcData.addressOutOfBoundsCheck(srcData.address + srcOffset, length, "copy from source");
    UnsafeMemoryAccessor.copyMemory(null, srcData.address + srcOffset, null,
        address + destOffset, length);
  }

  /**
   * Method to copy from a DeviceMemoryBuffer to a HostMemoryBuffer
   * @param deviceMemoryBuffer - Buffer to copy data from
   */
  public final void copyFromDeviceBuffer(DeviceMemoryBuffer deviceMemoryBuffer) {
    addressOutOfBoundsCheck(address, deviceMemoryBuffer.length, "copy range dest");
    assert !deviceMemoryBuffer.closed;
    Cuda.memcpy(address, deviceMemoryBuffer.address, deviceMemoryBuffer.length,
        CudaMemcpyKind.DEVICE_TO_HOST);
  }

  @Override
  protected void doClose() {
    UnsafeMemoryAccessor.free(address);
  }
}

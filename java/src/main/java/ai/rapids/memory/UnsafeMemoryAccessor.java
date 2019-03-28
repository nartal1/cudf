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

package ai.rapids.memory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Field;

/**
 * UnsafeMemory Accessor for accessing memory on host
 */
public class UnsafeMemoryAccessor {

    private static Logger log = LoggerFactory.getLogger(UnsafeMemoryAccessor.class);

    private static final sun.misc.Unsafe UNSAFE;

    static {
        sun.misc.Unsafe unsafe = null;
        try {
            Field unsafeField = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
            unsafeField.setAccessible(true);
            unsafe = (sun.misc.Unsafe) unsafeField.get(null);
        } catch (Throwable t) {
            log.error("Failed to get unsafe object, got this error: ", t);
            UNSAFE = null;
            throw new NullPointerException("Failed to get unsafe object, got this error: " + t.getMessage());
        }
        UNSAFE = unsafe;
    }


    /**
     * Allocate bytes on host
     * @param bytes - number of bytes to allocate
     * @return - allocated address
     */
    public static long allocate(long bytes) {
        return UNSAFE.allocateMemory(bytes);
    }

    /**
     * Sets the Byte value at that address
     * @param address - memory address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public static void setByte(long address, byte value) {
        UNSAFE.putByte(address, value);
    }

    /**
     * Returns the Byte value at this address
     * @param address - memory address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public static byte getByte(long address) {
        return UNSAFE.getByte(address);
    }

    /**
     * Returns the Integer value at this address
     * @param address - memory address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public static int getInt(long address) {
        return UNSAFE.getInt(address);
    }

    /**
     * Sets the values at this address repeatedly
     * @param address - memory location
     * @param size - number of bytes to set
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public static void setMemory(long address, long size, byte value) {
        UNSAFE.setMemory(address, size, value);
    }

    /**
     * Sets the Integer value at that address
     * @param address - memory address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public static void setInt(long address, int value) {
        UNSAFE.putInt(address, value);
    }

    /**
     * Free memory at that location
     * @param address - memory location
     */
    public static void free(long address) {
        UNSAFE.freeMemory(address);
    }

    /**
     * Sets the Long value at that address
     * @param address - memory address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public static void setLong(long address, long value) {
        UNSAFE.putLong(address, value);
    }

    /**
     * Returns the Long value at this address
     * @param address - memory address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public static long getLong(long address) {
        return UNSAFE.getLong(address);
    }

    /**
     * Returns the Short value at this address
     * @param address - memory address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public static short getShort(long address) {
        return UNSAFE.getShort(address);
    }

    /**
     * Sets the Short value at that address
     * @param address - memory address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public static void setShort(long address, short value) {
        UNSAFE.putShort(address, value);
    }

    /**
     * Sets the Double value at that address
     * @param address - memory address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public static void setDouble(long address, double value) {
        UNSAFE.putDouble(address, value);
    }

    /**
     * Returns the Double value at this address
     * @param address - memory address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public static double getDouble(long address) {
        return UNSAFE.getDouble(address);
    }

    /**
     * Returns the Float value at this address
     * @param address - memory address
     * @return - value
     * @throws IndexOutOfBoundsException
     */
    public static float getFloat(long address) {
        return UNSAFE.getFloat(address);
    }

    /**
     * Sets the Float value at that address
     * @param address - memory address
     * @param value - value to be set
     * @throws IndexOutOfBoundsException
     */
    public static void setFloat(long address, float value) {
        UNSAFE.putFloat(address, value);
    }

    /**
     * Limits the number of bytes to copy per {@link Unsafe#copyMemory(long, long, long)} to
     * allow safepoint polling during a large copy.
     */
    private static final long UNSAFE_COPY_THRESHOLD = 1024L * 1024L;

    /**
     * Copy memory from one address to the other
     * @param src
     * @param srcOffset
     * @param dst
     * @param dstOffset
     * @param length
     */
    public static void copyMemory(Object src, long srcOffset, Object dst, long dstOffset, long length) {
        // Check if dstOffset is before or after srcOffset to determine if we should copy
        // forward or backwards. This is necessary in case src and dst overlap.
        if (dstOffset < srcOffset) {
            while (length > 0) {
                long size = Math.min(length, UNSAFE_COPY_THRESHOLD);
                UNSAFE.copyMemory(src, srcOffset, dst, dstOffset, size);
                length -= size;
                srcOffset += size;
                dstOffset += size;
            }
        } else {
            srcOffset += length;
            dstOffset += length;
            while (length > 0) {
                long size = Math.min(length, UNSAFE_COPY_THRESHOLD);
                srcOffset -= size;
                dstOffset -= size;
                UNSAFE.copyMemory(src, srcOffset, dst, dstOffset, size);
                length -= size;
            }

        }
    }
}

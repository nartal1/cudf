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

import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * Abstract class depicting a Column Vector. This class represents the immutable vector created by the Builders from
 * each respective ColumnVector subclasses.  This class holds references to off heap memory and is
 * reference counted to know when to release it.  Call close to decrement the reference count when
 * you are done with the column, and call inRefCount to increment the reference count.
 */
public abstract class ColumnVector implements AutoCloseable {
    private static Logger log = LoggerFactory.getLogger(ColumnVector.class);
    static boolean REF_COUNT_DEBUG = Boolean.getBoolean("ai.rapids.refcount.debug");

    protected long rows;
    protected final DType type;
    protected long nullCount;
    protected final OffHeapState offHeap = new OffHeapState();
    private int refCount;

    private static <T> String stringJoin(String delim, Iterable<T> it) {
        return String.join(delim,
                StreamSupport.stream(it.spliterator(), false)
                        .map((i) -> i.toString())
                        .collect(Collectors.toList()));
    }

    protected ColumnVector(HostMemoryBuffer hostDataBuffer,
                           HostMemoryBuffer hostValidityBuffer, long rows, DType type, long nullCount) {
        if (nullCount > 0 && hostValidityBuffer == null) {
            throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
        }
        ColumnVectorCleaner.register(this, offHeap);
        offHeap.hostData = new BufferEncapsulator(hostDataBuffer, hostValidityBuffer);
        offHeap.deviceData = null;
        this.rows = rows;
        this.nullCount = nullCount;
        this.type = type;
        refCount = 0;
        incRefCount();
    }

    protected ColumnVector(DeviceMemoryBuffer dataBuffer,
                           DeviceMemoryBuffer validityBuffer, long rows, DType type) {
        ColumnVectorCleaner.register(this, offHeap);
        offHeap.deviceData = new BufferEncapsulator(dataBuffer, validityBuffer);
        offHeap.hostData = null;
        this.rows = rows;
        // This should be overwritten, as this constructor is just for output
        this.nullCount = 0;
        this.type = type;
        refCount = 0;
        incRefCount();
    }

    /**
     * Increment the reference count for this column.  You need to call close on this
     * to decrement the reference count again.
     */
    public void incRefCount() {
        refCount++;
        offHeap.addRef();
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
        offHeap.delRef();
        if (refCount == 0) {
            offHeap.clean(false);
        } else if (refCount < 0) {
            log.error("Close called too many times on {}", this);
            offHeap.logRefCountDebug("double free " + this);
            throw new IllegalStateException("Close called too many times");
        }
    }

    /**
     * Returns the number of nulls in the data.
     */
    public final long getNullCount(){
        return nullCount;
    }

    private void checkDeviceData() {
        if (offHeap.deviceData == null) {
            throw new IllegalStateException("Vector not on Device");
        }
    }

    private void checkHostData() {
        if (offHeap.hostData == null) {
            throw new IllegalStateException("Vector not on Host");
        }
    }

    /**
     * Returns if the vector has a validity vector allocated or not.
     */
    public final boolean hasValidityVector() {
        boolean ret;
        if (offHeap.hostData != null) {
            ret = (offHeap.hostData.valid != null);
        } else {
            ret = (offHeap.deviceData.valid != null);
        }
        return ret;
    }

    /**
     * Returns if the vector has nulls.
     */
    public final boolean hasNulls() {
        return getNullCount() > 0;
    }

    public final boolean isNull(long index) {
        checkHostData();
        if (hasNulls()) {
            return BitVectorHelper.isNull(offHeap.hostData.valid, index);
        }
        return false;
    }

    /**
     * Copies the HostBuffer data to DeviceBuffer.
     */
    public final void toDeviceBuffer() {
        checkHostData();
        if (offHeap.deviceData == null) {
            DeviceMemoryBuffer deviceDataBuffer = DeviceMemoryBuffer.allocate(offHeap.hostData.data.getLength());
            DeviceMemoryBuffer deviceValidityBuffer = null;
            boolean needsCleanup = true;
            try {
                if (hasNulls()) {
                    deviceValidityBuffer = DeviceMemoryBuffer.allocate(offHeap.hostData.valid.getLength());
                }
                offHeap.deviceData = new BufferEncapsulator(deviceDataBuffer, deviceValidityBuffer);
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
        offHeap.deviceData.data.copyFromHostBuffer(offHeap.hostData.data);
        if (offHeap.deviceData.valid != null) {
            offHeap.deviceData.valid.copyFromHostBuffer(offHeap.hostData.valid);
        }
    }

    /**
     * Copies the DeviceBuffer data to HostBuffer.
     */
    public final void toHostBuffer() {
        checkDeviceData();
        if (offHeap.hostData == null) {
            HostMemoryBuffer hostDataBuffer = HostMemoryBuffer.allocate(offHeap.deviceData.data.getLength());
            HostMemoryBuffer hostValidityBuffer = null;
            boolean needsCleanup = true;
            try {
                if (offHeap.deviceData.valid != null) {
                    hostValidityBuffer = HostMemoryBuffer.allocate(offHeap.deviceData.valid.getLength());
                }
                offHeap.hostData = new BufferEncapsulator(hostDataBuffer, hostValidityBuffer);
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
            offHeap.hostData.data.copyFromDeviceBuffer(offHeap.deviceData.data);
            if (offHeap.hostData.valid != null) {
                offHeap.hostData.valid.copyFromDeviceBuffer(offHeap.deviceData.valid);
            }
        }
    }

    @Override
    public String toString() {
        return "ColumnVector{" +
                "rows=" + rows +
                ", type=" + type +
                ", hostData=" + offHeap.hostData +
                ", deviceData=" + offHeap.deviceData +
                ", nullCount=" + nullCount +
                ", cudfColumn=" + offHeap.cudfColumn +
                '}';
    }

    protected final CudfColumn getCudfColumn() {
        if (offHeap.cudfColumn == null) {
            assert rows <= Integer.MAX_VALUE;
            assert getNullCount() <= Integer.MAX_VALUE;
            offHeap.cudfColumn = new CudfColumn(offHeap.deviceData.data.getAddress(),
                    (offHeap.deviceData.valid == null ? 0 : offHeap.deviceData.valid.getAddress()),
                    (int)rows,
                    type, (int) getNullCount(), CudfTimeUnit.NONE);
        }
        return offHeap.cudfColumn;
    }

    /**
     * Update any internal accounting from what is in the Native Code
     */
    protected void updateFromNative() {
        assert offHeap.cudfColumn != null;
        this.nullCount = offHeap.cudfColumn.getNullCount();
        this.rows = offHeap.cudfColumn.getSize();
    }

    /**
     * Encapsulator class to hold the two buffers as a cohesive object
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

    private static final class RefCountDebugItem {
        final StackTraceElement[] stackTrace;
        final long timeMs;
        final String op;

        public RefCountDebugItem(String op) {
            this.stackTrace = Thread.currentThread().getStackTrace();
            this.timeMs = System.currentTimeMillis();
            this.op = op;
        }

        public String toString() {
            Date date = new Date(timeMs);
            // Simple Date Format is horribly expensive only do this when debug is turned on!
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSSS z");
            return dateFormat.format(date) + ": " + op + "\n"
                    + stringJoin("\n", Arrays.asList(stackTrace))
                    + "\n";
        }
    }

    protected static class OffHeapState implements ColumnVectorCleaner.Cleaner {
        public BufferEncapsulator<HostMemoryBuffer> hostData;
        public BufferEncapsulator<DeviceMemoryBuffer> deviceData;
        public CudfColumn cudfColumn;
        private final List<RefCountDebugItem> refCountDebug;

        public OffHeapState() {
            if (REF_COUNT_DEBUG) {
                refCountDebug = new LinkedList<>();
            } else {
                refCountDebug = null;
            }
        }

        public final void addRef() {
            if (REF_COUNT_DEBUG) {
                refCountDebug.add(new RefCountDebugItem("INC"));
            }
        }

        public final void delRef() {
            if (REF_COUNT_DEBUG) {
                refCountDebug.add(new RefCountDebugItem("DEC"));
            }
        }

        public final void logRefCountDebug(String message) {
            if (REF_COUNT_DEBUG) {
                log.error("{}: {}", message, stringJoin("\n", refCountDebug));
            }
        }

        @Override
        public boolean clean(boolean logErrorIfNotClean) {
            boolean neededCleanup = false;
            if (hostData != null) {
                hostData.close();
                hostData = null;
                neededCleanup = true;
            }
            if (deviceData != null) {
                deviceData.close();
                deviceData = null;
                neededCleanup = true;
            }
            if (cudfColumn != null) {
                cudfColumn.close();
                cudfColumn = null;
                neededCleanup = true;
            }
            if (neededCleanup && logErrorIfNotClean) {
                log.error("YOU LEAKED A COLUMN VECTOR!!!!");
                logRefCountDebug("Leaked vector");
            }
            return neededCleanup;
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
            assert columnVector.offHeap.hostData != null;

            data.copyRange(currentIndex * type.sizeInBytes, columnVector.offHeap.hostData.data,
                    0L,
                    columnVector.getRows() * type.sizeInBytes);

            if (columnVector.nullCount != 0) {
                if (valid == null) {
                    allocateBitmaskAndSetDefaultValues();
                }
                //copy values from intColumnVector to this
                BitVectorHelper.append(columnVector.offHeap.hostData.valid, valid, currentIndex, columnVector.rows);
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

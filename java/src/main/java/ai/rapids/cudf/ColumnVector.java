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
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * A Column Vector. This class represents the immutable vector of data.  This class holds
 * references to off heap memory and is reference counted to know when to release it.  Call
 * close to decrement the reference count when you are done with the column, and call inRefCount
 * to increment the reference count.
 */
public final class ColumnVector implements AutoCloseable {
    static {
        NativeDepsLoader.loadNativeDeps();
    }
    private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);
    static final boolean REF_COUNT_DEBUG = Boolean.getBoolean("ai.rapids.refcount.debug");

    private final DType type;
    private final OffHeapState offHeap = new OffHeapState();
    // Time Unit of a TIMESTAMP vector
    private TimeUnit tsTimeUnit;
    private long rows;
    private long nullCount;
    private int refCount;

    /**
     * Convert elements in it to a String and join them together. Only use for debug messages
     * where the code execution itself can be disabled as this is not fast.
     */
    private static <T> String stringJoin(String delim, Iterable<T> it) {
        return String.join(delim,
                StreamSupport.stream(it.spliterator(), false)
                        .map((i) -> i.toString())
                        .collect(Collectors.toList()));
    }

    ColumnVector(long nativePointer) {
        assert nativePointer != 0;
        ColumnVectorCleaner.register(this, offHeap);
        offHeap.nativeCudfColumnHandle = nativePointer;
        this.type = getDType(nativePointer);
        offHeap.hostData = null;
        this.rows = getRowCount(nativePointer);
        this.nullCount = getNullCount(nativePointer);
        this.tsTimeUnit = getTimeUnit(nativePointer);
        DeviceMemoryBuffer data = new DeviceMemoryBuffer(getDataPtr(nativePointer), this.rows * type.sizeInBytes);
        DeviceMemoryBuffer valid = null;
        long validPtr = getValidPtr(nativePointer);
        if (validPtr != 0) {
            // We are not using the BitVectorHelper.getValidityAllocationSizeInBytes() because cudfColumn was
            // initialized by cudf and not by cudfjni
            valid = new DeviceMemoryBuffer(validPtr, BitVectorHelper.getValidityLengthInBytes(rows));
        }
        this.offHeap.deviceData = new BufferEncapsulator<>(data, valid);
        this.refCount = 0;
        incRefCount();
    }

    ColumnVector(DType type, TimeUnit tsTimeUnit, long rows, long nullCount,
                 HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer) {
        if (nullCount > 0 && hostValidityBuffer == null) {
            throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
        }
        if (type == DType.TIMESTAMP) {
            if (tsTimeUnit == TimeUnit.NONE) {
                this.tsTimeUnit = TimeUnit.MILLISECONDS;
            } else {
                this.tsTimeUnit = tsTimeUnit;
            }
        } else {
            this.tsTimeUnit = TimeUnit.NONE;
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

    ColumnVector(DType type, TimeUnit tsTimeUnit, long rows,
                 DeviceMemoryBuffer dataBuffer, DeviceMemoryBuffer validityBuffer) {
        ColumnVectorCleaner.register(this, offHeap);
        if (type == DType.TIMESTAMP) {
            if (tsTimeUnit == TimeUnit.NONE) {
                this.tsTimeUnit = TimeUnit.MILLISECONDS;
            } else {
                this.tsTimeUnit = tsTimeUnit;
            }
        } else {
            this.tsTimeUnit = TimeUnit.NONE;
        }
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
     * Close this Vector and free memory allocated for HostMemoryBuffer and DeviceMemoryBuffer
     */
    @Override
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

    @Override
    public String toString() {
        return "ColumnVector{" +
                "rows=" + rows +
                ", type=" + type +
                ", hostData=" + offHeap.hostData +
                ", deviceData=" + offHeap.deviceData +
                ", nullCount=" + nullCount +
                ", cudfColumn=" + offHeap.nativeCudfColumnHandle +
                '}';
    }

    /////////////////////////////////////////////////////////////////////////////
    // METADATA ACCESS
    /////////////////////////////////////////////////////////////////////////////

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
    public long getRowCount() {
        return rows;
    }

    /**
     * Returns the type of this vector.
     */
    public DType getType() {
        return type;
    }

    /**
     * Returns the number of nulls in the data.
     */
    public long getNullCount(){
        return nullCount;
    }

    /**
     * Returns if the vector has a validity vector allocated or not.
     */
    public boolean hasValidityVector() {
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
    public boolean hasNulls() {
        return getNullCount() > 0;
    }

    /**
     * For vector types that support a TimeUnit (TIMESTAMP),
     * get the unit of time. Will be NONE for vectors that
     * did not have one set.  For a TIMESTAMP NONE is the default
     * unit which should be the same as MILLISECONDS.
     */
    public TimeUnit getTimeUnit() {
        return tsTimeUnit;
    }

    /////////////////////////////////////////////////////////////////////////////
    // DATA MOVEMENT
    /////////////////////////////////////////////////////////////////////////////

    private void checkHasDeviceData() {
        if (offHeap.deviceData == null) {
            throw new IllegalStateException("Vector not on Device");
        }
    }

    private void checkHasHostData() {
        if (offHeap.hostData == null) {
            throw new IllegalStateException("Vector not on Host");
        }
    }

    /**
     * Be sure the data is on the device.
     */
    public final void ensureOnDevice() {
        if (offHeap.deviceData == null) {
            checkHasHostData();

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
            offHeap.deviceData.data.copyFromHostBuffer(offHeap.hostData.data);
            if (offHeap.deviceData.valid != null) {
                offHeap.deviceData.valid.copyFromHostBuffer(offHeap.hostData.valid);
            }
        }
    }

    /**
     * Be sure the data is on the host.
     */
    public final void ensureOnHost() {
        if (offHeap.hostData == null) {
            checkHasDeviceData();

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

    /////////////////////////////////////////////////////////////////////////////
    // DATA ACCESS
    /////////////////////////////////////////////////////////////////////////////

    /**
     * Check if the value at index is null or not.
     */
    public boolean isNull(long index) {
        assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
        if (hasNulls()) {
            checkHasHostData();
            return BitVectorHelper.isNull(offHeap.hostData.valid, index);
        }
        return false;
    }

    /**
     * For testing only.  Allows null checks to go past the number of rows, but not past the end
     * of the buffer.  NOTE: If the validity vector was allocated by cudf itself it is not
     * guaranteed to have the same padding, but for all practical purposes it does.  This is
     * just to verify that the buffer was allocated and initialized properly.
     */
    boolean isNullExtendedRange(long index) {
        long maxNullRow = BitVectorHelper.getValidityAllocationSizeInBytes(rows) * 8;
        assert (index >= 0 && index < maxNullRow) : "TEST: index is out of range 0 <= " + index + " < " + maxNullRow;
        if (hasNulls()) {
            checkHasHostData();
            return BitVectorHelper.isNull(offHeap.hostData.valid, index);
        }
        return false;
    }

    /**
     * Generic type independent asserts when getting a value from a single index.
     * @param index where to get the data from.
     */
    private void assertsForGet(long index) {
        assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
        assert offHeap.hostData != null : "data is not on the host";
        assert !isNull(index) : " value at " + index + " is null";
    }

    /**
     * Get the value at index.
     */
    public byte getByte(long index) {
        assert type == DType.INT8;
        assertsForGet(index);
        return offHeap.hostData.data.getByte(index * type.sizeInBytes);
    }

    /**
     * Get the value at index.
     */
    public final short getShort(long index) {
        assert type == DType.INT16;
        assertsForGet(index);
        return offHeap.hostData.data.getShort(index * type.sizeInBytes);
    }

    /**
     * Get the value at index.
     */
    public final int getInt(long index) {
        assert type == DType.INT32 || type == DType.DATE32;
        assertsForGet(index);
        return offHeap.hostData.data.getInt(index * type.sizeInBytes);
    }

    /**
     * Get the value at index.
     */
    public final long getLong(long index) {
        assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
        assertsForGet(index);
        return offHeap.hostData.data.getLong(index * type.sizeInBytes);
    }

    /**
     * Get the value at index.
     */
    public final float getFloat(long index) {
        assert type == DType.FLOAT32;
        assertsForGet(index);
        return offHeap.hostData.data.getFloat(index * type.sizeInBytes);
    }

    /**
     * Get the value at index.
     */
    public final double getDouble(long index) {
        assert type == DType.FLOAT64;
        assertsForGet(index);
        return offHeap.hostData.data.getDouble(index * type.sizeInBytes);
    }

    /////////////////////////////////////////////////////////////////////////////
    // DATE/TIME
    /////////////////////////////////////////////////////////////////////////////

    /**
     * Get year from DATE32, DATE64, or TIMESTAMP
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and
     *                  is responsible for its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector year() {
        assert type == DType.DATE32 || type == DType.DATE64 || type == DType.TIMESTAMP;
        return new ColumnVector(Cudf.gdfExtractDatetimeYear(this));
    }

    /**
     * Get month from DATE32, DATE64, or TIMESTAMP
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and
     *                  is responsible for its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector month() {
        assert type == DType.DATE32 || type == DType.DATE64 || type == DType.TIMESTAMP;
        return new ColumnVector(Cudf.gdfExtractDatetimeMonth(this));
    }

    /**
     * Get day from DATE32, DATE64, or TIMESTAMP
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and
     *                  is responsible for its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector day() {
        assert type == DType.DATE32 || type == DType.DATE64 || type == DType.TIMESTAMP;
        return new ColumnVector(Cudf.gdfExtractDatetimeDay(this));
    }

    /**
     * Get hour from DATE64 or TIMESTAMP
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and
     *                  is responsible for its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector hour() {
        assert type == DType.DATE64 || type == DType.TIMESTAMP;
        return new ColumnVector(Cudf.gdfExtractDatetimeHour(this));
    }

    /**
     * Get minute from DATE64 or TIMESTAMP
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and
     *                  is responsible for its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector minute() {
        assert type == DType.DATE64 || type == DType.TIMESTAMP;
        return new ColumnVector(Cudf.gdfExtractDatetimeMinute(this));
    }

    /**
     * Get second from DATE64 or TIMESTAMP
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and
     *                  is responsible for its lifecycle.
     *
     * @return - A new INT16 vector allocated on the GPU.
     */
    public ColumnVector second() {
        assert type == DType.DATE64 || type == DType.TIMESTAMP;
        return new ColumnVector(Cudf.gdfExtractDatetimeSecond(this));
    }


    /////////////////////////////////////////////////////////////////////////////
    // ARITHMETIC
    /////////////////////////////////////////////////////////////////////////////

    /**
     * Add two vectors.
     * Preconditions - vectors have to be the same size and same type. FLOAT32, FLOAT64, INT32 or INT64.
     * NULLs are not currently supported.
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and
     *                  is responsible for its lifecycle.
     * Example:
     *          try (ColumnVector v1 = ColumnVector.fromFloats(1.2f, 5.1f, ...);
     *               ColumnVector v2 = ColumnVector.fromFloats(5.1f, 13.1f, ...);
     *               ColumnVector v3 = v1.add(v2);
     *            ...
     *          }
     *
     * @param v1 - vector to be added to this vector.
     * @return - A new vector allocated on the GPU of the same type as the input types.
     */
    public ColumnVector add(ColumnVector v1) {
        assert type == v1.getType();
        assert type == DType.FLOAT32 || type == DType.FLOAT64 || type == DType.INT32 || type == DType.INT64;
        assert v1.getRowCount() == getRowCount(); // cudf will check this too.
        assert v1.getNullCount() == 0; // cudf add does not currently update nulls at all
        assert getNullCount() == 0;

        return new ColumnVector(Cudf.gdfAddGeneric(this, v1));
    }

    /////////////////////////////////////////////////////////////////////////////
    // INTERNAL/NATIVE ACCESS
    /////////////////////////////////////////////////////////////////////////////

    /**
     * USE WITH CAUTION: This method exposes the address of the native cudf_column.  This allows
     * writing custom kernels or other cuda operations on the data.  DO NOT close this column
     * vector until you are completely done using the native column.  DO NOT modify the column in
     * any way.  This should be treated as a read only data structure. This API is unstable as
     * the underlying C/C++ API is still not stabilized.  If the underlying data structure
     * is renamed this API may be replaced.  The underlying data structure can change from release
     * to release (it is not stable yet) so be sure that your native code is complied against the
     * exact same version of libcudf as this is released for.
     */
    public final long getNativeCudfColumnAddress() {
        if (offHeap.nativeCudfColumnHandle == 0) {
            assert rows <= Integer.MAX_VALUE;
            assert getNullCount() <= Integer.MAX_VALUE;
            checkHasDeviceData();
            offHeap.nativeCudfColumnHandle = allocateCudfColumn();
            cudfColumnViewAugmented(offHeap.nativeCudfColumnHandle,
                    offHeap.deviceData.data.getAddress(),
                    offHeap.deviceData.valid == null ? 0 : offHeap.deviceData.valid.getAddress(),
                    (int)rows, type.nativeId,
                    (int)getNullCount(), tsTimeUnit.getNativeId());
        }
        return offHeap.nativeCudfColumnHandle;
    }

    private static native long allocateCudfColumn() throws CudfException;

    /**
     * Set a CuDF column given data and validity bitmask pointers, size, and datatype, and
     * count of null (non-valid) elements
     *
     * @param cudfColumnHandle     native handle of gdf_column.
     * @param dataPtr    Pointer to data.
     * @param valid      Pointer to validity bitmask for the data.
     * @param size       Number of rows in the column.
     * @param dtype      Data type of the column.
     * @param null_count The number of non-valid elements in the validity bitmask.
     * @param timeUnit   {@link TimeUnit}
     */
    private static native void cudfColumnViewAugmented(long cudfColumnHandle, long dataPtr, long valid,
                                               int size, int dtype, int null_count,
                                               int timeUnit) throws CudfException;


    private static native void freeCudfColumn(long cudfColumnHandle) throws CudfException;

    private static native long getDataPtr(long cudfColumnHandle) throws CudfException;

    private static native long getValidPtr(long cudfColumnHandle) throws CudfException;

    private static native int getRowCount(long cudfColumnHandle) throws CudfException;

    private static DType getDType(long cudfColumnHandle) throws CudfException {
        return DType.fromNative(getDTypeInternal(cudfColumnHandle));
    }

    private static native int getDTypeInternal(long cudfColumnHandle) throws CudfException;

    private static TimeUnit getTimeUnit(long cudfColumnHandle) throws CudfException {
        return TimeUnit.fromNative(getTimeUnitInternal(cudfColumnHandle));
    }

    private static native int getTimeUnitInternal(long cudfColumnHandle) throws CudfException;

    private static native int getNullCount(long cudfColumnHandle) throws CudfException;

    /////////////////////////////////////////////////////////////////////////////
    // HELPER CLASSES
    /////////////////////////////////////////////////////////////////////////////

    /**
     * Encapsulator class to hold the two buffers as a cohesive object
     */
    private static final class BufferEncapsulator<T extends MemoryBuffer> implements AutoCloseable {
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
     * When debug is enabled holds information about inc and dec of ref count.
     */
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

    /**
     * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
     */
    protected static final class OffHeapState implements ColumnVectorCleaner.Cleaner {
        public BufferEncapsulator<HostMemoryBuffer> hostData;
        public BufferEncapsulator<DeviceMemoryBuffer> deviceData;
        private long nativeCudfColumnHandle = 0;
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
            if (nativeCudfColumnHandle != 0) {
                freeCudfColumn(nativeCudfColumnHandle);
                nativeCudfColumnHandle = 0;
                neededCleanup = true;
            }
            if (neededCleanup && logErrorIfNotClean) {
                log.error("YOU LEAKED A COLUMN VECTOR!!!!");
                logRefCountDebug("Leaked vector");
            }
            return neededCleanup;
        }
    }


    /////////////////////////////////////////////////////////////////////////////
    // BUILDER
    /////////////////////////////////////////////////////////////////////////////

    /**
     * Create a new Builder to hold the specified number of rows.  Be sure to close the builder when
     * done with it. Please try to use {@see #build(int, Consumer)} instead to avoid needing to
     * close the builder.
     * @param type the type of vector to build.
     * @param rows the number of rows this builder can hold
     * @return the builder to use.
     */
    public static Builder builder(DType type, int rows) {
        return new Builder(type, TimeUnit.NONE, rows);
    }

    /**
     * Create a new Builder to hold the specified number of rows.  Be sure to close the builder when
     * done with it. Please try to use {@see #build(int, Consumer)} instead to avoid needing to
     * close the builder.
     * @param type the type of vector to build.
     * @param rows the number of rows this builder can hold
     * @return the builder to use.
     */
    public static Builder builder(DType type, TimeUnit tsTimeUnit, int rows) {
        return new Builder(type, tsTimeUnit, rows);
    }

    /**
     * Create a new vector.
     * @param rows maximum number of rows that the vector can hold.
     * @param init what will initialize the vector.
     * @return the created vector.
     */
    public static ColumnVector build(DType type, int rows, Consumer<Builder> init) {
        return build(type, TimeUnit.NONE, rows, init);
    }

    /**
     * Create a new vector.
     * @param rows maximum number of rows that the vector can hold.
     * @param tsTimeUnit the unit of time, really only applicable for TIMESTAMP.
     * @param init what will initialize the vector.
     * @return the created vector.
     */
    public static ColumnVector build(DType type, TimeUnit tsTimeUnit, int rows, Consumer<Builder> init) {
        try (Builder builder = builder(type, tsTimeUnit, rows)) {
            init.accept(builder);
            return builder.build();
        }
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector fromBytes(byte ... values) {
        return build(DType.INT8, values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector fromShorts(short ... values) {
        return build(DType.INT16, values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector fromInts(int ... values) {
        return build(DType.INT32, values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector fromLongs(long ... values) {
        return build(DType.INT64, values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector fromFloats(float ... values) {
        return build(DType.FLOAT32, values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector fromDoubles(double ... values) {
        return build(DType.FLOAT64, values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector datesFromInts(int ... values) {
        return build(DType.DATE32, values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector datesFromLongs(long ... values) {
        return build(DType.DATE64, values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector timestampsFromLongs(long ... values) {
        return build(DType.TIMESTAMP, values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector timestampsFromLongs(TimeUnit tsTimeUnit, long ... values) {
        return build(DType.TIMESTAMP, tsTimeUnit, values.length, (b) -> b.appendArray(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector fromBoxedBytes(Byte ... values) {
        return build(DType.INT8, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector fromBoxedShorts(Short ... values) {
        return build(DType.INT16, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector fromBoxedInts(Integer ... values) {
        return build(DType.INT32, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector fromBoxedLongs(Long ... values) {
        return build(DType.INT64, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector fromBoxedFloats(Float ... values) {
        return build(DType.FLOAT32, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector fromBoxedDoubles(Double ... values) {
        return build(DType.FLOAT64, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector datesFromBoxedInts(Integer ... values) {
        return build(DType.DATE32, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector datesFromBoxedLongs(Long ... values) {
        return build(DType.DATE64, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector timestampsFromBoxedLongs(Long ... values) {
        return build(DType.TIMESTAMP, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector timestampsFromBoxedLongs(TimeUnit tsTimeUnit, Long ... values) {
        return build(DType.TIMESTAMP, tsTimeUnit, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Build
     */
    public static final class Builder implements AutoCloseable {
        private HostMemoryBuffer data;
        private HostMemoryBuffer valid;
        private long currentIndex = 0;
        private long nullCount;
        private final long rows;
        private boolean built;
        private final DType type;
        private final TimeUnit tsTimeUnit;

        /**
         * Create a builder with a buffer of size rows
         * @param type datatype
         * @param tsTimeUnit for TIMESTAMP the unit of time it is storing.
         * @param rows number of rows to allocate.
         */
        Builder(DType type, TimeUnit tsTimeUnit, long rows) {
            this.type = type;
            this.tsTimeUnit = tsTimeUnit;
            this.rows = rows;
            this.data = HostMemoryBuffer.allocate(rows * type.sizeInBytes);
        }

        /**
         * Create a builder with a buffer of size rows (for testing ONLY).
         * @param type datatype
         * @param tsTimeUnit for TIMESTAMP the unit of time it is storing.
         * @param rows number of rows to allocate.
         * @param testData a buffer to hold the data (should be large enough to hold rows entries).
         * @param testValid a buffer to hold the validity vector (should be large enough to hold
         *                 rows entries or is null).
         */
        Builder(DType type, TimeUnit tsTimeUnit, long rows, HostMemoryBuffer testData, HostMemoryBuffer testValid) {
            this.type = type;
            this.tsTimeUnit = tsTimeUnit;
            this.rows = rows;
            this.data = testData;
            this.valid = testValid;
        }

        public final Builder append(byte value) {
            assert type == DType.INT8;
            assert currentIndex < rows;
            data.setByte(currentIndex *  type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder append(byte value, long count) {
            assert (count + currentIndex) <= rows;
            assert type == DType.INT8;
            data.setMemory(currentIndex * type.sizeInBytes, count, value);
            currentIndex += count;
            return this;
        }

        public final Builder append(short value) {
            assert type == DType.INT16;
            assert currentIndex < rows;
            data.setShort(currentIndex *  type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder append(int value) {
            assert (type == DType.INT32 || type == DType.DATE32);
            assert currentIndex < rows;
            data.setInt(currentIndex *  type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder append(long value) {
            assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
            assert currentIndex < rows;
            data.setLong(currentIndex * type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder append(float value) {
            assert type == DType.FLOAT32;
            assert currentIndex < rows;
            data.setFloat(currentIndex * type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder append(double value) {
            assert type == DType.FLOAT64;
            assert currentIndex < rows;
            data.setDouble(currentIndex * type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder appendArray(byte ... values) {
            assert (values.length + currentIndex) <= rows;
            assert type == DType.INT8;
            data.setBytes(currentIndex * type.sizeInBytes, values, 0, values.length);
            currentIndex += values.length;
            return this;
        }

        public final Builder appendArray(short ... values) {
            assert type == DType.INT16;
            assert (values.length + currentIndex) <= rows;
            data.setShorts(currentIndex *  type.sizeInBytes, values, 0, values.length);
            currentIndex += values.length;
            return this;
        }

        public final Builder appendArray(int... values) {
            assert (type == DType.INT32 || type == DType.DATE32);
            assert (values.length + currentIndex) <= rows;
            data.setInts(currentIndex *  type.sizeInBytes, values, 0, values.length);
            currentIndex += values.length;
            return this;
        }

        public final Builder appendArray(long ... values) {
            assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
            assert (values.length + currentIndex) <= rows;
            data.setLongs(currentIndex *  type.sizeInBytes, values, 0, values.length);
            currentIndex += values.length;
            return this;
        }

        public final Builder appendArray(float... values) {
            assert type == DType.FLOAT32;
            assert (values.length + currentIndex) <= rows;
            data.setFloats(currentIndex *  type.sizeInBytes, values, 0, values.length);
            currentIndex += values.length;
            return this;
        }

        public final Builder appendArray(double... values) {
            assert type == DType.FLOAT64;
            assert (values.length + currentIndex) <= rows;
            data.setDoubles(currentIndex *  type.sizeInBytes, values, 0, values.length);
            currentIndex += values.length;
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
                    appendNull();
                } else {
                    append(b);
                }
            }
            return this;
        }

        /**
         * Append multiple values.  This is very slow and should really only be used for tests.
         * @param values the values to append, including nulls.
         * @return  this for chaining.
         * @throws  {@link IndexOutOfBoundsException}
         */
        public final Builder appendBoxed(Short ... values) throws IndexOutOfBoundsException {
            for (Short b: values) {
                if (b == null) {
                    appendNull();
                } else {
                    append(b);
                }
            }
            return this;
        }

        /**
         * Append multiple values.  This is very slow and should really only be used for tests.
         * @param values the values to append, including nulls.
         * @return  this for chaining.
         * @throws  {@link IndexOutOfBoundsException}
         */
        public final Builder appendBoxed(Integer ... values) throws IndexOutOfBoundsException {
            for (Integer b: values) {
                if (b == null) {
                    appendNull();
                } else {
                    append(b);
                }
            }
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
                    appendNull();
                } else {
                    append(b);
                }
            }
            return this;
        }

        /**
         * Append multiple values.  This is very slow and should really only be used for tests.
         * @param values the values to append, including nulls.
         * @return  this for chaining.
         * @throws  {@link IndexOutOfBoundsException}
         */
        public final Builder appendBoxed(Float ... values) throws IndexOutOfBoundsException {
            for (Float b: values) {
                if (b == null) {
                    appendNull();
                } else {
                    append(b);
                }
            }
            return this;
        }

        /**
         * Append multiple values.  This is very slow and should really only be used for tests.
         * @param values the values to append, including nulls.
         * @return  this for chaining.
         * @throws  {@link IndexOutOfBoundsException}
         */
        public final Builder appendBoxed(Double ... values) throws IndexOutOfBoundsException {
            for (Double b: values) {
                if (b == null) {
                    appendNull();
                } else {
                    append(b);
                }
            }
            return this;
        }

        /**
         * Append this vector to the end of this vector
         * @param columnVector - Vector to be added
         * @return  - The ColumnVector based on this builder values
         */
        public final Builder append(ColumnVector columnVector) {
            assert columnVector.rows <= (rows - currentIndex);
            assert columnVector.type == type;
            assert columnVector.offHeap.hostData != null;

            data.copyRange(currentIndex * type.sizeInBytes, columnVector.offHeap.hostData.data,
                    0L,
                    columnVector.getRowCount() * type.sizeInBytes);

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

        private void allocateBitmaskAndSetDefaultValues() {
            long bitmaskSize = BitVectorHelper.getValidityAllocationSizeInBytes(rows);
            valid = HostMemoryBuffer.allocate(bitmaskSize);
            valid.setMemory(0, bitmaskSize, (byte) 0xFF);
        }

        /**
         * Append null value.
         */
        public final Builder appendNull() {
            setNullAt(currentIndex);
            currentIndex++;
            return this;
        }

        /**
         * Set a specific index to null.
         * @param index
         */
        public final Builder setNullAt(long index) {
            assert index < rows;

            // add null
            if (this.valid == null) {
                allocateBitmaskAndSetDefaultValues();
            }
            nullCount += BitVectorHelper.setNullAt(valid, index);
            return this;
        }

        /**
         * Finish and create the immutable ColumnVector.
         */
        public final ColumnVector build() {
            if (built) {
                throw new IllegalStateException("Cannot reuse a builder.");
            }
            built = true;
            return new ColumnVector(type, tsTimeUnit, currentIndex, nullCount, data, valid);
        }

        /**
         * Close this builder and free memory if the ColumnVector wasn't generated. Verifies that
         * the data was released even in the case of an error.
         */
        @Override
        public final void close() {
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

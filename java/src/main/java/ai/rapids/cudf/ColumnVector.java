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
public class ColumnVector implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);
    static final boolean REF_COUNT_DEBUG = Boolean.getBoolean("ai.rapids.refcount.debug");

    //TODO make these private
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

    protected ColumnVector(CudfColumn cudfColumn) {
        assert cudfColumn != null;
        ColumnVectorCleaner.register(this, offHeap);
        this.type = cudfColumn.getDtype();
        offHeap.hostData = null;
        this.rows = cudfColumn.getSize();
        this.nullCount = cudfColumn.getNullCount();
        DeviceMemoryBuffer data = new DeviceMemoryBuffer(cudfColumn.getDataPtr(), this.rows * type.sizeInBytes);
        DeviceMemoryBuffer valid = null;
        if (cudfColumn.getValidPtr() != 0) {
            // We are not using the BitVectorHelper.getValidityAllocationSizeInBytes() because cudfColumn was
            // initialized by cudf and not by cudfjni
            valid = new DeviceMemoryBuffer(cudfColumn.getValidPtr(), BitVectorHelper.getValidityLengthInBytes(rows));
        }
        this.offHeap.deviceData = new BufferEncapsulator<>(data, valid);
        this.offHeap.cudfColumn = cudfColumn;
        this.refCount = 0;
        incRefCount();
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

    // TODO this should be private
    static ColumnVector newOutputVector(ColumnVector v1, ColumnVector v2, DType outputType) {
        assert v1.rows == v2.rows;
        return newOutputVector(v1.rows, v1.hasValidityVector() || v2.hasValidityVector(), outputType);
    }

    //TODO this should be private
    static ColumnVector newOutputVector(long rows, boolean hasValidity, DType type) {
        ColumnVector columnVector = null;
        switch (type) {
            case INT32:
                columnVector = IntColumnVector.newOutputVector(rows, hasValidity);
                break;
            case INT64:
                columnVector = LongColumnVector.newOutputVector(rows, hasValidity);
                break;
            case FLOAT32:
                columnVector = FloatColumnVector.newOutputVector(rows, hasValidity);
                break;
            case INT16:
                columnVector = ShortColumnVector.newOutputVector(rows, hasValidity);
                break;
            case INT8:
            case DATE32:
            case DATE64:
            case FLOAT64:
                DeviceMemoryBuffer data = DeviceMemoryBuffer.allocate(rows * type.sizeInBytes);
                DeviceMemoryBuffer valid = null;
                if (hasValidity) {
                    valid = DeviceMemoryBuffer.allocate(BitVectorHelper.getValidityAllocationSizeInBytes(rows));
                }
                columnVector = new ColumnVector(data, valid, rows, type);
                break;
            case TIMESTAMP:
                columnVector = TimestampColumnVector.newOutputVector(rows, hasValidity);
                break;
            case INVALID:
            default:
                throw new IllegalArgumentException("Invalid type: " + type);
        }
        return columnVector;
    }


    static ColumnVector fromCudfColumn(CudfColumn cudfColumn) {
        ColumnVector columnVector = null;
        switch (cudfColumn.getDtype()) {
            case INT32:
                columnVector = new IntColumnVector(cudfColumn);
                break;
            case INT64:
                columnVector = new LongColumnVector(cudfColumn);
                break;
            case FLOAT32:
                columnVector = new FloatColumnVector(cudfColumn);
                break;
            case INT16:
                columnVector = new ShortColumnVector(cudfColumn);
                break;
            case INT8:
            case DATE32:
            case DATE64:
            case FLOAT64:
                columnVector = new ColumnVector(cudfColumn);
                break;
            case TIMESTAMP:
                columnVector = new TimestampColumnVector(cudfColumn);
                break;
            case INVALID:
            default:
                throw new IllegalArgumentException("Invalid type: " + cudfColumn.getDtype());
        }
        return columnVector;
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
     * Be sure the data is on the device.
     */
    public final void ensureOnDevice() {
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
     * Be sure the data is on the host.
     */
    public final void ensureOnHost() {
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

    /**
     * Get the byte value at index.
     */
    public final byte getByte(long index) {
        assert type == DType.INT8;
        assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
        assert offHeap.hostData != null : "data is not on the host";
        assert !isNull(index) : " value at " + index + " is null";
        return offHeap.hostData.data.getByte(index * type.sizeInBytes);
    }

    /**
     * Get the value at index.
     */
    public final int getInt(long index) {
        assert type == DType.INT32 || type == DType.DATE32;
        assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
        assert offHeap.hostData != null : "data is not on the host";
        assert !isNull(index) : " value at " + index + " is null";
        return offHeap.hostData.data.getInt(index * type.sizeInBytes);
    }

    /**
     * Get the value at index.
     */
    public final long getLong(long index) {
        assert type == DType.INT64 || type == DType.DATE64;
        assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
        assert offHeap.hostData != null : "data is not on the host";
        assert !isNull(index) : " value at " + index + " is null";
        return offHeap.hostData.data.getLong(index * type.sizeInBytes);
    }

    /**
     * Get the value at index.
     */
    public final double getDouble(long index) {
        assert type == DType.FLOAT64;
        assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
        assert offHeap.hostData != null : "data is not on the host";
        assert !isNull(index) : " value at " + index + " is null";
        return offHeap.hostData.data.getDouble(index * type.sizeInBytes);
    }

    /**
     * Get year from DATE32 or DATE64
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new vector allocated on the GPU.
     */
    public ShortColumnVector year() {
        assert type == DType.DATE32 || type == DType.DATE64;
        ShortColumnVector result = ShortColumnVector.newOutputVector(this);
        Cudf.gdfExtractDatetimeYear(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Get month from DATE32 or DATE64
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new vector allocated on the GPU.
     */
    public ShortColumnVector month() {
        assert type == DType.DATE32 || type == DType.DATE64;
        ShortColumnVector result = ShortColumnVector.newOutputVector(this);
        Cudf.gdfExtractDatetimeMonth(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Get day from DATE32 or DATE64
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new vector allocated on the GPU.
     */
    public ShortColumnVector day() {
        assert type == DType.DATE32 || type == DType.DATE64;
        ShortColumnVector result = ShortColumnVector.newOutputVector(this);
        Cudf.gdfExtractDatetimeDay(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Get hour from DATE64
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new vector allocated on the GPU.
     */
    public ShortColumnVector hour() {
        assert type == DType.DATE64;
        ShortColumnVector result = ShortColumnVector.newOutputVector(this);
        Cudf.gdfExtractDatetimeHour(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Get minute from DATE64
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new vector allocated on the GPU.
     */
    public ShortColumnVector minute() {
        assert type == DType.DATE64;
        ShortColumnVector result = ShortColumnVector.newOutputVector(this);
        Cudf.gdfExtractDatetimeMinute(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Get second from DATE64
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     *
     * @return - A new vector allocated on the GPU.
     */
    public ShortColumnVector second() {
        assert type == DType.DATE64;
        ShortColumnVector result = ShortColumnVector.newOutputVector(this);
        Cudf.gdfExtractDatetimeSecond(getCudfColumn(), result.getCudfColumn());
        result.updateFromNative();
        return result;
    }

    /**
     * Add two vectors.
     * Preconditions - vectors have to be the same size
     *
     * Postconditions - A new vector is allocated with the result. The caller owns the vector and is responsible for
     *                  its lifecycle.
     * Example:
     *          try (ColumnVector v1 = ColumnVector.build(DType.FLOAT64, 5, (b) -> b.appendDouble(1.2).appendDouble(5.1)...);
     *               ColumnVector v2 = ColumnVector.build(DType.FLOAT64, 5, (b) -> b.appendDouble(5.1).appendDouble(13.1)...);
     *               ColumnVector v3 = v1.add(v2);
     *            ...
     *          }
     *
     * @param v1 - vector to be added to this vector.
     * @return - A new vector allocated on the GPU.
     */
    public ColumnVector add(ColumnVector v1) {
        assert type == v1.getType();
        assert type == DType.FLOAT64; // TODO need others...
        assert v1.getRows() == getRows(); // cudf will check this too.
        assert v1.getNullCount() == 0; // cudf add does not currently update nulls at all
        assert getNullCount() == 0;

        ColumnVector result = newOutputVector(v1, this, type);
        switch (type) {
            case FLOAT64:
                Cudf.gdfAddF64(getCudfColumn(), v1.getCudfColumn(), result.getCudfColumn());
                break;
            default:
                throw new IllegalArgumentException(type + " is not yet supported");
        }
        result.updateFromNative();
        return result;
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
        return getCudfColumn().getNativeHandle();
    }

    protected final CudfColumn getCudfColumn() {
        if (offHeap.cudfColumn == null) {
            assert rows <= Integer.MAX_VALUE;
            assert getNullCount() <= Integer.MAX_VALUE;
            checkDeviceData();
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
     * Create a new Builder to hold the specified number of rows.  Be sure to close the builder when
     * done with it. Please try to use {@see #build(int, Consumer)} instead to avoid needing to
     * close the builder.
     * @param type the type of vector to build.
     * @param rows the number of rows this builder can hold
     * @return the builder to use.
     */
    public static Builder builder(DType type, int rows) {
        return new Builder(type, rows);
    }

    /**
     * Create a builder but with some things possibly replaced for testing.
     */
    static Builder builder(DType type, int rows, HostMemoryBuffer testData, HostMemoryBuffer testValid) {
        return new Builder(type, rows, testData, testValid);
    }

    /**
     * Create a new vector.
     * @param rows maximum number of rows that the vector can hold.
     * @param init what will initialize the vector.
     * @return the created vector.
     */
    public static ColumnVector build(DType type, int rows, Consumer<Builder> init) {
        try (Builder builder = builder(type, rows)) {
            init.accept(builder);
            return builder.build();
        }
    }

    /**
     * Create a new byte vector from the given values.
     */
    public static ColumnVector build(byte ... values) {
        return build(DType.INT8, values.length, (b) -> b.appendBytes(values));
    }

    /**
     * Create a new byte vector from the given values.
     */
    public static ColumnVector build(double ... values) {
        return build(DType.FLOAT64, values.length, (b) -> b.appendDoubles(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector buildDate(int ... values) {
        return build(DType.DATE32, values.length, (b) -> b.appendInts(values));
    }

    /**
     * Create a new vector from the given values.
     */
    public static ColumnVector buildDate(long ... values) {
        return build(DType.DATE64, values.length, (b) -> b.appendLongs(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector buildBoxed(Byte ... values) {
        return build(DType.INT8, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector buildBoxed(Double ... values) {
        return build(DType.FLOAT64, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector buildBoxedDate(Integer ... values) {
        return build(DType.DATE32, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Create a new vector from the given values.  This API supports inline nulls,
     * but is much slower than using a regular array and should really only be used
     * for tests.
     */
    public static ColumnVector buildBoxedDate(Long ... values) {
        return build(DType.DATE64, values.length, (b) -> b.appendBoxed(values));
    }

    /**
     * Base class for Builder
     */
    public static final class Builder implements AutoCloseable {
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

        public final Builder appendByte(byte value) {
            assert type == DType.INT8;
            assert currentIndex < rows;
            data.setByte(currentIndex *  type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder appendBytes(byte value, long count) {
            assert (count + currentIndex) <= rows;
            assert type == DType.INT8;
            data.setMemory(currentIndex * type.sizeInBytes, count, value);
            currentIndex += count;
            return this;
        }

        public final Builder appendBytes(byte[] values) {
            assert (values.length + currentIndex) <= rows;
            assert type == DType.INT8;
            data.setBytes(currentIndex * type.sizeInBytes, values, values.length);
            currentIndex += values.length;
            return this;
        }

        public final Builder appendShort(short value) {
            assert type == DType.INT16;
            assert currentIndex < rows;
            data.setShort(currentIndex *  type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder appendShorts(short[] values) {
            assert type == DType.INT16;
            assert (values.length + currentIndex) <= rows;
            data.setShorts(currentIndex *  type.sizeInBytes, values, values.length);
            currentIndex += values.length;
            return this;
        }

        public final Builder appendInt(int value) {
            assert (type == DType.INT32 || type == DType.DATE32);
            assert currentIndex < rows;
            data.setInt(currentIndex *  type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder appendInts(int[] values) {
            assert (type == DType.INT32 || type == DType.DATE32);
            assert (values.length + currentIndex) <= rows;
            data.setInts(currentIndex *  type.sizeInBytes, values, values.length);
            currentIndex += values.length;
            return this;
        }

        public final Builder appendLong(long value) {
            assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
            assert currentIndex < rows;
            data.setLong(currentIndex * type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder appendLongs(long[] values) {
            assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
            assert (values.length + currentIndex) <= rows;
            data.setLongs(currentIndex *  type.sizeInBytes, values, values.length);
            currentIndex += values.length;
            return this;
        }

        public final Builder appendFloat(float value) {
            assert type == DType.FLOAT32;
            assert currentIndex < rows;
            data.setFloat(currentIndex * type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder appendFloats(float[] values) {
            assert type == DType.FLOAT32;
            assert (values.length + currentIndex) <= rows;
            data.setFloats(currentIndex *  type.sizeInBytes, values, values.length);
            currentIndex += values.length;
            return this;
        }

        public final Builder appendDouble(double value) {
            assert type == DType.FLOAT64;
            assert currentIndex < rows;
            data.setDouble(currentIndex * type.sizeInBytes, value);
            currentIndex++;
            return this;
        }

        public final Builder appendDoubles(double[] values) {
            assert type == DType.FLOAT64;
            assert (values.length + currentIndex) <= rows;
            data.setDoubles(currentIndex *  type.sizeInBytes, values, values.length);
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
                    appendByte(b);
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
                    appendInt(b);
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
                    appendLong(b);
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
                    appendDouble(b);
                }
            }
            return this;
        }

        private void allocateBitmaskAndSetDefaultValues() {
            long bitmaskSize = BitVectorHelper.getValidityAllocationSizeInBytes(rows);
            valid = HostMemoryBuffer.allocate(bitmaskSize);
            valid.setMemory(0, bitmaskSize, (byte) 0xFF);
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

        public final ColumnVector build() {
            built = true;
            return new ColumnVector(data, valid, currentIndex, type, nullCount);
        }

        /**
         * Close this builder and free memory if the ColumnVector wasn't generated
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

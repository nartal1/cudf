/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.cudf;

/**
 * Base class for a Column of data
 */
class CudfColumn implements AutoCloseable {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    // Constructor will allocate gdf_column in native. And store the returned gdf_column pointer in
    // nativeHandle.
    private long nativeHandle = 0;
    private Long nativeValidHandle;
    private Long nativeDataHandle;
    private Integer size;
    private DType type;
    private Integer nullCount;
    private TimeUnit timeUnit;

    CudfColumn(long data, long valid,
                     int size, DType dtype) {
        nativeHandle = allocate();
        cudfColumnView(nativeHandle, data, valid, size, dtype.nativeId);
    }

    CudfColumn(DType dtype, TimeUnit timeUnit, int size, int null_count, long dataPtr, long valid) {
        nativeHandle = allocate();
        cudfColumnViewAugmented(nativeHandle, dataPtr, valid, size, dtype.nativeId,
                null_count, timeUnit.getNativeId());
    }

    final long getNativeHandle() {
        return nativeHandle;
    }

    CudfColumn(long nativeHandle) {
        this.nativeHandle = nativeHandle;
    }

    private native long allocate() throws CudfException;

    /**
     * Set a CuDF column given data and validity bitmask pointers, size, and datatype.
     *
     * @param column native handle of gdf_column.
     * @param data   Pointer to data.
     * @param valid  Pointer to validity bitmask for the data.
     * @param size   Number of rows in the column.
     * @param dtype  Data type of the column.
     */
    private native void cudfColumnView(long column, long data, long valid,
                                      int size, int dtype) throws CudfException;

    /**
     * Set a CuDF column given data and validity bitmask pointers, size, and datatype, and
     * count of null (non-valid) elements
     *
     * @param column     native handle of gdf_column.
     * @param dataPtr    Pointer to data.
     * @param valid      Pointer to validity bitmask for the data.
     * @param size       Number of rows in the column.
     * @param dtype      Data type of the column.
     * @param null_count The number of non-valid elements in the validity bitmask.
     * @param timeUnit   {@link TimeUnit}
     */
    private native void cudfColumnViewAugmented(long column, long dataPtr, long valid,
                                               int size, int dtype, int null_count,
                                               int timeUnit) throws CudfException;


    @Override
    public void close() {
        free(nativeHandle);
    }

    private native void free(long handle) throws CudfException;


    /* get */
    public final long getDataPtr() {
        if (nativeDataHandle == null) {
            return nativeDataHandle = getDataPtr(nativeHandle);
        }
        return nativeDataHandle;
    }

    private native long getDataPtr(long handle) throws CudfException;


    public final long getValidPtr() {
        if (nativeValidHandle == null) {
            return nativeValidHandle = getValidPtr(nativeHandle);
        }
        return nativeValidHandle;
    }

    private native long getValidPtr(long handle) throws CudfException;


    public int getSize() {
        if (size == null) {
            size = getSize(nativeHandle);
        }
        return size;
    }

    private native int getSize(long handle) throws CudfException;


    public DType getDtype() {
        if (type == null) {
            type = DType.fromNative(getDtype(nativeHandle));
        }
        return type;
    }

    private native int getDtype(long handle) throws CudfException;



    public TimeUnit getTimeUnit() {
        if (timeUnit == null) {
            timeUnit = TimeUnit.fromNative(getTimeUnit(nativeHandle));
        }
        return timeUnit;
    }

    private native int getTimeUnit(long handle) throws CudfException;

    public int getNullCount() {
        if (nullCount == null) {
            nullCount = getNullCount(nativeHandle);
        }
        return nullCount;
    }

    private native int getNullCount(long handle) throws CudfException;

    @Override
    public String toString() {
        return "CudfColumn{" +
                "nativeHandle=" + nativeHandle +
                ", type=" + type +
                ", data=" + Long.toHexString(getDataPtr()) +
                ", valid=" + Long.toHexString(getValidPtr()) +
                ", nullCount=" + getNullCount() +
                '}';
    }
}

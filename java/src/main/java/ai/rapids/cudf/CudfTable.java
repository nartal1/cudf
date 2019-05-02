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

import java.util.Arrays;

/**
 * This class doesn't take the ownership of the cudfColumns. The caller is responsible for clearing the resources
 * claimed by the {@link CudfColumn} array
 */
class CudfTable implements AutoCloseable {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    // Constructor will allocate table in native. And store the returned table pointer in nativeHandle.
    private final long nativeHandle;
    private final CudfColumn[] cudfColumns;

    CudfTable(CudfColumn[] cudfColumns) {
        long[] cudfColumnPointers = new long[cudfColumns.length];
        for (int i = 0; i < cudfColumns.length; i++) {
            cudfColumnPointers[i] = cudfColumns[i].getNativeHandle();
        }
        this.cudfColumns = cudfColumns;
        nativeHandle = createCudfTable(cudfColumnPointers);
    }

    @Override
    public void close() {
        free(nativeHandle);
    }

    private static native long createCudfTable(long[] cudfColumnPointers) throws CudfException;

    private static native void free(long handle) throws CudfException;

    /**
     * The caller is responsible for clearing the resources claimed by the {@link CudfColumn} array
     */
    void gdfOrderBy(int[] sortKeysIndices, boolean[] isDescending, CudfTable output, boolean areNullsSmallest) {
        long[] sortKeys = new long[sortKeysIndices.length];
        for (int i = 0 ; i < sortKeysIndices.length ; i++) {
            sortKeys[i] = cudfColumns[sortKeysIndices[i]].getNativeHandle();
        }
        gdfOrderBy(nativeHandle, sortKeys, isDescending, output.nativeHandle, areNullsSmallest);
    }

    private static native void gdfOrderBy(long inputTable, long[] sortKeys, boolean[] isDescending, long outputTable,
                                                                        boolean areNullsSmallest) throws CudfException;

    @Override
    public String toString() {
        return "CudfTable{" +
                "nativeHandle=0x" + Long.toHexString(nativeHandle) +
                ", cudfColumns=" + Arrays.toString(cudfColumns) +
                '}';
    }
}

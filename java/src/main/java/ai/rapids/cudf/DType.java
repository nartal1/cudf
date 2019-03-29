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

enum DType {
    CUDF_INVALID(0, 0),
    CUDF_INT8(1, 1),
    CUDF_INT16(2, 2),
    CUDF_INT32(4, 3),
    CUDF_INT64(8, 4),
    CUDF_FLOAT32(4, 5),
    CUDF_FLOAT64(8, 6),
    /**
     * Days since the UNIX epoch
     */
    CUDF_DATE32(4, 7),
    /**
     * ms since the UNIX epoch
     */
    CUDF_DATE64(8, 8),
    /**
     * Exact timestamp encoded with int64 since the UNIX epoch (Default unit ms)
     */
    CUDF_TIMESTAMP(8, 9);
    //CUDF_CATEGORY(??, 10),
    //CUDF_STRING(??, 11);

    public final int sizeInBytes;
    public final int nativeId;

    DType(int sizeInBytes, int nativeId) {
        this.sizeInBytes = sizeInBytes;
        this.nativeId = nativeId;
    }

    static final DType[] D_TYPES = DType.values();
}

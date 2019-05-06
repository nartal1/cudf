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

public enum DType {
    INVALID(0, 0, "invalid"),
    INT8(1, 1, "int"),
    INT16(2, 2, "short"),
    INT32(4, 3, "int32"),
    INT64(8, 4, "int64"),
    FLOAT32(4, 5, "float32"),
    FLOAT64(8, 6, "float64"),
    /**
     * Days since the UNIX epoch
     */
    DATE32(4, 7, "date32"),
    /**
     * ms since the UNIX epoch
     */
    DATE64(8, 8, "date64"),
    /**
     * Exact timestamp encoded with int64 since the UNIX epoch (Default unit ms)
     */
    TIMESTAMP(8, 9, "timestamp");

    final int sizeInBytes;
    final int nativeId;
    final String simpleName;

    DType(int sizeInBytes, int nativeId, String simpleName) {
        this.sizeInBytes = sizeInBytes;
        this.nativeId = nativeId;
        this.simpleName = simpleName;
    }

    static final DType[] D_TYPES = DType.values();
}

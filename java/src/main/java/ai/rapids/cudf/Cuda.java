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

public class Cuda {

    // Defined in driver_types.h in cuda library.
    static final int CPU_DEVICE_ID = -1;

    /**
     * Mapping: cudaMemGetInfo(size_t *free, size_t *total)
     */
    public static native CudaMemInfo memGetInfo() throws CudaException;

    /**
     * Copies count bytes from the memory area pointed to by src to the memory area pointed to by dst.
     * Calling cudaMemcpy() with dst and src pointers that do not
     * match the direction of the copy results in an undefined behavior.
     *
     * @param dst   - Destination memory address
     * @param src   - Source memory address
     * @param count - Size in bytes to copy
     * @param kind  - Type of transfer. {@link CudaMemcpyKind}
     */
    static void memcpy(long dst, long src, long count, CudaMemcpyKind kind) {
        memcpy(dst, src, count, kind.getValue());
    }

    private static native void memcpy(long dst, long src, long count, int kind) throws CudaException;

}

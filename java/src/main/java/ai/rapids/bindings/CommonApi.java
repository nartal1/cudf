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
package ai.rapids.bindings;

import ai.rapids.bindings.cuda.Cuda;
import ai.rapids.bindings.cuda.CudaException;
import ai.rapids.bindings.cuda.CudaMemInfo;
import ai.rapids.bindings.rmm.Rmm;
import ai.rapids.bindings.rmm.RmmAllocationMode;
import ai.rapids.bindings.rmm.RmmException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * This class provides more high level apis for java layer.
 */
public class CommonApi {
    private static Logger log = LoggerFactory.getLogger(CommonApi.class);

    private static volatile Throwable libExcpt = null;

    /**
     * --------------------------------------------------------------------------------*
     * Load libcudfjni.so and initialize the memory in device
     * throws Exception if library is not loaded
     */
    static {
        try {
            System.loadLibrary("cudfjni");
            boolean debug = false;
            String RMM_DEBUG = System.getenv("RMM_DEBUG");
            if (RMM_DEBUG != null) {
                debug = Boolean.valueOf(RMM_DEBUG);
            }
            long initAlloc = 0;
            String RMM_INIT_ALLOC = System.getenv("RMM_INIT_ALLOC_BYTES");
            if (RMM_INIT_ALLOC != null) {
                try {
                    initAlloc = Long.valueOf(RMM_INIT_ALLOC);
                } catch (NumberFormatException e) {
                    log.warn("The format of RMM_INIT_ALLOC_BYTES is not a number " + RMM_INIT_ALLOC);
                }
            }
            rmmInit(initAlloc, debug);
        } catch (Throwable e) {
            libExcpt = e;
            log.error("Error loading cudfjni: " + e.getMessage());
            throw e;
        }
    }

    /**
     * ---------------------------------------------------------------------------------*
     *
     * @param initSize the size in bytes to allocate in device memory
     * @param debug    the value of RMM_DEBUG
     * @throws CudaException
     */
    private static void rmmInit(long initSize, boolean debug) throws CudaException {
        CudaMemInfo memInfo = Cuda.memGetInfo();

        if (initSize <= 0) {
            initSize = memInfo.free - (1024 * 1024 * 10);
        }
        long size = Long.compareUnsigned(initSize, memInfo.total) < 0 ? initSize : (memInfo.total - 1);
        if (Long.compareUnsigned(size, 0) < 0) {
            throw new RmmException("pool error size: " + size);
        }
        Rmm.initialize(RmmAllocationMode.POOL, debug, size);
    }

    /**
     * @return null if library is not loaded
     */
    public static boolean libraryLoaded() {
        return libExcpt == null;
    }
}

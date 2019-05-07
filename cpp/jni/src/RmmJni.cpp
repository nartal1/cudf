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

#include "jni_utils.hpp"

#include <rmm/rmm_api.h>


extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_initialize
        (JNIEnv *env, jclass clazz, jint allocationMode, jboolean enableLogging, jlong poolSize) {
    rmmOptions_t opts;
    opts.allocation_mode = static_cast<rmmAllocationMode_t>(allocationMode);
    opts.enable_logging = enableLogging == JNI_TRUE;
    opts.initial_pool_size = poolSize;
    JNI_RMM_TRY(env, , rmmInitialize(&opts));
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_alloc
        (JNIEnv *env, jclass clazz, jlong size, jlong stream) {
    void *ret = 0;
    cudaStream_t cStream = reinterpret_cast<cudaStream_t>(stream);
    JNI_RMM_TRY(env, 0, RMM_ALLOC(&ret, size, cStream));
    return (jlong) ret;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_free
        (JNIEnv *env, jclass clazz, jlong ptr, jlong stream) {
    void *cptr = reinterpret_cast<void *>(ptr);
    cudaStream_t cStream = reinterpret_cast<cudaStream_t>(stream);
    JNI_RMM_TRY(env, , RMM_FREE(cptr, cStream));
}

}


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

#include "ai_rapids_cudf_Cuda.h"
#include "jni_utils.h"


extern "C" {

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_Cuda_memGetInfo
        (JNIEnv *env, jclass clazz) {
    size_t free, total;
    JNI_CUDA_TRY(env, NULL, cudaMemGetInfo(&free, &total));

    jclass infoClass = env->FindClass("Lai/rapids/cudf/CudaMemInfo;");
    if (infoClass == NULL) {
        JNI_THROW(env, "java/lang/NoClassDefFoundError",
                  "Could not find class of ai.rapids.cudf.CudaMemInfo", NULL);
    }

    jmethodID ctorID = env->GetMethodID(infoClass, "<init>", "(JJ)V");
    if (ctorID == NULL) {
        JNI_THROW(env, "java/lang/NoSuchMethodException",
                  "Could not find constructor of ai.rapids.cudf.CudaMemInfo ",
                  NULL);
    }

    jobject infoObj = env->NewObject(infoClass, ctorID, (jlong) free, (jlong) total);
    // No need to check for exceptions of null return value as we are just handing the object back to the JVM.
    // which will handle throwing any exceptions that happened in the constructor.
    return infoObj;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_memcpy
        (JNIEnv *env, jclass, jlong dst, jlong src, jlong count, jint kind) {
    JNI_NULL_CHECK(env, dst, "dst memory pointer is null",);
    JNI_NULL_CHECK(env, src, "src memory pointer is null",);
    JNI_CUDA_TRY(env, , cudaMemcpy((void *) dst, (const void *) src, count, (cudaMemcpyKind) kind));
}

}

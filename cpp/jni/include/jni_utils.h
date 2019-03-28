#pragma once

#include "rmm/rmm.h"
#include "cudf.h"

#include <jni.h>
#include <string.h>


#define JNI_THROW(env, clazz_name, message, ret_val) \
{\
  jclass exClass = env->FindClass(clazz_name); \
  if (exClass == NULL) { \
     exClass = env->FindClass("java/lang/NoClassDefFoundError"); \
     env->ThrowNew(exClass, clazz_name); \
     return ret_val; \
  } \
  env->ThrowNew(exClass, message); \
  return ret_val; \
}


#define JNI_CUDA_TRY(env, ret_val, call) \
{ \
  cudaError_t internal_cudaStatus = (call); \
  if (cudaSuccess != internal_cudaStatus ) \
  { \
    JNI_THROW(env, "ai/rapids/bindings/cuda/CudaException", cudaGetErrorString(internal_cudaStatus), ret_val); \
  } \
}

#define JNI_RMM_TRY(env, ret_val, call) \
{ \
  rmmError_t internal_rmmStatus = (call); \
  if (RMM_SUCCESS != internal_rmmStatus ) \
  { \
    if (RMM_ERROR_CUDA_ERROR == internal_rmmStatus) { \
        JNI_CUDA_TRY(env, ret_val, cudaPeekAtLastError());\
    } \
    JNI_THROW(env, "ai/rapids/bindings/rmm/RmmException", rmmGetErrorString(internal_rmmStatus), ret_val); \
  } \
}

#define JNI_GDF_TRY(env, ret_val, call) \
{ \
  gdf_error internal_gdfStatus = (call); \
  if (GDF_SUCCESS != internal_gdfStatus ) \
  { \
    if (GDF_CUDA_ERROR == internal_gdfStatus) { \
        JNI_CUDA_TRY(env, ret_val, cudaPeekAtLastError());\
    } \
    JNI_THROW(env, "ai/rapids/bindings/cudf/GdfException", gdf_error_get_name(internal_gdfStatus), ret_val); \
  } \
}

#define JNI_NULL_CHECK(env, obj, error_msg, ret_val) \
{\
  if (obj == 0) { \
        JNI_THROW(env, "java/lang/NullPointerException", error_msg, ret_val); \
  } \
}

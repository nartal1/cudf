#pragma once

#include "rmm/rmm.h"
#include "cudf.h"

#include <jni.h>
#include <string.h>

#define JNI_THROW_NEW(env, clazz_name, message, ret_val) \
{\
  jclass exClass = env->FindClass(clazz_name); \
  if (exClass == NULL) { \
    return ret_val; \
  } \
  env->ThrowNew(exClass, message); \
  return ret_val; \
}

namespace cudf {
  inline jthrowable cudaException(JNIEnv *env, cudaError_t status, jthrowable cause = NULL) {
    jclass exClass = env->FindClass("ai/rapids/cudf/CudaException");
    if (exClass == NULL) {
        return NULL;
    }
    jmethodID ctorID = env->GetMethodID(exClass, "<init>", "(Ljava/lang/String;Ljava/lang/Throwable;)V");
    if (ctorID == NULL) {
        return NULL;
    }

    jstring msg = env->NewStringUTF(cudaGetErrorString(status));
    if (msg == NULL) {
        return NULL;
    }

    jobject ret = env->NewObject(exClass, ctorID, msg, cause);
    return (jthrowable)ret;
  }

  inline jthrowable cudfException(JNIEnv *env, gdf_error status, jthrowable cause = NULL) {
    jclass exClass = env->FindClass("ai/rapids/cudf/CudfException");
    if (exClass == NULL) {
        return NULL;
    }
    jmethodID ctorID = env->GetMethodID(exClass, "<init>", "(Ljava/lang/String;Ljava/lang/Throwable;)V");
    if (ctorID == NULL) {
        return NULL;
    }

    jstring msg = env->NewStringUTF(gdf_error_get_name(status));
    if (msg == NULL) {
        return NULL;
    }

    jobject ret = env->NewObject(exClass, ctorID, msg, cause);
    return (jthrowable)ret;
  }

  inline jthrowable rmmException(JNIEnv *env, rmmError_t status, jthrowable cause = NULL) {
    jclass exClass = env->FindClass("ai/rapids/cudf/RmmException");
    if (exClass == NULL) {
        return NULL;
    }
    jmethodID ctorID = env->GetMethodID(exClass, "<init>", "(Ljava/lang/String;Ljava/lang/Throwable;)V");
    if (ctorID == NULL) {
        return NULL;
    }

    jstring msg = env->NewStringUTF(rmmGetErrorString(status));
    if (msg == NULL) {
        return NULL;
    }

    jobject ret = env->NewObject(exClass, ctorID, msg, cause);
    return (jthrowable)ret;
  }

}

#define JNI_CUDA_TRY(env, ret_val, call) \
{ \
  cudaError_t internal_cudaStatus = (call); \
  if (cudaSuccess != internal_cudaStatus ) \
  { \
    /* Clear the last error so it does not propagate.*/ \
    cudaGetLastError(); \
    jthrowable jt = cudf::cudaException(env, internal_cudaStatus); \
    if (jt != NULL) { \
      env->Throw(jt); \
    } \
    return ret_val; \
  } \
}

#define JNI_RMM_TRY(env, ret_val, call) \
{ \
  rmmError_t internal_rmmStatus = (call); \
  if (RMM_SUCCESS != internal_rmmStatus ) \
  { \
    jthrowable cudaE = NULL; \
    if (RMM_ERROR_CUDA_ERROR == internal_rmmStatus) { \
        cudaE = cudf::cudaException(env, cudaGetLastError()); \
    } \
    jthrowable jt = cudf::rmmException(env, internal_rmmStatus, cudaE); \
    if (jt != NULL) { \
      env->Throw(jt); \
    } \
    return ret_val; \
  } \
}

#define JNI_GDF_TRY(env, ret_val, call) \
{ \
  gdf_error internal_gdfStatus = (call); \
  if (GDF_SUCCESS != internal_gdfStatus ) \
  { \
    jthrowable cudaE = NULL; \
    if (GDF_CUDA_ERROR == internal_gdfStatus) { \
        cudaE = cudf::cudaException(env, cudaGetLastError()); \
    } \
    jthrowable jt = cudf::cudfException(env, internal_gdfStatus, cudaE); \
    if (jt != NULL) { \
      env->Throw(jt); \
    } \
    return ret_val; \
  } \
}

#define JNI_NULL_CHECK(env, obj, error_msg, ret_val) \
{\
  if (obj == 0) { \
        JNI_THROW_NEW(env, "java/lang/NullPointerException", error_msg, ret_val); \
  } \
}

#define JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val) \
{\
    if (env -> ExceptionOccurred()) { \
        return ret_val; \
    } \
}

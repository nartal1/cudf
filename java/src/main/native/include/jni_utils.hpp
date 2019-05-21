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
#pragma once

#include "rmm/rmm.h"
#include "cudf.h"
#include "utilities/error_utils.hpp"

#include <jni.h>
#include <string>
#include <utility>

namespace cudf {
namespace jni {

  /**
   * @brief indicates that a JNI error of some kind was thrown and the main function should return.
   */
  class jni_exception : public std::runtime_error {
  public:
      jni_exception(char const* const message) : std::runtime_error(message) {}
      jni_exception(std::string const& message) : std::runtime_error(message) {}
  };

  /**
   * @brief throw a java exception and a C++ one for flow control.
   */
  inline void throwJavaException(JNIEnv * const env, const char * clazz_name, const char * message) {
    jclass exClass = env->FindClass(clazz_name); \
    if (exClass != NULL) {
      env->ThrowNew(exClass, message);
    }
    throw jni_exception(message);
  }


  /**
   * @brief check if an java exceptions have been thrown and if so throw a C++ exception
   * so the flow control stop processing. 
   */
  inline void checkJavaException(JNIEnv * const env) {
    if (env->ExceptionOccurred()) {
      // Not going to try to get the message out of the Exception, too complex and might fail.
      throw jni_exception("JNI Exception...");
    }
  }

  /**
   * @brief RAII for jlongArray to be sure it is handled correctly. 
   *
   * By default any changes to the array will be committed back when
   * the destructor is called unless cancel is called first.
   */
  class native_jlongArray {
  private:
     JNIEnv * const env;
     jlongArray orig;
     int len;
     mutable jlong *data_ptr;

     void init_data_ptr() const {
       if (orig != NULL && data_ptr == NULL) {
         data_ptr = env->GetLongArrayElements(orig, NULL);
         checkJavaException(env);
       }
     }
  public:
     native_jlongArray(native_jlongArray const&) = delete;
     native_jlongArray& operator=(native_jlongArray const&) = delete;

     native_jlongArray(JNIEnv * const env, jlongArray orig) :
         env(env),
         orig(orig),
         len(0),
         data_ptr(NULL) {
       if (orig != NULL) {
         len = env->GetArrayLength(orig);
         checkJavaException(env);
       }
     }

     native_jlongArray(JNIEnv* const env, int len) :
         env(env),
         orig(env->NewLongArray(len)),
         len(len),
         data_ptr(NULL) {
       checkJavaException(env);
     }

     native_jlongArray(JNIEnv * const env, jlong * arr, int len) :
         env(env),
         orig(env->NewLongArray(len)),
         len(len),
         data_ptr(NULL) {
       checkJavaException(env);
       env->SetLongArrayRegion(orig, 0, len, arr);
       checkJavaException(env);
     }

     bool isNull() const noexcept {
       return orig == NULL;
     }

     int size() const noexcept {
       return len;
     }

     jlong operator[](int index) const {
       if (orig == NULL) {
         throwJavaException(env, "java/lang/NullPointerException", "jlongArray pointer is NULL");
       }
       if (index < 0 || index >= len) {
         throwJavaException(env, "java/lang/ArrayIndexOutOfBoundsException", "NOT IN BOUNDS");
       }
       return data()[index];
     }

     jlong& operator[](int index) {
       if (orig == NULL) {
         throwJavaException(env, "java/lang/NullPointerException", "jlongArray pointer is NULL");
       }
       if (index < 0 || index >= len) {
         throwJavaException(env, "java/lang/ArrayIndexOutOfBoundsException", "NOT IN BOUNDS");
       }
       return data()[index];
     }

     const jlong * const data() const {
       init_data_ptr();
       return data_ptr;
     }

     jlong * data() {
       init_data_ptr();
       return data_ptr;
     }

     const jlongArray get_jlongArray() const {
       return orig;
     }

     jlongArray get_jlongArray() {
       return orig;
     }

     /**
      * @brief if data has been written back into this array, don't commit
      * it.
      */
     void cancel() {
       if (data_ptr != NULL && orig != NULL) {
         env->ReleaseLongArrayElements(orig, data_ptr, JNI_ABORT);
         data_ptr = NULL;
       }
     }

     void commit() {
       if (data_ptr != NULL && orig != NULL) {
         env->ReleaseLongArrayElements(orig, data_ptr, 0);
         data_ptr = NULL;
       }
     }

     ~native_jlongArray() {
       commit();
     }
  };

  /**
   * @brief wrapper around native_jlongArray to make it take pointers instead.
   *
   * By default any changes to the array will be committed back when
   * the destructor is called unless cancel is called first.
   */
  template<typename T>
  class native_jpointerArray {
  private:
     native_jlongArray wrapped;

  public:
     native_jpointerArray(native_jpointerArray const&) = delete;
     native_jpointerArray& operator=(native_jpointerArray const&) = delete;

     native_jpointerArray(JNIEnv * const env, jlongArray orig) :
         wrapped(env, orig) {
     }

     native_jpointerArray(JNIEnv* const env, int len) :
         wrapped(env, len) {
     }

     native_jpointerArray(JNIEnv * const env, T * arr, int len) :
         wrapped(env, arr, len) {
     }

     bool isNull() const noexcept {
       return wrapped.isNull();
     }

     int size() const noexcept {
       return wrapped.size();
     }

     T* operator[](int index) const {
       return data()[index];
     }

     T*& operator[](int index) {
       return data()[index];
     }

     T* const * data() const {
       return reinterpret_cast<T**>(wrapped.data());
     }

     T** data() {
       return reinterpret_cast<T**>(wrapped.data());
     }

     const jlongArray get_jlongArray() const {
       return wrapped.get_jlongArray();
     }

     jlongArray get_jlongArray() {
       return wrapped.get_jlongArray();
     }

     /**
      * @brief if data has been written back into this array, don't commit
      * it.
      */
     void cancel() {
       wrapped.cancel();
     }

     void commit() {
       wrapped.commit();
     }
  };

  /**
   * @brief wrapper around native_jlongArray to hold pointers that are deleted
   * if not released, like std::unique_ptr.
   *
   * By default any changes to the array will be committed back when
   * released unless cancel is called first.
   */
  template<typename T, typename D = std::default_delete<T>>
  class unique_jpointerArray {
  private:
     std::unique_ptr<native_jpointerArray<T>> wrapped;
     D del;

  public:
     unique_jpointerArray(unique_jpointerArray const&) = delete;
     unique_jpointerArray& operator=(unique_jpointerArray const&) = delete;

     unique_jpointerArray(JNIEnv * const env, jlongArray orig) :
         wrapped(new native_jpointerArray<T>(env, orig)) {
     }

     unique_jpointerArray(JNIEnv * const env, jlongArray orig, const D & del) :
         wrapped(new native_jpointerArray<T>(env, orig)),
         del(del) {
     }

     unique_jpointerArray(JNIEnv* const env, int len) :
         wrapped(new native_jpointerArray<T>(env, len)) {
     }

     unique_jpointerArray(JNIEnv* const env, int len, const D & del) :
         wrapped(new native_jpointerArray<T>(env, len)),
         del(del) {
     }

     unique_jpointerArray(JNIEnv * const env, T * arr, int len) :
         wrapped(new native_jpointerArray<T>(env, arr, len)) {
     }

     unique_jpointerArray(JNIEnv * const env, T * arr, int len, const D & del) :
         wrapped(new native_jpointerArray<T>(env, arr, len)),
         del(del) {
     }

     bool isNull() const noexcept {
       return wrapped == NULL || wrapped->isNull();
     }

     int size() const noexcept {
       return wrapped == NULL ? 0 : wrapped->size();
     }

     void reset(int index, T* new_ptr = NULL) {
       if (wrapped == NULL) {
         throw std::logic_error("using unique_jpointerArray after release");
       }
       T* old = (*wrapped)[index];
       if (old != new_ptr) {
         (*wrapped)[index] = new_ptr;
         del(old);
       }
     }

     T* get(int index) {
       if (wrapped == NULL) {
         throw std::logic_error("using unique_jpointerArray after release");
       }
       return (*wrapped)[index];
     }

     T* const * get() {
       if (wrapped == NULL) {
         throw std::logic_error("using unique_jpointerArray after release");
       }
       return wrapped->data();
     }

     jlongArray release() {
       if (wrapped == NULL) {
         return NULL;
       }
       wrapped->commit();
       jlongArray ret = wrapped->get_jlongArray();
       wrapped.reset(NULL);
       return ret;
     }

     ~unique_jpointerArray() {
       if (wrapped != NULL) {
         for (int i = 0; i < wrapped->size(); i++) {
           reset(i, NULL);
         }
       }
     }
  };

  /**
   * @brief RAII for jintArray to be sure it is handled correctly. 
   *
   * By default any changes to the array will be committed back when
   * the destructor is called unless cancel is called first.
   */
  class native_jintArray {
  private:
     JNIEnv * const env;
     jintArray orig;
     int len;
     mutable jint *data_ptr;

     void init_data_ptr() const {
       if (orig != NULL && data_ptr == NULL) {
         data_ptr = env->GetIntArrayElements(orig, NULL);
         checkJavaException(env);
       }
     }
  public:
     native_jintArray(native_jintArray const&) = delete;
     native_jintArray& operator=(native_jintArray const&) = delete;

     native_jintArray(JNIEnv * const env, jintArray orig) :
         env(env),
         orig(orig),
         len(0),
         data_ptr(NULL) {
       if (orig != NULL) {
         len = env->GetArrayLength(orig);
         checkJavaException(env);
       }
     }

     bool isNull() const noexcept {
       return orig == NULL;
     }

     int size() const noexcept {
       return len;
     }

     jint operator[](int index) const {
       if (orig == NULL) {
         throwJavaException(env, "java/lang/NullPointerException", "jintArray pointer is NULL");
       }
       if (index < 0 || index >= len) {
         throwJavaException(env, "java/lang/ArrayIndexOutOfBoundsException", "NOT IN BOUNDS");
       }
       return data()[index];
     }

     jint& operator[](int index) {
       if (orig == NULL) {
         throwJavaException(env, "java/lang/NullPointerException", "jintArray pointer is NULL");
       }
       if (index < 0 || index >= len) {
         throwJavaException(env, "java/lang/ArrayIndexOutOfBoundsException", "NOT IN BOUNDS");
       }
       return data()[index];
     }

     const jint * const data() const {
       init_data_ptr();
       return data_ptr;
     }

     jint * data() {
       init_data_ptr();
       return data_ptr;
     }

     /**
      * @brief if data has been written back into this array, don't commit
      * it.
      */
     void cancel() {
       if (data_ptr != NULL && orig != NULL) {
         env->ReleaseIntArrayElements(orig, data_ptr, JNI_ABORT);
         data_ptr = NULL;
       }
     }

     void commit() {
       if (data_ptr != NULL && orig != NULL) {
         env->ReleaseIntArrayElements(orig, data_ptr, 0);
         data_ptr = NULL;
       }
     }

     ~native_jintArray() {
       commit();
     }
  };

  /**
   * @brief RAII for jbooleanArray to be sure it is handled correctly. 
   *
   * By default any changes to the array will be committed back when
   * the destructor is called unless cancel is called first.
   */
  class native_jbooleanArray {
  private:
     JNIEnv * const env;
     jbooleanArray orig;
     int len;
     mutable jboolean *data_ptr;

     void init_data_ptr() const {
       if (orig != NULL && data_ptr == NULL) {
         data_ptr = env->GetBooleanArrayElements(orig, NULL);
         checkJavaException(env);
       }
     }
  public:
     native_jbooleanArray(native_jbooleanArray const&) = delete;
     native_jbooleanArray& operator=(native_jbooleanArray const&) = delete;

     native_jbooleanArray(JNIEnv * const env, jbooleanArray orig) :
         env(env),
         orig(orig),
         len(0),
         data_ptr(NULL) {
       if (orig != NULL) {
         len = env->GetArrayLength(orig);
         checkJavaException(env);
       }
     }

     bool isNull() const noexcept {
       return orig == NULL;
     }

     int size() const noexcept {
       return len;
     }

     jboolean operator[](int index) const {
       if (orig == NULL) {
         throwJavaException(env, "java/lang/NullPointerException", "jbooleanArray pointer is NULL");
       }
       if (index < 0 || index >= len) {
         throwJavaException(env, "java/lang/ArrayIndexOutOfBoundsException", "NOT IN BOUNDS");
       }
       return data()[index];
     }

     jboolean& operator[](int index) {
       if (orig == NULL) {
         throwJavaException(env, "java/lang/NullPointerException", "jbooleanArray pointer is NULL");
       }
       if (index < 0 || index >= len) {
         throwJavaException(env, "java/lang/ArrayIndexOutOfBoundsException", "NOT IN BOUNDS");
       }
       return data()[index];
     }

     const jboolean * const data() const {
       init_data_ptr();
       return data_ptr;
     }

     jboolean * data() {
       init_data_ptr();
       return data_ptr;
     }

     /**
      * @brief if data has been written back into this array, don't commit
      * it.
      */
     void cancel() {
       if (data_ptr != NULL && orig != NULL) {
         env->ReleaseBooleanArrayElements(orig, data_ptr, JNI_ABORT);
         data_ptr = NULL;
       }
     }

     void commit() {
       if (data_ptr != NULL && orig != NULL) {
         env->ReleaseBooleanArrayElements(orig, data_ptr, 0);
         data_ptr = NULL;
       }
     }

     ~native_jbooleanArray() {
       commit();
     }
  };

  /**
   * @brief RAII for jstring to be sure it is handled correctly.
   */
  class native_jstring {
  private:
      JNIEnv * env;
      jstring orig;
      mutable const char * cstr;
      mutable size_t cstr_length;

      void init_cstr() const {
        if (orig != NULL && cstr == NULL) {
          cstr_length = env->GetStringUTFLength(orig);
          cstr = env->GetStringUTFChars(orig, 0);
          checkJavaException(env);
        }
      }

  public:
      native_jstring(native_jstring const&) = delete;
      native_jstring& operator=(native_jstring const&) = delete;

      native_jstring(native_jstring && other) noexcept :
          env(other.env),
          orig(other.orig),
          cstr(other.cstr),
          cstr_length(other.cstr_length) {
        other.cstr = NULL;
      }

      native_jstring(JNIEnv * const env, jstring orig) : 
          env(env),
          orig(orig),
          cstr(NULL),
          cstr_length(0) {
      }

      native_jstring& operator=(native_jstring const && other) {
        if (orig != NULL && cstr != NULL) {
          env->ReleaseStringUTFChars(orig, cstr);
        }
        this->env = other.env;
        this->orig = other.orig;
        this->cstr = other.cstr;
        this->cstr_length = other.cstr_length;
        other.cstr = NULL;
      }

      bool isNull() const noexcept {
        return orig == NULL;
      }

      const char * get() const {
        init_cstr();
        return cstr;
      }

      size_t size_bytes() const {
        init_cstr();
        return cstr_length;
      }

      bool is_empty() const {
        if (cstr != NULL) {
          return cstr_length <= 0;
        } else if (orig != NULL) {
          jsize len = env->GetStringLength(orig);
          checkJavaException(env);
          return len <= 0;
        }
        return true;
      }

      const jstring get_jstring() const {
        return orig;
      }

      ~native_jstring() {
        if (orig != NULL && cstr != NULL) {
          env->ReleaseStringUTFChars(orig, cstr);
        }
      }
  };

  /**
   * @brief jobjectArray wrapper to make accessing it more convenient.
   */
  template<typename T>
  class native_jobjectArray {
  private:
     JNIEnv * const env;
     jobjectArray orig;
     int len;
  public:
     native_jobjectArray(JNIEnv * const env, jobjectArray orig) :
         env(env),
         orig(orig),
         len(0) {
       if (orig != NULL) {
         len = env->GetArrayLength(orig);
         checkJavaException(env);
       }
     }

     bool isNull() const noexcept {
       return orig == NULL;
     }

     int size() const noexcept {
       return len;
     }

     T operator[](int index) const {
       return get(index);
     }

     T get(int index) const {
       if (orig == NULL) {
         throwJavaException(env, "java/lang/NullPointerException", "jobjectArray pointer is NULL");
       }
       T ret = static_cast<T>(env->GetObjectArrayElement(orig, index));
       checkJavaException(env);
       return ret;
     }

     void set(int index, const T& val) {
       if (orig == NULL) {
         throwJavaException(env, "java/lang/NullPointerException", "jobjectArray pointer is NULL");
       }
       env->SetObjectArrayElement(orig, index, val);
       checkJavaException(env);
     }
  };

  /**
   * @brief jobjectArray wrapper to make accessing strings safe through RAII
   * and convenient.
   */
  class native_jstringArray {
  private:
     JNIEnv * const env;
     native_jobjectArray<jstring> arr;
     mutable std::vector<native_jstring> cache;
     mutable std::vector<const char *> c_cache;

     void init_cache() const {
       if (!arr.isNull() && cache.empty()) {
         int size = this->size();
         cache.reserve(size);
         for (int i = 0; i < size; i++) {
           cache.push_back(native_jstring(env, arr.get(i)));
         }
       }
     }

     void init_c_cache() const {
       if (!arr.isNull() && c_cache.empty()) {
         init_cache();
         int size = this->size();
         c_cache.reserve(size);
         for (int i = 0; i < size; i++) {
           c_cache.push_back(cache[i].get());
         }
       }
     }

     void update_caches(int index, jstring val) {
       if (!cache.empty()) {
         cache[index] = native_jstring(env, val);
         if (!c_cache.empty()) {
           c_cache[index] = cache[index].get();
         }
       } else if (!c_cache.empty()) {
         //Illegal state
         throw std::logic_error("CACHING IS MESSED UP");
       }
     }
  public:
     native_jstringArray(JNIEnv * const env, jobjectArray orig) :
         env(env),
         arr(env, orig) {}

     bool isNull() const noexcept {
       return arr.isNull();
     }

     int size() const noexcept {
       return arr.size();
     }

     native_jstring& operator[](int index) const {
       return get(index);
     }

     native_jstring& get(int index) const {
       if (arr.isNull()) {
         throwJavaException(env, "java/lang/NullPointerException", "jstringArray pointer is NULL");
       }
       init_cache();
       return cache[index];
     }

     const char ** const as_c_array() const {
       init_c_cache();
       return c_cache.data();
     }

     void set(int index, jstring val) {
       arr.set(index, val);
       update_caches(index, val);
     }

     void set(int index, const native_jstring& val) {
       arr.set(index, val.get_jstring());
       update_caches(index, val.get_jstring());
     }

     void set(int index, const char * val) {
       jstring str = env->NewStringUTF(val);
       checkJavaException(env);
       arr.set(index, str);
       update_caches(index, str);
     }
  };

  /**
   * Wrapper for gdf_column. This class owns the underlying gdf_column and release() should be called before assuming
   * ownership of the underlying gdf_column.
   */
  class gdf_column_wrapper {
  public:
      gdf_column_wrapper(gdf_size_type size, gdf_dtype dtype, bool hasValidityBuffer) {
        col = new gdf_column();
        gdf_column_view(col, nullptr, nullptr, size, dtype);

        if (size > 0) {
            RMM_TRY(RMM_ALLOC(&col->data, size * gdf_dtype_size(col->dtype), 0));
            if (hasValidityBuffer) {
                RMM_TRY(RMM_ALLOC(&col->valid, gdf_valid_allocation_size(size), 0));
            }
        }
    }

    ~gdf_column_wrapper() {
        if (col) {
            RMM_FREE(col->data, 0);
            RMM_FREE(col->valid, 0);
            delete col;
        }
    }

    gdf_column_wrapper(gdf_column_wrapper const &) = delete;

    gdf_column_wrapper &operator=(gdf_column_wrapper const &) = delete;

    gdf_column_wrapper(gdf_column_wrapper &&other) : col(other.col) {
            other.col = nullptr;
    }

    gdf_column *operator->() const noexcept { return col; }

    gdf_column *get() const noexcept { return col; }

    gdf_column *release() noexcept {
        auto temp = col;
        col = nullptr;
        return temp;
    }

    private:
        gdf_column *col = nullptr;
  };

  /**
   * @brief create a cuda exception from a given cudaError_t
   */
  inline jthrowable cudaException(JNIEnv * const env, cudaError_t status, jthrowable cause = NULL) {
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

  /**
   * @brief create a cudf exception from a given gdf_error
   */
  inline jthrowable cudfException(JNIEnv * const env, gdf_error status, jthrowable cause = NULL) {
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

  /**
   * @brief create a rmm exception from a given rmmError_t
   */
  inline jthrowable rmmException(JNIEnv * const env, rmmError_t status, jthrowable cause = NULL) {
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

  /**
   * @brief will properly free something allocated through rmm. If the free fails
   * a java exception will be thrown, but not a C++ exception, so we can try and clean
   * up anything else properly.
   */
  template <typename T>
  struct rmm_deleter {
    private:
      JNIEnv * const env;
      cudaStream_t stream;

    public:
      rmm_deleter(JNIEnv * const env, cudaStream_t stream = 0) noexcept :
          env(env),
          stream(stream) {}

    inline void operator() (T* ptr) {
      rmmError_t rmmStatus = RMM_FREE(ptr, stream);
      if (RMM_SUCCESS != rmmStatus)
      {
        jthrowable cudaE = NULL;
        if (RMM_ERROR_CUDA_ERROR == rmmStatus) {
          cudaE = cudaException(env, cudaGetLastError());
        }
        jthrowable jt = rmmException(env, rmmStatus, cudaE);
        if (jt != NULL) {
          jthrowable orig = env->ExceptionOccurred();
          if (orig != NULL) {
            jclass clz = env->GetObjectClass(jt);
            if (clz != NULL) {
              jmethodID id = env->GetMethodID(clz, "addSuppressed", "(Ljava/lang/Throwable;)V");
              if (id != NULL) {
                env->CallVoidMethod(jt, id, orig);
              }
            }
          }
          env->Throw(jt);
          // Don't throw a C++ exception, we will let java handle it later on.
        }
      }
    }
  };

  template <typename T>
      using jni_rmm_unique_ptr = std::unique_ptr<T, rmm_deleter<T>>;

  /**
   * @brief Allocate memory using RMM in a C++ safe way. Will throw java and C++ exceptions
   * on errors.
   */
  template <typename T>
  inline jni_rmm_unique_ptr<T> jniRmmAlloc(JNIEnv * const env, const size_t size, const cudaStream_t stream = 0) {
    T * ptr;
    rmmError_t rmmStatus = RMM_ALLOC(&ptr, size, stream);
    if (RMM_SUCCESS != rmmStatus ) 
    { 
      jthrowable cudaE = NULL; 
      if (RMM_ERROR_CUDA_ERROR == rmmStatus) {
        cudaE = cudaException(env, cudaGetLastError());
      }
      jthrowable jt = rmmException(env, rmmStatus, cudaE);
      if (jt != NULL) {
        env->Throw(jt);
        throw jni_exception("RMM Error...");
      }
    }
    return jni_rmm_unique_ptr<T>(ptr, rmm_deleter<T>(env, stream));
  }

  inline void jniCudaCheck(JNIEnv * const env, cudaError_t cudaStatus) {
    if (cudaSuccess != cudaStatus ) {
      // Clear the last error so it does not propagate.
      cudaGetLastError();
      jthrowable jt = cudaException(env, cudaStatus);
      if (jt != NULL) {
        env->Throw(jt);
        throw jni_exception("CUDA ERROR");
      }
    }
  }

  inline void jniCudfCheck(JNIEnv * const env, gdf_error gdfStatus) {
    if (GDF_SUCCESS != gdfStatus) {
      jthrowable cudaE = NULL;
      if (GDF_CUDA_ERROR == gdfStatus) {
        cudaE = cudaException(env, cudaGetLastError());
      }
      jthrowable jt = cudfException(env, gdfStatus, cudaE);
      if (jt != NULL) {
        env->Throw(jt);
        throw jni_exception("CUDF ERROR");
      }
    }
  }
} // namespace jni
} // namespace cudf

#define JNI_THROW_NEW(env, clazz_name, message, ret_val) \
{\
  jclass exClass = env->FindClass(clazz_name); \
  if (exClass == NULL) { \
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
    /* Clear the last error so it does not propagate.*/ \
    cudaGetLastError(); \
    jthrowable jt = cudf::jni::cudaException(env, internal_cudaStatus); \
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
        cudaE = cudf::jni::cudaException(env, cudaGetLastError()); \
    } \
    jthrowable jt = cudf::jni::rmmException(env, internal_rmmStatus, cudaE); \
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
        cudaE = cudf::jni::cudaException(env, cudaGetLastError()); \
    } \
    jthrowable jt = cudf::jni::cudfException(env, internal_gdfStatus, cudaE); \
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

#define CATCH_STD(env, ret_val) \
    catch (const std::bad_alloc& e) { \
        JNI_THROW_NEW(env, "java/lang/OutOfMemoryError", "Could not allocate native memory", ret_val); \
    } catch (const cudf::jni::jni_exception& e) { \
        /* indicates that a java exception happened, just return so java can throw it. */ \
        return ret_val; \
    } catch (const cudf::cuda_error& e) { \
        JNI_THROW_NEW(env, "ai/rapids/cudf/CudaException", e.what(), ret_val); \
    } catch (const std::exception& e) { \
        JNI_THROW_NEW(env, "ai/rapids/cudf/CudfException", e.what(), ret_val); \
    }

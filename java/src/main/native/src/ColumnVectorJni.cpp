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

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_allocateCudfColumn
        (JNIEnv *env, jobject object) {
    try {
      return reinterpret_cast<jlong>(calloc(1, sizeof(gdf_column)));
    } CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_freeCudfColumn
        (JNIEnv *env, jobject jObject, jlong handle) {
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    if (column != NULL) {
      free(column->col_name);
    }
    free(column);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_getDataPtr
        (JNIEnv *env, jobject jObject, jlong handle) {
    JNI_NULL_CHECK(env, handle, "native handle is null", 0);
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    return reinterpret_cast<jlong>(column->data);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_getValidPtr
        (JNIEnv *env, jobject jObject, jlong handle) {
    JNI_NULL_CHECK(env, handle, "native handle is null", 0);
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    return reinterpret_cast<jlong>(column->valid);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getRowCount
        (JNIEnv *env, jobject jObject, jlong handle) {
    JNI_NULL_CHECK(env, handle, "native handle is null", 0);
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    return static_cast<jint>(column->size);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getNullCount
        (JNIEnv *env, jobject jObject, jlong handle) {
    JNI_NULL_CHECK(env, handle, "native handle is null", 0);
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    return column->null_count;
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getDTypeInternal
        (JNIEnv *env, jobject jObject, jlong handle) {
    JNI_NULL_CHECK(env, handle, "native handle is null", 0);
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    return column->dtype;
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getTimeUnitInternal
        (JNIEnv *env, jobject jObject, jlong handle) {
    JNI_NULL_CHECK(env, handle, "native handle is null", 0);
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    return column->dtype_info.time_unit;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_cudfColumnViewAugmented
        (JNIEnv *env, jobject, jlong handle, jlong dataPtr, jlong jValid, jint size,
         jint dtype, jint null_count, jint timeUnit) {
    JNI_NULL_CHECK(env, handle, "column is null",);
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    void *data = reinterpret_cast<void *>(dataPtr);
    gdf_valid_type *valid = reinterpret_cast<gdf_valid_type *>(jValid);
    gdf_dtype cDtype = static_cast<gdf_dtype>(dtype);
    gdf_dtype_extra_info info;
    info.time_unit = static_cast<gdf_time_unit>(timeUnit);
    JNI_GDF_TRY(env, ,
                gdf_column_view_augmented(column, data, valid, size, cDtype, null_count, info));
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_concatenate
        (JNIEnv *env, jclass clazz, jlongArray columnHandles) {
    JNI_NULL_CHECK(env, columnHandles, "input columns are null", 0);
    try {
      cudf::jni::native_jpointerArray<gdf_column> columns(env, columnHandles);
      size_t total_size = 0;
      bool need_validity = false;
      for (int i = 0; i < columns.size(); ++i) {
        total_size += columns[i]->size;
        // Should be checking for null_count != 0 but libcudf is checking valid != nullptr
        need_validity |= columns[i]->valid != nullptr;
      }
      if (total_size != static_cast<gdf_size_type>(total_size)) {
        cudf::jni::throwJavaException(env, "java/lang/IllegalArgumentException",
            "resulting column is too large");
      }
      cudf::jni::gdf_column_wrapper outcol(total_size, columns[0]->dtype, need_validity);
      JNI_GDF_TRY(env, 0, gdf_column_concat(outcol.get(), columns.data(), columns.size()));
      return reinterpret_cast<jlong>(outcol.release());
    } CATCH_STD(env, 0);
  }
}

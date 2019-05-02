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

#include "ai_rapids_cudf_CudfTable.h"
#include "jni_utils.h"
#include "types.hpp"
#include "copying.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CudfTable_createCudfTable(JNIEnv *env, jclass classObject,
                                                                                            jlongArray cudfColumns) {
    JNI_NULL_CHECK(env, cudfColumns, "input columns are null", 1);

    int const len = env->GetArrayLength(cudfColumns);
    JNI_EXCEPTION_OCCURRED_CHECK(env, 1);
    gdf_column** cols;
    try {
        cols = new gdf_column *[len];
    } catch (const std::bad_alloc& e) {
        JNI_THROW_NEW(env, "java/lang/OutOfMemoryError", "Could not allocate array of native gdf_column pointers", 1);
    }

    jlong* cudfColumnsData = env->GetLongArrayElements(cudfColumns, 0);
    if (cudfColumnsData == NULL) {
        delete[] cols;
        return 1;
    }

    memcpy(cols, cudfColumnsData, sizeof(gdf_column*) * len);

    env->ReleaseLongArrayElements(cudfColumns, cudfColumnsData, 0);

    try {
        cudf::table* table = new cudf::table(cols, len);
        return (jlong) table;
    } catch (const std::bad_alloc& e) {
        delete[] cols;
        JNI_THROW_NEW(env, "java/lang/OutOfMemoryError", "Could not allocate cudf::table", 1);
    }
    return 0;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_CudfTable_free(JNIEnv *env, jclass classObject, jlong handle) {
    cudf::table* table = (cudf::table*) handle;
    if (table != NULL) {
        delete[] table->begin();
    }
    delete table;
}

/**
 * Copy contents of a jbooleanArray into an array of int8_t pointers
 */
int8_t *copyToDevice(JNIEnv * env, jbooleanArray arr) {
    jsize len = env->GetArrayLength(arr);
    JNI_EXCEPTION_OCCURRED_CHECK(env, NULL);
    size_t byteLen = len * sizeof(int8_t);
    jboolean *tmp = env->GetBooleanArrayElements(arr, 0);
    if (tmp == NULL) {
        return NULL;
    }
    int8_t *host;
    try {
        host = new int8_t[byteLen];
    } catch (const std::bad_alloc& e) {
        env->ReleaseBooleanArrayElements(arr, tmp,0);
        JNI_THROW_NEW(env, "java/lang/OutOfMemoryError", "Could not allocate native pointer", NULL);
    }
    for ( int i = 0; i<len; i++) {
        host[i] = (int8_t)tmp[i];
    }
    env->ReleaseBooleanArrayElements(arr, tmp, 0);

    int8_t *device;
    rmmError_t rmmStatus = RMM_ALLOC(&device, byteLen, 0);
    if (rmmStatus != RMM_SUCCESS) {
        delete[] host;
        JNI_RMM_TRY(env, NULL, rmmStatus);
    }
    cudaError_t cudaStatus = cudaMemcpy(device, host, byteLen, cudaMemcpyHostToDevice);
    delete[] host;
    if (cudaSuccess != cudaStatus) {
        JNI_RMM_TRY(env, NULL, RMM_FREE(device, 0));
        JNI_CUDA_TRY(env, NULL, cudaStatus);
    }
    return device;
}

/**
 * Convert an array of longs into an array of gdf_column pointers.
 */
gdf_column** as_gdf_columns(JNIEnv * env, jlongArray columnPtrs) {
    jsize numColumns = env->GetArrayLength(columnPtrs);
    JNI_EXCEPTION_OCCURRED_CHECK(env, NULL);
    jlong * column_tmp = env->GetLongArrayElements(columnPtrs, 0);
    if (column_tmp == NULL) {
        return NULL;
    }

    gdf_column ** columns;
    try {
        columns = new gdf_column *[numColumns];
    } catch (const std::bad_alloc& e) {
        env->ReleaseLongArrayElements(columnPtrs, column_tmp, 0);
        JNI_THROW_NEW(env, "java/lang/OutOfMemoryError", "Could not allocate native gdf_column pointer", NULL);
    }
    for (int i = 0; i < numColumns; i++) {
        columns[i] = (gdf_column*)column_tmp[i];
    }
    env->ReleaseLongArrayElements(columnPtrs, column_tmp, 0);
    return columns;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_CudfTable_gdfOrderBy(JNIEnv *env, jclass jClassObject,
                            jlong jInputTable, jlongArray jSortKeysGdfcolumns, jbooleanArray jIsDescending,
                            jlong jOutputTable, jboolean jAreNullsSmallest) {
    //input validations & verifications
    JNI_NULL_CHECK(env, jInputTable, "input table is null", );
    JNI_NULL_CHECK(env, jSortKeysGdfcolumns, "input table is null", );
    JNI_NULL_CHECK(env, jIsDescending, "sort order array is null", );
    JNI_NULL_CHECK(env, jOutputTable, "output table is null", );

    jsize numColumns = env->GetArrayLength(jSortKeysGdfcolumns);
    JNI_EXCEPTION_OCCURRED_CHECK(env, );
    jsize numColumnsIsDesc = env->GetArrayLength(jIsDescending);
    JNI_EXCEPTION_OCCURRED_CHECK(env, );
    if ( numColumnsIsDesc != numColumns) {
        JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "columns and isDescending lengths don't match", );
    }

    gdf_column** columns = as_gdf_columns(env, jSortKeysGdfcolumns);
    if (columns == NULL) {
        return;
    }

    int8_t * isDescending = copyToDevice(env, jIsDescending);
    if (isDescending == NULL) {
        delete[] columns;
        return; // Exceptions already thrown
    }

    cudf::table* outputTable = (cudf::table*) jOutputTable;
    cudf::table* inputTable = (cudf::table*) jInputTable;
    bool areNullsSmallest = (bool) jAreNullsSmallest;

    void* col_data = 0;
    JNI_RMM_TRY(env, , RMM_ALLOC(&col_data, columns[0]->size * 4, 0));

    gdf_column intermediateOutput;
    // construct column view
    gdf_error status = gdf_column_view(&intermediateOutput, col_data, nullptr, columns[0]->size, gdf_dtype::GDF_INT32);

    status = gdf_order_by(columns, isDescending, (size_t) numColumns, &intermediateOutput, (int) jAreNullsSmallest);
    gather(inputTable, (int*) intermediateOutput.data, outputTable);
}

};


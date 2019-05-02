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
#include "jni_utils.hpp"
#include "types.hpp"
#include "copying.hpp"
#include <cstring>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CudfTable_createCudfTable(JNIEnv *env,
        jclass classObject,
        jlongArray cudfColumns) {
    JNI_NULL_CHECK(env, cudfColumns, "input columns are null", 0);

    try {
      const cudf::native_jlongArray nCudfColumns(env, cudfColumns);
      int const len = nCudfColumns.size();

      std::unique_ptr<gdf_column *[]> cols(new gdf_column *[len]);
      const jlong* cudfColumnsData = nCudfColumns.data();

      std::memcpy(cols.get(), cudfColumnsData, sizeof(gdf_column*) * len);

      cudf::table* table = new cudf::table(cols.get(), len);
      cols.release();
      return reinterpret_cast<jlong>(table);
    } CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_CudfTable_free(JNIEnv *env, jclass classObject, jlong handle) {
    cudf::table* table = reinterpret_cast<cudf::table*>(handle);
    if (table != NULL) {
      delete[] table->begin();
    }
    delete table;
}

/**
 * Copy contents of a jbooleanArray into an array of int8_t pointers
 */
cudf::jni_rmm_unique_ptr<int8_t> copyToDevice(JNIEnv * env, const cudf::native_jbooleanArray &nArr) {
  jsize len = nArr.size();
  size_t byteLen = len * sizeof(int8_t);
  const jboolean *tmp = nArr.data();

  std::unique_ptr<int8_t[]> host(new int8_t[byteLen]);

  for (int i = 0; i<len; i++) {
    host[i] = static_cast<int8_t>(nArr[i]);
  }

  auto device = cudf::jniRmmAlloc<int8_t>(env, byteLen);
  cudf::jniCudaCheck(env, cudaMemcpy(device.get(), host.get(), byteLen, cudaMemcpyHostToDevice));
  return device;
}

/**
 * Convert an array of longs into an array of gdf_column pointers.
 */
std::vector<gdf_column *> as_gdf_columns(const cudf::native_jlongArray &nColumnPtrs) {
    jsize numColumns = nColumnPtrs.size();

    std::vector<gdf_column *> columns(numColumns);
    for (int i = 0; i < numColumns; i++) {
      columns[i] = reinterpret_cast<gdf_column*>(nColumnPtrs[i]);
    }
    return columns;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_CudfTable_gdfOrderBy(JNIEnv *env,
        jclass jClassObject,
        jlong jInputTable,
        jlongArray jSortKeysGdfcolumns,
        jbooleanArray jIsDescending,
        jlong jOutputTable,
        jboolean jAreNullsSmallest) {

    //input validations & verifications
    JNI_NULL_CHECK(env, jInputTable, "input table is null", );
    JNI_NULL_CHECK(env, jSortKeysGdfcolumns, "input table is null", );
    JNI_NULL_CHECK(env, jIsDescending, "sort order array is null", );
    JNI_NULL_CHECK(env, jOutputTable, "output table is null", );

    try {
        const cudf::native_jlongArray nSortKeysGdfcolumns(env, jSortKeysGdfcolumns);
        jsize numColumns = nSortKeysGdfcolumns.size();
        const cudf::native_jbooleanArray nIsDescending(env, jIsDescending);
        jsize numColumnsIsDesc = nIsDescending.size();

        if ( numColumnsIsDesc != numColumns) {
            JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "columns and isDescending lengths don't match", );
        }

        std::vector<gdf_column*> columns = as_gdf_columns(nSortKeysGdfcolumns);
        auto isDescending = copyToDevice(env, nIsDescending);

        cudf::table* outputTable = reinterpret_cast<cudf::table*>(jOutputTable);
        cudf::table* inputTable = reinterpret_cast<cudf::table*>(jInputTable);
        bool areNullsSmallest = static_cast<bool>(jAreNullsSmallest);

        auto col_data = cudf::jniRmmAlloc<int32_t>(env, columns[0]->size * sizeof(int32_t), 0);

        gdf_column intermediateOutput;
        // construct column view
        cudf::jniCudfCheck(env, gdf_column_view(&intermediateOutput, col_data.get(), nullptr, columns[0]->size, gdf_dtype::GDF_INT32));

        cudf::jniCudfCheck(env, gdf_order_by(columns.data(), isDescending.get(), static_cast<size_t>(numColumns), &intermediateOutput, static_cast<int>(jAreNullsSmallest)));
        
        gather(inputTable, col_data.get(), outputTable);
    } CATCH_STD(env, );
}

};


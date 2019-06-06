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

#include <cstring>

#include "copying.hpp"
#include "jni_utils.hpp"
#include "table.hpp"
#include "types.hpp"

namespace cudf {
namespace jni {

/**
 * Copy contents of a jbooleanArray into an array of int8_t pointers
 */
static jni_rmm_unique_ptr<int8_t> copy_to_device(JNIEnv *env, const native_jbooleanArray &nArr) {
  jsize len = nArr.size();
  size_t byteLen = len * sizeof(int8_t);
  const jboolean *tmp = nArr.data();

  std::unique_ptr<int8_t[]> host(new int8_t[byteLen]);

  for (int i = 0; i < len; i++) {
    host[i] = static_cast<int8_t>(nArr[i]);
  }

  auto device = jniRmmAlloc<int8_t>(env, byteLen);
  jniCudaCheck(env, cudaMemcpy(device.get(), host.get(), byteLen, cudaMemcpyHostToDevice));
  return device;
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_createCudfTable(JNIEnv *env, jclass classObject,
                                                                  jlongArray cudfColumns) {
  JNI_NULL_CHECK(env, cudfColumns, "input columns are null", 0);

  try {
    cudf::jni::native_jpointerArray<gdf_column> nCudfColumns(env, cudfColumns);
    cudf::table *table = new cudf::table(nCudfColumns.data(), nCudfColumns.size());
    return reinterpret_cast<jlong>(table);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_freeCudfTable(JNIEnv *env, jclass classObject,
                                                               jlong handle) {
  cudf::table *table = reinterpret_cast<cudf::table *>(handle);
  delete table;
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfOrderBy(JNIEnv *env, jclass jClassObject,
                                                                  jlong jInputTable,
                                                                  jlongArray jSortKeysGdfcolumns,
                                                                  jbooleanArray jIsDescending,
                                                                  jboolean jAreNullsSmallest) {

  // input validations & verifications
  JNI_NULL_CHECK(env, jInputTable, "input table is null", NULL);
  JNI_NULL_CHECK(env, jSortKeysGdfcolumns, "input table is null", NULL);
  JNI_NULL_CHECK(env, jIsDescending, "sort order array is null", NULL);

  try {
    cudf::jni::native_jpointerArray<gdf_column> nSortKeysGdfcolumns(env, jSortKeysGdfcolumns);
    jsize numColumns = nSortKeysGdfcolumns.size();
    const cudf::jni::native_jbooleanArray nIsDescending(env, jIsDescending);
    jsize numColumnsIsDesc = nIsDescending.size();

    if (numColumnsIsDesc != numColumns) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                    "columns and isDescending lengths don't match", NULL);
    }

    auto isDescending = cudf::jni::copy_to_device(env, nIsDescending);

    cudf::table *inputTable = reinterpret_cast<cudf::table *>(jInputTable);
    cudf::jni::output_table output(env, inputTable);

    bool areNullsSmallest = static_cast<bool>(jAreNullsSmallest);

    auto col_data =
        cudf::jni::jniRmmAlloc<int32_t>(env, nSortKeysGdfcolumns[0]->size * sizeof(int32_t), 0);

    gdf_column intermediateOutput;
    // construct column view
    cudf::jni::jniCudfCheck(env,
                            gdf_column_view(&intermediateOutput, col_data.get(), nullptr,
                                            nSortKeysGdfcolumns[0]->size, gdf_dtype::GDF_INT32));

    gdf_context context{};
    // Most of these are probably ignored, but just to be safe
    context.flag_sorted = false;
    context.flag_method = GDF_SORT;
    context.flag_distinct = 0;
    context.flag_sort_result = 1;
    context.flag_sort_inplace = 0;
    context.flag_groupby_include_nulls = true;
    // There is also a MULTI COLUMN VERSION, that we may want to support in the future.
    context.flag_null_sort_behavior =
        jAreNullsSmallest ? GDF_NULL_AS_SMALLEST : GDF_NULL_AS_LARGEST;

    cudf::jni::jniCudfCheck(env, gdf_order_by(nSortKeysGdfcolumns.data(), isDescending.get(),
                                              static_cast<size_t>(numColumns), &intermediateOutput,
                                              &context));

    cudf::table *cudfTable = output.get_cudf_table();

    gather(inputTable, col_data.get(), cudfTable);

    return output.get_native_handles_and_release();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfReadCSV(
    JNIEnv *env, jclass jClassObject, jobjectArray colNames, jobjectArray dataTypes,
    jobjectArray filterColNames, jstring inputfilepath, jlong buffer, jlong bufferLength,
    jint headerRow, jbyte delim, jbyte quote, jbyte comment, jobjectArray nullValues,
    jobjectArray trueValues, jobjectArray falseValues) {
  JNI_NULL_CHECK(env, nullValues, "nullValues must be supplied, even if it is empty", NULL);

  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an inputfilepath", NULL);
  } else if (bufferLength <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported",
                  NULL);
  }

  try {
    cudf::jni::native_jstringArray nColNames(env, colNames);
    cudf::jni::native_jstringArray nDataTypes(env, dataTypes);

    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    cudf::jni::native_jstringArray nNullValues(env, nullValues);
    cudf::jni::native_jstringArray nTrueValues(env, trueValues);
    cudf::jni::native_jstringArray nFalseValues(env, falseValues);
    cudf::jni::native_jstringArray nFilterColNames(env, filterColNames);

    csv_read_arg read_arg{};

    if (read_buffer) {
      read_arg.filepath_or_buffer = reinterpret_cast<const char *>(buffer);
      read_arg.input_data_form = HOST_BUFFER;
      read_arg.buffer_size = bufferLength;
    } else {
      read_arg.filepath_or_buffer = filename.get();

      read_arg.input_data_form = FILE_PATH;
      // don't use buffer, use file path
      read_arg.buffer_size = 0;
    }

    read_arg.windowslinetermination = false;
    read_arg.lineterminator = '\n';
    // delimiter ideally passed in
    read_arg.delimiter = delim;
    read_arg.delim_whitespace = 0;
    read_arg.skipinitialspace = 0;
    read_arg.nrows = -1;
    read_arg.header = headerRow;

    read_arg.num_names = nColNames.size();
    read_arg.names = nColNames.as_c_array();
    read_arg.num_dtype = nDataTypes.size();
    read_arg.dtype = nDataTypes.as_c_array();

    // leave blank
    // read_arg.index_col

    // only support picking columns by name
    read_arg.use_cols_int = NULL;
    read_arg.use_cols_int_len = 0;
    read_arg.use_cols_char = nFilterColNames.as_c_array();
    read_arg.use_cols_char_len = nFilterColNames.size();

    read_arg.skiprows = 0;
    read_arg.skipfooter = 0;
    read_arg.skip_blank_lines = true;

    read_arg.true_values = nTrueValues.as_c_array();
    read_arg.num_true_values = nTrueValues.size();
    read_arg.false_values = nFalseValues.as_c_array();
    read_arg.num_false_values = nFalseValues.size();

    read_arg.num_na_values = nNullValues.size();
    read_arg.na_values = nNullValues.as_c_array();
    read_arg.keep_default_na = false; ///< Keep the default NA values
    read_arg.na_filter = read_arg.num_na_values > 0;

    read_arg.prefix = NULL;
    read_arg.mangle_dupe_cols = true;
    read_arg.parse_dates = 1;
    // read_arg.infer_datetime_format = true;
    read_arg.dayfirst = 0;
    read_arg.compression = nullptr;
    // read_arg.thousands
    read_arg.decimal = '.';
    read_arg.quotechar = quote;
    // read_arg.quoting = QUOTE_NONNUMERIC;
    read_arg.quoting = QUOTE_MINIMAL;
    read_arg.doublequote = true;
    // read_arg.escapechar =
    read_arg.comment = comment;
    read_arg.encoding = NULL;
    read_arg.byte_range_offset = 0;
    read_arg.byte_range_size = 0;

    gdf_error gdfStatus = read_csv(&read_arg);
    JNI_GDF_TRY(env, NULL, gdfStatus);

    cudf::jni::native_jlongArray nativeHandles(env, reinterpret_cast<jlong *>(read_arg.data),
                                               read_arg.num_cols_out);
    return nativeHandles.get_jlongArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfReadParquet(
    JNIEnv *env, jclass jClassObject, jobjectArray filterColNames, jstring inputfilepath,
    jlong buffer, jlong bufferLength) {
  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an inputfilepath", NULL);
  } else if (bufferLength <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported",
                  NULL);
  }

  try {
    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    cudf::jni::native_jstringArray nFilterColNames(env, filterColNames);

    pq_read_arg read_arg{};

    if (read_buffer) {
      read_arg.source = reinterpret_cast<const char *>(buffer);
      read_arg.source_type = HOST_BUFFER;
      read_arg.buffer_size = bufferLength;
    } else {
      read_arg.source = filename.get();
      read_arg.source_type = FILE_PATH;
      // don't use buffer, use file path
      read_arg.buffer_size = 0;
    }

    read_arg.use_cols = nFilterColNames.as_c_array();
    read_arg.use_cols_len = nFilterColNames.size();

    read_arg.row_group = -1;
    read_arg.skip_rows = 0;
    read_arg.num_rows = -1;
    read_arg.strings_to_categorical = false;

    gdf_error gdfStatus = read_parquet(&read_arg);
    JNI_GDF_TRY(env, NULL, gdfStatus);

    cudf::jni::native_jlongArray nativeHandles(env, reinterpret_cast<jlong *>(read_arg.data),
                                               read_arg.num_cols_out);
    return nativeHandles.get_jlongArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfLeftJoin(JNIEnv *env, jclass clazz,
                                                                   jlong leftTable,
                                                                   jintArray leftColJoinIndices,
                                                                   jlong rightTable,
                                                                   jintArray rightColJoinIndices) {
  JNI_NULL_CHECK(env, leftTable, "leftTable is null", NULL);
  JNI_NULL_CHECK(env, leftColJoinIndices, "leftColJoinIndices is null", NULL);
  JNI_NULL_CHECK(env, rightTable, "rightTable is null", NULL);
  JNI_NULL_CHECK(env, rightColJoinIndices, "rightColJoinIndices is null", NULL);

  try {
    cudf::table *nLefTable = reinterpret_cast<cudf::table *>(leftTable);
    cudf::table *nRightTable = reinterpret_cast<cudf::table *>(rightTable);
    cudf::jni::native_jintArray leftJoinColsArr(env, leftColJoinIndices);
    cudf::jni::native_jintArray rightJoinColsArr(env, rightColJoinIndices);

    gdf_context context{};
    context.flag_sorted = 0;
    context.flag_method = GDF_HASH;
    context.flag_distinct = 0;
    context.flag_sort_result = 1;
    context.flag_sort_inplace = 0;

    int resultNumCols =
        nLefTable->num_columns() + nRightTable->num_columns() - leftJoinColsArr.size();

    // gdf_left_join is allocating the memory for the results so
    // allocate the output column structures here when we get it back fill in
    // the the outPtrs
    cudf::jni::unique_jpointerArray<gdf_column> output_columns(env, resultNumCols);
    for (int i = 0; i < resultNumCols; i++) {
      output_columns.reset(i, new gdf_column());
    }
    JNI_GDF_TRY(env, NULL,
                gdf_left_join(nLefTable->begin(), nLefTable->num_columns(), leftJoinColsArr.data(),
                              nRightTable->begin(), nRightTable->num_columns(),
                              rightJoinColsArr.data(), leftJoinColsArr.size(), resultNumCols,
                              const_cast<gdf_column **>(
                                  output_columns.get()), // API does not respect const values
                              nullptr, nullptr, &context));
    return output_columns.release();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfInnerJoin(JNIEnv *env, jclass clazz,
                                                                    jlong leftTable,
                                                                    jintArray leftColJoinIndices,
                                                                    jlong rightTable,
                                                                    jintArray rightColJoinIndices) {
  JNI_NULL_CHECK(env, leftTable, "leftTable is null", NULL);
  JNI_NULL_CHECK(env, leftColJoinIndices, "leftColJoinIndices is null", NULL);
  JNI_NULL_CHECK(env, rightTable, "rightTable is null", NULL);
  JNI_NULL_CHECK(env, rightColJoinIndices, "rightColJoinIndices is null", NULL);

  try {
    cudf::table *nLefTable = reinterpret_cast<cudf::table *>(leftTable);
    cudf::table *nRightTable = reinterpret_cast<cudf::table *>(rightTable);
    cudf::jni::native_jintArray leftJoinColsArr(env, leftColJoinIndices);
    cudf::jni::native_jintArray rightJoinColsArr(env, rightColJoinIndices);

    gdf_context context{};
    context.flag_sorted = 0;
    context.flag_method = GDF_HASH;
    context.flag_distinct = 0;
    context.flag_sort_result = 1;
    context.flag_sort_inplace = 0;

    int resultNumCols =
        nLefTable->num_columns() + nRightTable->num_columns() - leftJoinColsArr.size();

    // gdf_inner_join is allocating the memory for the results so
    // allocate the output column structures here when we get it back fill in
    // the the outPtrs
    cudf::jni::native_jlongArray outputHandles(env, resultNumCols);
    std::vector<std::unique_ptr<gdf_column>> output_columns(resultNumCols);
    for (int i = 0; i < resultNumCols; i++) {
      output_columns[i].reset(new gdf_column());
      outputHandles[i] = reinterpret_cast<jlong>(output_columns[i].get());
    }
    JNI_GDF_TRY(env, NULL,
                gdf_inner_join(nLefTable->begin(), nLefTable->num_columns(), leftJoinColsArr.data(),
                               nRightTable->begin(), nRightTable->num_columns(),
                               rightJoinColsArr.data(), leftJoinColsArr.size(), resultNumCols,
                               reinterpret_cast<gdf_column **>(outputHandles.data()), nullptr,
                               nullptr, &context));
    for (int i = 0; i < resultNumCols; i++) {
      output_columns[i].release();
    }
    return outputHandles.get_jlongArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_concatenate(JNIEnv *env, jclass clazz,
                                                                   jlongArray tableHandles) {
  JNI_NULL_CHECK(env, tableHandles, "input tables are null", NULL);
  try {
    cudf::jni::native_jpointerArray<cudf::table> tables(env, tableHandles);

    // calculate output table size and whether each column needs a validity vector
    int num_columns = tables[0]->num_columns();
    std::vector<bool> need_validity(num_columns);
    size_t total_size = 0;
    for (int table_idx = 0; table_idx < tables.size(); ++table_idx) {
      total_size += tables[table_idx]->num_rows();
      for (int col_idx = 0; col_idx < num_columns; ++col_idx) {
        gdf_column const *col = tables[table_idx]->get_column(col_idx);
        // Should be checking for null_count != 0 but libcudf is checking valid != nullptr
        if (col->valid != nullptr) {
          need_validity[col_idx] = true;
        }
      }
    }

    // check for overflow
    if (total_size != static_cast<gdf_size_type>(total_size)) {
      cudf::jni::throwJavaException(env, "java/lang/IllegalArgumentException",
                                    "resulting column is too large");
    }

    std::vector<cudf::jni::gdf_column_wrapper> outcols;
    outcols.reserve(num_columns);
    std::vector<gdf_column *> outcol_ptrs(num_columns);
    std::vector<gdf_column *> concat_input_ptrs(tables.size());
    for (int col_idx = 0; col_idx < num_columns; ++col_idx) {
      outcols.emplace_back(total_size, tables[0]->get_column(col_idx)->dtype,
                           need_validity[col_idx]);
      outcol_ptrs[col_idx] = outcols[col_idx].get();
      for (int table_idx = 0; table_idx < tables.size(); ++table_idx) {
        concat_input_ptrs[table_idx] = tables[table_idx]->get_column(col_idx);
      }
      JNI_GDF_TRY(env, NULL,
                  gdf_column_concat(outcol_ptrs[col_idx], concat_input_ptrs.data(), tables.size()));
    }

    cudf::jni::native_jlongArray outcol_handles(env, reinterpret_cast<jlong *>(outcol_ptrs.data()),
                                                num_columns);
    jlongArray result = outcol_handles.get_jlongArray();
    for (int i = 0; i < num_columns; ++i) {
      outcols[i].release();
    }

    return result;
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfPartition(
    JNIEnv *env, jclass clazz, jlong input_table, jintArray columns_to_hash,
    jint cudf_hash_function, jint number_of_partitions, jintArray output_offsets) {

  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, columns_to_hash, "columns_to_hash is null", NULL);
  JNI_NULL_CHECK(env, output_offsets, "output_offsets is null", NULL);
  JNI_ARG_CHECK(env, number_of_partitions > 0, "number_of_partitions is zero", NULL);

  try {
    cudf::table *n_input_table = reinterpret_cast<cudf::table *>(input_table);
    cudf::jni::native_jintArray n_columns_to_hash(env, columns_to_hash);
    gdf_hash_func n_cudf_hash_function = static_cast<gdf_hash_func>(cudf_hash_function);
    int n_number_of_partitions = static_cast<int>(number_of_partitions);
    cudf::jni::native_jintArray nOutputOffsets(env, output_offsets);

    JNI_ARG_CHECK(env, n_columns_to_hash.size() > 0, "columns_to_hash is zero", NULL);

    cudf::jni::output_table output(env, n_input_table);

    std::vector<gdf_column *> cols = output.get_gdf_columns();

    JNI_GDF_TRY(env, NULL,
                gdf_hash_partition(n_input_table->num_columns(), n_input_table->begin(),
                                   n_columns_to_hash.data(), n_columns_to_hash.size(),
                                   n_number_of_partitions, cols.data(), nOutputOffsets.data(),
                                   n_cudf_hash_function));

    return output.get_native_handles_and_release();
  }
  CATCH_STD(env, NULL);
}
} // extern "C"

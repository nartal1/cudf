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

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfAddGeneric
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    JNI_NULL_CHECK(env, lhs, "lhs is null", 0);
    JNI_NULL_CHECK(env, rhs, "rhs is null", 0);
    try {
      gdf_column* leftCol = reinterpret_cast<gdf_column*>(lhs);
      gdf_column* rightCol = reinterpret_cast<gdf_column*>(rhs);
      cudf::jni::gdf_column_wrapper output(leftCol->size, leftCol->dtype, leftCol->null_count != 0 || rightCol->null_count != 0);
      JNI_GDF_TRY(env, 0, gdf_add_generic(leftCol, rightCol, output.get()));
      return reinterpret_cast<jlong>(output.release());
    } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeYear
        (JNIEnv * env, jclass, jlong inputPtr) {
    JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
    try {
        gdf_column* input = reinterpret_cast<gdf_column*>(inputPtr);
        cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
        JNI_GDF_TRY(env, 0, gdf_extract_datetime_year(input, output.get()));
        return reinterpret_cast<jlong>(output.release());
    } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeMonth
        (JNIEnv * env, jclass, jlong inputPtr) {
    JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
    try {
        gdf_column* input = reinterpret_cast<gdf_column*>(inputPtr);
        cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
        JNI_GDF_TRY(env, 0, gdf_extract_datetime_month(input, output.get()));
        return reinterpret_cast<jlong>(output.release());
    } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeDay
        (JNIEnv * env, jclass, jlong inputPtr) {
    JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
    try {
        gdf_column* input = reinterpret_cast<gdf_column*>(inputPtr);
        cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
        JNI_GDF_TRY(env, 0, gdf_extract_datetime_day(input, output.get()));
        return reinterpret_cast<jlong>(output.release());
    } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeHour
        (JNIEnv * env, jclass, jlong inputPtr) {
    JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
    try {
        gdf_column* input = reinterpret_cast<gdf_column*>(inputPtr);
        cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
        JNI_GDF_TRY(env, 0, gdf_extract_datetime_hour(input, output.get()));
        return reinterpret_cast<jlong>(output.release());
    } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeMinute
        (JNIEnv * env, jclass, jlong inputPtr) {
    JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
    try {
        gdf_column* input = reinterpret_cast<gdf_column*>(inputPtr);
        cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
        JNI_GDF_TRY(env, 0, gdf_extract_datetime_minute(input, output.get()));
        return reinterpret_cast<jlong>(output.release());
    } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeSecond
        (JNIEnv * env, jclass, jlong inputPtr) {
    JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
    try {
        gdf_column* input = reinterpret_cast<gdf_column*>(inputPtr);
        cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
        JNI_GDF_TRY(env, 0, gdf_extract_datetime_second(input, output.get()));
        return reinterpret_cast<jlong>(output.release());
    } CATCH_STD(env, 0);
}

}

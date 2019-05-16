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

namespace cudf {
namespace jni {

static void gdf_scalar_init(gdf_scalar * scalar, jlong intValues, jfloat fValue, jdouble dValue, jboolean isValid, int dType) {
    scalar->dtype = static_cast<gdf_dtype>(dType);
    scalar->is_valid = isValid;
    switch(scalar->dtype) {
    case GDF_INT8:
        scalar->data.si08 = static_cast<char>(intValues);
        break;
    case GDF_INT16:
        scalar->data.si16 = static_cast<short>(intValues);
        break;
    case GDF_INT32:
        scalar->data.si32 = static_cast<int>(intValues);
        break;
    case GDF_INT64:
        scalar->data.si64 = static_cast<long>(intValues);
        break;
    case GDF_DATE32:
        scalar->data.dt32 = static_cast<gdf_date32>(intValues);
        break;
    case GDF_DATE64:
        scalar->data.dt64 = static_cast<gdf_date64>(intValues);
        break;
    case GDF_TIMESTAMP:
        scalar->data.tmst = static_cast<gdf_timestamp>(intValues);
        break;
    case GDF_BOOL8:
        scalar->data.b08 = static_cast<char>(intValues);
        break;
    case GDF_FLOAT32:
        scalar->data.fp32 = fValue;
        break;
    case GDF_FLOAT64:
        scalar->data.fp64 = dValue;
        break;
    default:
        throw std::logic_error("Unsupported scalar type");
    }
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfUnaryMath
        (JNIEnv *env, jclass, jlong inputPtr, jint intOp, jint outDtype) {
    JNI_NULL_CHECK(env, inputPtr, "input is null", 0);
    try {
      gdf_column* input = reinterpret_cast<gdf_column*>(inputPtr);
      gdf_dtype out_type = static_cast<gdf_dtype>(outDtype);
      gdf_unary_math_op op = static_cast<gdf_unary_math_op>(intOp);
      cudf::jni::gdf_column_wrapper ret(input->size, out_type, input->null_count > 0);

      JNI_GDF_TRY(env, 0,
                gdf_unary_math(input, ret.get(), op));
      return reinterpret_cast<jlong>(ret.release());
    } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpVV
        (JNIEnv *env, jclass, jlong lhsPtr, jlong rhsPtr, jint intOp, jint outDtype) {
    JNI_NULL_CHECK(env, lhsPtr, "lhs is null", 0);
    JNI_NULL_CHECK(env, rhsPtr, "rhs is null", 0);
    try {
      gdf_column* lhs = reinterpret_cast<gdf_column*>(lhsPtr);
      gdf_column* rhs = reinterpret_cast<gdf_column*>(rhsPtr);
      gdf_dtype out_type = static_cast<gdf_dtype>(outDtype);
      gdf_binary_operator op = static_cast<gdf_binary_operator>(intOp);
      cudf::jni::gdf_column_wrapper ret(lhs->size, out_type, lhs->null_count > 0 || rhs->null_count > 0);

      JNI_GDF_TRY(env, 0,
                gdf_binary_operation_v_v(ret.get(), lhs, rhs, op));
      return reinterpret_cast<jlong>(ret.release());
    } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpSV
        (JNIEnv *env, jclass, 
         jlong lhsIntValues, jfloat lhsFValue, jdouble lhsDValue, jboolean lhsIsValid, int lhsDType, 
         jlong rhsPtr, jint intOp, jint outDtype) {
    JNI_NULL_CHECK(env, rhsPtr, "rhs is null", 0);
    try {
      gdf_scalar lhs{};
      cudf::jni::gdf_scalar_init(&lhs, lhsIntValues, lhsFValue, lhsDValue, lhsIsValid, lhsDType);
      gdf_column* rhs = reinterpret_cast<gdf_column*>(rhsPtr);
      gdf_dtype out_type = static_cast<gdf_dtype>(outDtype);
      gdf_binary_operator op = static_cast<gdf_binary_operator>(intOp);
      cudf::jni::gdf_column_wrapper ret(rhs->size, out_type, !lhs.is_valid || rhs->null_count > 0);

      JNI_GDF_TRY(env, 0,
                gdf_binary_operation_s_v(ret.get(), &lhs, rhs, op));
      return reinterpret_cast<jlong>(ret.release());
    } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpVS
        (JNIEnv *env, jclass, 
         jlong lhsPtr,
         jlong rhsIntValues, jfloat rhsFValue, jdouble rhsDValue, jboolean rhsIsValid, int rhsDType, 
         jint intOp, jint outDtype) {
    JNI_NULL_CHECK(env, lhsPtr, "lhs is null", 0);
    try {
      gdf_column* lhs = reinterpret_cast<gdf_column*>(lhsPtr);
      gdf_scalar rhs{};
      cudf::jni::gdf_scalar_init(&rhs, rhsIntValues, rhsFValue, rhsDValue, rhsIsValid, rhsDType);
      gdf_dtype out_type = static_cast<gdf_dtype>(outDtype);
      gdf_binary_operator op = static_cast<gdf_binary_operator>(intOp);
      cudf::jni::gdf_column_wrapper ret(lhs->size, out_type, !rhs.is_valid || lhs->null_count > 0);

      JNI_GDF_TRY(env, 0,
                gdf_binary_operation_v_s(ret.get(), lhs, &rhs, op));
      return reinterpret_cast<jlong>(ret.release());
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

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cudf_gdfCastTo
        (JNIEnv * env, jclass, jlong inputPtr, jlong outputPtr) {
    JNI_NULL_CHECK(env, inputPtr, "input is null",);
    JNI_NULL_CHECK(env, outputPtr, "output is null",);
    try {
      JNI_GDF_TRY(env, ,
        gdf_cast(reinterpret_cast<gdf_column*>(inputPtr), reinterpret_cast<gdf_column*>(outputPtr)));
    } CATCH_STD(env, );
}
}

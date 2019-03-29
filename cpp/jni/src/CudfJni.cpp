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

#include "ai_rapids_cudf_Cudf.h"
#include "jni_utils.h"

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cudf_gdfAddGeneric
        (JNIEnv *env, jclass, jlong lhs, jlong rhs, jlong output) {
    JNI_NULL_CHECK(env, lhs, "lhs is null",);
    JNI_NULL_CHECK(env, rhs, "rhs is null",);
    JNI_NULL_CHECK(env, output, "output is null",);
    JNI_GDF_TRY(env, ,
                gdf_add_generic((gdf_column *) lhs, (gdf_column *) rhs, (gdf_column *) output));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cudf_gdfAddI32
        (JNIEnv *env, jclass, jlong lhs, jlong rhs, jlong output) {
    JNI_NULL_CHECK(env, lhs, "lhs is null",);
    JNI_NULL_CHECK(env, rhs, "rhs is null",);
    JNI_NULL_CHECK(env, output, "output is null",);
    JNI_GDF_TRY(env, ,
                gdf_add_i32((gdf_column *) lhs, (gdf_column *) rhs, (gdf_column *) output));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cudf_gdfAddI64
        (JNIEnv *env, jclass, jlong lhs, jlong rhs, jlong output) {
    JNI_NULL_CHECK(env, lhs, "lhs is null",);
    JNI_NULL_CHECK(env, rhs, "rhs is null",);
    JNI_NULL_CHECK(env, output, "output is null",);
    JNI_GDF_TRY(env, ,
                gdf_add_i64((gdf_column *) lhs, (gdf_column *) rhs, (gdf_column *) output));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cudf_gdfAddF32
        (JNIEnv *env, jclass, jlong lhs, jlong rhs, jlong output) {
    JNI_NULL_CHECK(env, lhs, "lhs is null",);
    JNI_NULL_CHECK(env, rhs, "rhs is null",);
    JNI_NULL_CHECK(env, output, "output is null",);
    JNI_GDF_TRY(env, ,
                gdf_add_f32((gdf_column *) lhs, (gdf_column *) rhs, (gdf_column *) output));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cudf_gdfAddF64
        (JNIEnv *env, jclass, jlong lhs, jlong rhs, jlong output) {
    JNI_NULL_CHECK(env, lhs, "lhs is null",);
    JNI_NULL_CHECK(env, rhs, "rhs is null",);
    JNI_NULL_CHECK(env, output, "output is null",);
    JNI_GDF_TRY(env, ,
                gdf_add_f64((gdf_column *) lhs, (gdf_column *) rhs, (gdf_column *) output));
}

}

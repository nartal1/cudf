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
package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * This class will load the native dependencies.
 */
public class NativeDepsLoader {
    private static Logger log = LoggerFactory.getLogger(NativeDepsLoader.class);

    private static boolean loaded = false;
    static synchronized void loadNativeDeps() {
        if (!loaded) {
            try {
                System.loadLibrary("cudfjni");
                loaded = true;
            } catch (Throwable t) {
                log.error("Could not load cudf jni library...", t);
            }
        }
    }

    public static boolean libraryLoaded() {
        if (!loaded) {
            loadNativeDeps();
        }
        return loaded;
    }
}

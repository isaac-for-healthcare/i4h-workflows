/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>
#include <iostream>


#define CudaCheck(FUNC)                                                                     \
  {                                                                                         \
    const CUresult result = FUNC;                                                           \
    if (result != CUDA_SUCCESS) {                                                           \
      const char *error_name = "";                                                          \
      cuGetErrorName(result, &error_name);                                                  \
      const char *error_string = "";                                                        \
      cuGetErrorString(result, &error_string);                                              \
      std::stringstream buf;                                                                \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] CUDA driver error " << result << " (" \
          << error_name << "): " << error_string;                                           \
      throw std::runtime_error(buf.str().c_str());                                          \
    }                                                                                       \
  }

/*

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#endif

uint8_t* copyImage(const uint8_t* inData, int width, int height, int inChannels, int outChannels) {
    uint8_t* newData = new uint8_t[width * height * outChannels];
    for (int i = 0; i < width * height; ++i) {
        for (int j = 0; j < inChannels; ++j) {
            newData[i * outChannels + j] = inData[i * inChannels + j];
        }
        for (int j = inChannels; j < outChannels; ++j) {
            newData[i * outChannels + j] = 255;
        }
    }
    return newData;
}

std::string padUpto(size_t num, size_t totalDigits) {
	std::string numStr = std::to_string(num);
	size_t pad = totalDigits - numStr.length();
	pad = pad > 0 ? pad : 0;

	numStr.insert(0, pad, '0');
	return numStr;
}

bool saveImage(const std::string& fileNamePrefix, int index, const std::string& fileNameSuffix, int width, int height, int channels, const uint8_t* data) {
    std::string filename = fileNamePrefix + padUpto(index, 5) + fileNameSuffix;
    uint8_t* dataCopy = copyImage(data, width, height, channels, 4);
    for (int i = 0; i < width * height; i++) {
        dataCopy[i * 4 + 3] = 255;
    }
    int result = stbi_write_png(filename.c_str(), width, height, 4, dataCopy, width * 4);
    delete[] dataCopy;
    return result != 0;
}
*/

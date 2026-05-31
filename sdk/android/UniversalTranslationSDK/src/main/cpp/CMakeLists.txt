# android/UniversalTranslationSDK/src/main/cpp/CMakeLists.txt

cmake_minimum_required(VERSION 3.22.1)
project("universal_encoder")

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_library(log-lib log)
find_library(android-lib android)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../../../encoder_core/include
)

# Add ONNX Runtime
set(ONNXRUNTIME_ROOT_PATH ${CMAKE_SOURCE_DIR}/../jniLibs)
include_directories(${ONNXRUNTIME_ROOT_PATH}/include)

# Link ONNX Runtime libraries for each ABI
link_directories(${ONNXRUNTIME_ROOT_PATH}/${ANDROID_ABI})

# Create JNI wrapper library
add_library(universal_encoder SHARED
    jni_wrapper.cpp
    ${CMAKE_SOURCE_DIR}/../../../encoder_core/src/universal_encoder.cpp
    ${CMAKE_SOURCE_DIR}/../../../encoder_core/src/vocabulary_pack.cpp
)

# Link libraries
target_link_libraries(universal_encoder
    ${log-lib}
    ${android-lib}
    onnxruntime
    lz4
    msgpackc
)

# Copy ONNX Runtime libraries to output
add_custom_command(TARGET universal_encoder POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${ONNXRUNTIME_ROOT_PATH}/${ANDROID_ABI}/libonnxruntime.so
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libonnxruntime.so
)
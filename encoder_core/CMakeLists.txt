# encoder_core/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(UniversalEncoderCore VERSION 1.0.0)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Threads REQUIRED)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(USE_SYSTEM_ONNXRUNTIME "Use system ONNX Runtime" OFF)

# External dependencies
include(FetchContent)

# JSON library
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)

# LZ4 compression
FetchContent_Declare(
    lz4
    GIT_REPOSITORY https://github.com/lz4/lz4.git
    GIT_TAG v1.9.4
)

FetchContent_MakeAvailable(json lz4)

# ONNX Runtime
if(NOT USE_SYSTEM_ONNXRUNTIME)
    # Download pre-built ONNX Runtime
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-x86_64-1.16.3.tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-win-x64-1.16.3.zip")
    endif()
    
    FetchContent_Declare(
        onnxruntime
        URL ${ONNXRUNTIME_URL}
    )
    FetchContent_MakeAvailable(onnxruntime)
    
    set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
    set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
else()
    find_package(onnxruntime REQUIRED)
endif()

# Create library
add_library(universal_encoder_core
    src/universal_encoder.cpp
    src/vocabulary_pack.cpp
)

# Include directories
target_include_directories(universal_encoder_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(universal_encoder_core
    PUBLIC
        nlohmann_json::nlohmann_json
        lz4_static
        Threads::Threads
)

# Link ONNX Runtime
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(universal_encoder_core PUBLIC ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib)
else()
    target_link_libraries(universal_encoder_core PUBLIC ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)
endif()

# Platform-specific settings
if(ANDROID)
    target_compile_definitions(universal_encoder_core PRIVATE __ANDROID__)
elseif(IOS)
    target_compile_definitions(universal_encoder_core PRIVATE __APPLE__)
endif()

# Export targets
install(TARGETS universal_encoder_core
    EXPORT UniversalEncoderTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
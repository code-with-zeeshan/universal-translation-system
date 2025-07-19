# encoder_core/tests/CMakeLists.txt
# Add Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Create test executable
add_executable(encoder_tests
    test_vocabulary_pack.cpp
    test_encoder.cpp
)

target_link_libraries(encoder_tests
    universal_encoder_core
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(encoder_tests)
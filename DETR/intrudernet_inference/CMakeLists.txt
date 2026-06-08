cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
project(intrudernet_inference LANGUAGES CXX)

# =========================
# Compiler settings
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

SET(TEST_MODE True)
if (TEST_MODE)
    add_compile_definitions(TEST_MODE=1)
else()
    add_compile_definitions(TEST_MODE=0)
endif()

# =========================
# OpenCV
# =========================
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# =========================
# JsonCpp
# =========================
find_package(jsoncpp REQUIRED)

# =========================
# LibTorch
# =========================set(Torch_DIR
set(Torch_DIR ${CMAKE_SOURCE_DIR}/thirdparty/libtorch/share/cmake/Torch)    
find_package(Torch REQUIRED)
message(STATUS "Found Torch")

# =========================
# Executable
# =========================
if (TEST_MODE)    
    add_executable(intrudernet_test
        src/main.cpp
        inference/inference.cpp
        test/test.cpp
    )

    # =========================
    # Include paths
    # =========================
    target_include_directories(intrudernet_test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
    )

    # =========================
    # Link libraries
    # =========================
    target_link_libraries(intrudernet_test
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        jsoncpp
    )

    # =========================
    # Torch compile flags
    # =========================
    set_property(TARGET intrudernet_test PROPERTY CXX_STANDARD 17)

    # Fix for Raspberry Pi + GCC
    target_compile_definitions(intrudernet_test PRIVATE
        -D_GLIBCXX_USE_CXX11_ABI=1
    )

    # =========================
    # RPATH (important on Pi)
    # =========================
    set_target_properties(intrudernet_test PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    )
else()
    add_executable(intrudernet
        src/main.cpp
        inference/inference.cpp
    )

    # =========================
    # Include paths
    # =========================
    target_include_directories(intrudernet PRIVATE
        ${OpenCV_INCLUDE_DIRS}
    )

    # =========================
    # Link libraries
    # =========================
    target_link_libraries(intrudernet
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        jsoncpp
    )

    # =========================
    # Torch compile flags
    # =========================
    set_property(TARGET intrudernet PROPERTY CXX_STANDARD 17)

    # Fix for Raspberry Pi + GCC
    target_compile_definitions(intrudernet PRIVATE
        -D_GLIBCXX_USE_CXX11_ABI=1
    )

    # =========================
    # RPATH (important on Pi)
    # =========================
    set_target_properties(intrudernet PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    )
endif()



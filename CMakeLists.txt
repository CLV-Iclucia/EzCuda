cmake_minimum_required(VERSION 3.29)
project(EzCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 52;70;75;86)

add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

file(GLOB_RECURSE HEADERS include/*.h include/*.hpp)

add_library(ez-cuda INTERFACE ${HEADERS})
target_include_directories(ez-cuda INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
target_link_libraries(ez-cuda INTERFACE cuda)

file(GLOB TESTS tests/*.cu)
foreach (TEST ${TESTS})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    add_executable(${TEST_NAME} ${TEST} ${HEADERS})
    target_link_libraries(${TEST_NAME} ez-cuda)
endforeach ()
cmake_minimum_required(VERSION 3.15)
project(maba_v1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

include_directories(include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -Wall -Wextra")
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(maba_cli src/main.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(maba_cli PRIVATE OpenMP::OpenMP_CXX)
endif()



cmake_minimum_required(VERSION 3.16)
project(pendulum_renderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PENDULUM_RENDERER_WARNINGS "Enable extra compiler warnings" ON)

if(PENDULUM_RENDERER_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
    else()
        add_compile_options(-Wall -Wextra -pedantic)
    endif()
endif()

add_executable(pendulum_renderer renderer.cpp)

if(NOT MSVC)
    target_compile_options(pendulum_renderer PRIVATE -O3)
endif()


cmake_minimum_required(VERSION 3.20)

project(chemworld_cpu VERSION 0.1.0 LANGUAGES CXX)

option(CHEMWORLD_CPU_BUILD_TESTS "Build the native ABI smoke test" OFF)

add_library(chemworld_cpu SHARED src/c_api.cpp)
target_compile_features(chemworld_cpu PRIVATE cxx_std_20)
target_compile_definitions(chemworld_cpu PRIVATE CHEMWORLD_CPU_BUILD)
target_include_directories(
    chemworld_cpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(
    chemworld_cpu
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        OUTPUT_NAME chemworld_cpu
)

include(GNUInstallDirs)
install(
    TARGETS chemworld_cpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})

if(CHEMWORLD_CPU_BUILD_TESTS)
    enable_testing()
    add_executable(chemworld_cpu_smoke tests/smoke.cpp)
    target_link_libraries(chemworld_cpu_smoke PRIVATE chemworld_cpu)
    target_compile_features(chemworld_cpu_smoke PRIVATE cxx_std_20)
    add_test(NAME chemworld_cpu_smoke COMMAND chemworld_cpu_smoke)
endif()

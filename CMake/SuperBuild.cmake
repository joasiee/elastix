include(ExternalProject)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0135 NEW)

set (DEPENDENCIES_PREFIX ${PROJECT_SOURCE_DIR}/build/external)
set (PLASTIMATCH_PATCH ${PROJECT_SOURCE_DIR}/tools/plastimatch.patch)

set (EXTRA_CMAKE_ARGS
    --no-warn-unused-cli
    -G ${CMAKE_GENERATOR}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -g")
set(CMAKE_CACHE_ARGS 
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DITK_DIR:STRING=${DEPENDENCIES_PREFIX}/src/itk-build
)

set(TARGET_DEPENDENCIES)

set(Eigen3_DIR ${DEPENDENCIES_PREFIX}/install/share/eigen3/cmake)
find_package( Eigen3 QUIET NO_DEFAULT_PATH)

if(NOT ${Eigen3_FOUND})
    message(STATUS "Eigen3 not found, downloading and building")
    ExternalProject_Add(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
        PREFIX ${DEPENDENCIES_PREFIX}
        INSTALL_DIR ${DEPENDENCIES_PREFIX}/install
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:STRING=${DEPENDENCIES_PREFIX}/install ${EXTRA_CMAKE_ARGS} -DCMAKE_BUILD_TYPE:STRING=Release
        CMAKE_CACHE_ARGS ${CMAKE_CACHE_ARGS}
        BUILD_COMMAND ${CMAKE_COMMAND} --build ${DEPENDENCIES_PREFIX}/src/eigen-build --config Release --target all
    )
    set(TARGET_DEPENDENCIES ${TARGET_DEPENDENCIES} eigen)
endif()

set(ITK_DIR ${DEPENDENCIES_PREFIX}/src/itk-build)
find_package( ITK 5.3 QUIET NO_DEFAULT_PATH)
if(NOT ${ITK_FOUND})
    message(STATUS "ITK not found, downloading and building")
    ExternalProject_Add(
        itk
        GIT_REPOSITORY https://github.com/joasiee/ITK.git
        GIT_TAG origin/gomea
        PREFIX ${DEPENDENCIES_PREFIX}
        INSTALL_DIR ${DEPENDENCIES_PREFIX}/install
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:STRING=${DEPENDENCIES_PREFIX}/install ${EXTRA_CMAKE_ARGS} -DCMAKE_BUILD_TYPE:STRING=Release
        CMAKE_CACHE_ARGS ${CMAKE_CACHE_ARGS}
        BUILD_COMMAND ${CMAKE_COMMAND} --build ${DEPENDENCIES_PREFIX}/src/itk-build --config Release --target all
        INSTALL_COMMAND ""
    )
    set(TARGET_DEPENDENCIES ${TARGET_DEPENDENCIES} itk)
endif()


set(PLASTIMATCH_ARGS
    -DPLM_BUILD_TESTING:BOOL=FALSE
    -DPLM_CONFIG_ENABLE_CUDA:BOOL=FALSE
    -DPLM_CONFIG_ENABLE_DCMTK:BOOL=FALSE
    -DPLM_CONFIG_ENABLE_OPENCL:BOOL=FALSE
    -DCMAKE_PREFIX_PATH:STRING=${DEPENDENCIES_PREFIX}/install
)

set(Plastimatch_DIR ${DEPENDENCIES_PREFIX}/install/lib/cmake/plastimatch)
find_package( Plastimatch QUIET NO_DEFAULT_PATH)
if(NOT ${Plastimatch_FOUND})
    message(STATUS "Plastimatch not found, downloading and building")
    ExternalProject_Add(
        plastimatch
        DEPENDS ${TARGET_DEPENDENCIES}
        GIT_REPOSITORY https://gitlab.com/plastimatch/plastimatch.git
        GIT_TAG 3734adbfdb0b4cdce733ef1275b4e76093ffc6d5
        PATCH_COMMAND ${GIT_EXECUTABLE} apply ${PLASTIMATCH_PATCH} || echo "patch already applied"
        BUILD_COMMAND ${CMAKE_COMMAND} --build ${DEPENDENCIES_PREFIX}/src/plastimatch-build --config Release --target plmregister
        PREFIX ${DEPENDENCIES_PREFIX}
        INSTALL_DIR ${DEPENDENCIES_PREFIX}/install
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:STRING=${DEPENDENCIES_PREFIX}/install ${PLASTIMATCH_ARGS} ${EXTRA_CMAKE_ARGS} -DCMAKE_BUILD_TYPE:STRING=Release
        CMAKE_CACHE_ARGS ${CMAKE_CACHE_ARGS}
)
set(TARGET_DEPENDENCIES ${TARGET_DEPENDENCIES} plastimatch)
endif()

set(ELASTIX_ARGS
    -DCMAKE_PREFIX_PATH:STRING=${DEPENDENCIES_PREFIX}/install
    -DPlastimatch_DIR:STRING=${DEPENDENCIES_PREFIX}/install/lib/cmake
    -DCMAKE_INCLUDE_PATH:STRING=${DEPENDENCIES_PREFIX}/install/include
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
)

ExternalProject_Add (
    elastix
    DEPENDS ${TARGET_DEPENDENCIES}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}
    CMAKE_ARGS -DUSE_SUPERBUILD=OFF ${EXTRA_CMAKE_ARGS} ${ELASTIX_ARGS}
    CMAKE_CACHE_ARGS ${CMAKE_CACHE_ARGS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --config ${CMAKE_BUILD_TYPE} --target all
    INSTALL_COMMAND ""
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

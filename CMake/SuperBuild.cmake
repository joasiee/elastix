include(ExternalProject)
find_package(Git REQUIRED)

set (DEPENDENCIES_PREFIX ${PROJECT_SOURCE_DIR}/external)
set (PLASTIMATCH_PATCH ${PROJECT_SOURCE_DIR}/tools/plastimatch.patch)
set (EXTRA_CMAKE_ARGS
    --no-warn-unused-cli
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DITK_DIR:STRING=${DEPENDENCIES_PREFIX}/src/itk-build
    -G ${CMAKE_GENERATOR}
)


# ExternalProject_Add(
#     eigen
#     URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
#     PREFIX ${DEPENDENCIES_PREFIX}
#     INSTALL_DIR ${DEPENDENCIES_PREFIX}/install
#     CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:STRING=${DEPENDENCIES_PREFIX}/install ${EXTRA_CMAKE_ARGS}
#     BUILD_COMMAND ${CMAKE_COMMAND} --build ${DEPENDENCIES_PREFIX}/src/eigen-build --config Release --target all
# )

# ExternalProject_Add(
#     itk
#     GIT_REPOSITORY https://github.com/joasiee/ITK.git
#     GIT_TAG origin/gomea
#     PREFIX ${DEPENDENCIES_PREFIX}
#     INSTALL_DIR ${DEPENDENCIES_PREFIX}/install
#     CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:STRING=${DEPENDENCIES_PREFIX}/install ${EXTRA_CMAKE_ARGS}
#     BUILD_COMMAND ${CMAKE_COMMAND} --build ${DEPENDENCIES_PREFIX}/src/itk-build --config Release --target all
#     INSTALL_COMMAND ""
# )


set(PLASTIMATCH_ARGS
    -DPLM_BUILD_TESTING:BOOL=FALSE
    -DPLM_CONFIG_ENABLE_CUDA:BOOL=FALSE
    -DPLM_CONFIG_ENABLE_DCMTK:BOOL=FALSE
    -DPLM_CONFIG_ENABLE_OPENCL:BOOL=FALSE
    -DCMAKE_PREFIX_PATH:STRING=${DEPENDENCIES_PREFIX}/install
)

ExternalProject_Add(
    plastimatch
    # DEPENDS itk
    GIT_REPOSITORY https://gitlab.com/plastimatch/plastimatch.git
    GIT_TAG 3734adbfdb0b4cdce733ef1275b4e76093ffc6d5
    PATCH_COMMAND ${GIT_EXECUTABLE} apply ${PLASTIMATCH_PATCH} || echo "patch already applied"
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${DEPENDENCIES_PREFIX}/src/plastimatch-build --config Release --target plmregister
    PREFIX ${DEPENDENCIES_PREFIX}
    INSTALL_DIR ${DEPENDENCIES_PREFIX}/install
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:STRING=${DEPENDENCIES_PREFIX}/install ${PLASTIMATCH_ARGS} ${EXTRA_CMAKE_ARGS}
)

# ExternalProject_Add (
#     elastix
#     DEPENDS eigen itk plastimatch
#     SOURCE_DIR ${PROJECT_SOURCE_DIR}
#     CMAKE_ARGS -DUSE_SUPERBUILD=OFF ${EXTRA_CMAKE_ARGS}
#     BUILD_COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --config ${CMAKE_BUILD_TYPE} --target all
#     INSTALL_COMMAND ""
#     BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
# )
add_executable(EllipseSphereScene EllipseSphereExample.cxx)
target_link_libraries(EllipseSphereScene ${ITK_LIBRARIES})

add_executable(SphereInCubeScene SphereInCube.cxx)
target_link_libraries(SphereInCubeScene ${ITK_LIBRARIES})

add_executable(AlternatingCubeSphereScene AlternatingCubeSphere.cxx)
target_link_libraries(AlternatingCubeSphereScene ${ITK_LIBRARIES})
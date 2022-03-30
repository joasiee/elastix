
ADD_ELXCOMPONENT(GOMEA
  elxGOMEA.h
  elxGOMEA.hxx
  elxGOMEA.cxx
  itkGOMEAOptimizer.h
  itkGOMEAOptimizer.cxx
  util/FOS.h
  util/Tools.h
  util/FOS.cxx
  util/Tools.cxx
)

target_link_libraries(GOMEA
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${BOOST_LIBRARIES}
)
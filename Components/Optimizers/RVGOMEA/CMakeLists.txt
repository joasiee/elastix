
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

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Boost 1.70.0 REQUIRED)

target_link_libraries(GOMEA
  BLAS::BLAS
  LAPACK::LAPACK
  Boost::boost
)
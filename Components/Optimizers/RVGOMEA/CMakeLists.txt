
ADD_ELXCOMPONENT(GOMEA
  elxGOMEA.h
  elxGOMEA.hxx
  elxGOMEA.cxx
  itkGOMEAOptimizer.h
  itkGOMEAOptimizer.cxx
  util/FOS.h
  util/Tools.h
  util/Instrumentor.hpp
  util/FOS.cxx
  util/Tools.cxx
)

target_link_libraries(GOMEA
  BLAS::BLAS
  LAPACK::LAPACK
  Boost::boost
)
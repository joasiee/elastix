#ifndef elxGOMEA_h
#define elxGOMEA_h

#include "elxIncludes.h"
#include "itkGOMEAOptimizer.h"

namespace elastix
{
template <class TElastix>
class ITK_TEMPLATE_EXPORT GOMEA
  : public itk::GOMEAOptimizer
  , public OptimizerBase<TElastix>
{
public:
  /** Standard ITK.*/
  typedef GOMEA          Self;
  typedef GOMEAOptimizer Superclass1;
  typedef OptimizerBase<TElastix>       Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  itkNewMacro(Self);
  itkTypeMacro(GOMEA, GOMEAOptimizer);
  elxClassNameMacro("GOMEA");

  /** Typedef's inherited from Superclass1.*/
  using Superclass1::CostFunctionType;
  using Superclass1::CostFunctionPointer;
  using Superclass1::StopConditionType;
  using Superclass1::ParametersType;
  using Superclass1::DerivativeType;
  using Superclass1::ScalesType;

  /** Typedef's inherited from Elastix.*/
  using typename Superclass2::ElastixType;
  using typename Superclass2::ElastixPointer;
  using typename Superclass2::ConfigurationType;
  using typename Superclass2::ConfigurationPointer;
  using typename Superclass2::RegistrationType;
  using typename Superclass2::RegistrationPointer;
  typedef typename Superclass2::ITKBaseType ITKBaseType;

  /** Check if any scales are set, and set the UseScales flag on or off;
   * after that call the superclass' implementation */
  void
  StartOptimization(void) override;

  /** Methods to set parameters and print output at different stages
   * in the registration process.*/
  void
  BeforeRegistration(void) override;

  void
  BeforeEachResolution(void) override;

  void
  AfterEachResolution(void) override;

  void
  AfterEachIteration(void) override;

  void
  AfterRegistration(void) override;

protected:
  GOMEA() = default;
  ~GOMEA() override = default;

private:
  elxOverrideGetSelfMacro;

  GOMEA(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxGOMEA.hxx"
#endif

#endif

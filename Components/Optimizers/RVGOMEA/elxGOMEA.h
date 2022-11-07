#ifndef elxGOMEA_h
#define elxGOMEA_h

#include "elxIncludes.h"
#include "itkGOMEAOptimizer.h"
#include "itkAdvancedImageToImageMetric.h"

namespace elastix
{
template <class TElastix>
class ITK_TEMPLATE_EXPORT GOMEA final
  : public itk::GOMEAOptimizer
  , public OptimizerBase<TElastix>
{
public:
  /** Standard ITK.*/
  typedef GOMEA                         Self;
  typedef GOMEAOptimizer                Superclass1;
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

  /** Typedef's inherited from Superclass2. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using SizeValueType = itk::SizeValueType;

  /** Other typedef's. */
  using FixedImageType = typename ElastixType::FixedImageType;
  using FixedPointType = typename FixedImageType::PointType;
  using FixedPointValueType = typename FixedPointType::ValueType;
  using MovingImageType = typename ElastixType::MovingImageType;
  using MovingPointType = typename MovingImageType::PointType;
  using MovingPointValueType = typename MovingPointType::ValueType;

  using AdvancedMetricType = itk::AdvancedImageToImageMetric<FixedImageType, MovingImageType>;
  using CombinationTransformType = itk::AdvancedCombinationTransform<double, FixedImageType::ImageDimension>;
  using BSplineBaseType = itk::AdvancedBSplineDeformableTransformBase<double, FixedImageType::ImageDimension>;

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

  size_t
  GetNumberOfPixelEvaluations() const override
  {
    using MetricType = typename ElastixType::MetricBaseType::AdvancedMetricType;
    MetricType * testPtr = dynamic_cast<MetricType *>(this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType());
    size_t       numberOfPixelEvaluations = testPtr ? testPtr->GetNumberOfPixelEvaluations() : 0;
    return numberOfPixelEvaluations;
  }

private:
  elxOverrideGetSelfMacro;

  GOMEA(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  std::ofstream m_DistMultOutFile;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxGOMEA.hxx"
#endif

#endif

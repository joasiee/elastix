#pragma once

#include "elxIncludes.h"
#include "itkMOGOMEAOptimizer.h"
#include "../../Registrations/MultiMetricMultiResolutionRegistration/itkCombinationImageToImageMetric.h"

namespace elastix
{
template <class TElastix>
class ITK_TEMPLATE_EXPORT MOGOMEA
  : public itk::MOGOMEAOptimizer
  , public OptimizerBase<TElastix>
{
public:
  /** Standard ITK.*/
  typedef MOGOMEA                       Self;
  typedef MOGOMEAOptimizer              Superclass1;
  typedef OptimizerBase<TElastix>       Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  itkNewMacro(Self);
  itkTypeMacro(MOGOMEA, MOGOMEAOptimizer);
  elxClassNameMacro("MOGOMEA");

  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  using FixedImageType = typename ElastixType::FixedImageType;
  using FixedPointType = typename FixedImageType::PointType;
  using FixedPointValueType = typename FixedPointType::ValueType;
  using MovingImageType = typename ElastixType::MovingImageType;
  using MovingPointType = typename MovingImageType::PointType;
  using MovingPointValueType = typename MovingPointType::ValueType;
  
  using Superclass1::ParametersType;
  using Superclass1::MeasureType;

  using AdvancedMetricType = itk::AdvancedImageToImageMetric<FixedImageType, MovingImageType>;
  using CombinationTransformType = itk::AdvancedCombinationTransform<double, FixedImageType::ImageDimension>;
  using BSplineBaseType = itk::AdvancedBSplineDeformableTransformBase<double, FixedImageType::ImageDimension>;

  using CombinationMetricType = itk::CombinationImageToImageMetric<FixedImageType, MovingImageType>;

  void
  BeforeRegistration(void) override;

  void
  BeforeEachResolution(void) override;

  void
  AfterEachResolution(void) override;

  void
  AfterEachIteration(void) override;

protected:
  MOGOMEA() = default;
  ~MOGOMEA() override = default;

  void
  UpdateMetricIterationOutput() override;

  void
  InitializeRegistration() override;

  void
  SetPositionForMixingComponent(int component_index, const ParametersType & parameters);

  const ParametersType &
  GetPositionForMixingComponent(int component_index) const override;

  MeasureType
  GetValueForMetric(int metric_index) const override;

  void
  WriteTransformParam(int index) const override;

  size_t
  GetNumberOfPixelEvaluations() const override
  {
    using MetricType = typename ElastixType::MetricBaseType::AdvancedMetricType;
    MetricType * testPtr = dynamic_cast<MetricType *>(this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType());
    return testPtr ? testPtr->GetNumberOfPixelEvaluations() : 0;
  }

private:
  elxOverrideGetSelfMacro;

  MOGOMEA(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  const CombinationMetricType *
  GetCostFunctionAsCombinationMetric() const
  {
    return dynamic_cast<const CombinationMetricType *>(this->m_Registration->GetAsITKBaseType()->GetMetric());
  }

  CombinationMetricType *
  GetCostFunctionAsCombinationMetric()
  {
    return dynamic_cast<CombinationMetricType *>(this->m_Registration->GetAsITKBaseType()->GetMetric());
  }
};
} // namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMOGOMEA.hxx"
#endif

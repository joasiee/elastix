#pragma once

#include "elxMOGOMEA.h"

namespace elastix
{
template <class TElastix>
void
MOGOMEA<TElastix>::BeforeRegistration(void)
{
  const CombinationMetricType * combinationMetric = this->GetCostFunctionAsCombinationMetric();
  if (combinationMetric)
    MOGOMEA_UTIL::number_of_objectives = combinationMetric->GetNumberOfMetrics();
  else
    itkExceptionMacro("Need to use the CombinationImageToImageMetric for MOGOMEA.");

  const Configuration & configuration = Deref(Superclass2::GetConfiguration());

  int64_t randomSeed = 0;
  configuration.ReadParameter(randomSeed, "RandomSeed", 0, false);
  MOGOMEA_UTIL::random_seed_changing = static_cast<int64_t>(randomSeed);

  MOGOMEA_UTIL::output_folder = configuration.GetCommandLineArgument("-out");
  std::filesystem::create_directories(MOGOMEA_UTIL::output_folder + "solutions/");
  std::filesystem::create_directories(MOGOMEA_UTIL::output_folder + "approximation/");

  this->m_ImageDimension = this->GetElastix()->GetFixedImage()->GetImageDimension();

  this->AddTargetCellToIterationInfo("3a:|El.Archive|");
  this->AddTargetCellToIterationInfo("3b:Hypervolume");
  this->AddTargetCellToIterationInfo("3c:DistMult");

  this->GetIterationInfoAt("3a:|El.Archive|") << std::left << std::setw(12) << std::setfill(' ');
  this->GetIterationInfoAt("3b:Hypervolume") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3c:DistMult") << std::showpoint << std::fixed;
}

template <class TElastix>
void
MOGOMEA<TElastix>::BeforeEachResolution(void)
{
  unsigned int level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());
  this->m_CurrentResolution = level;
  const Configuration & configuration = Deref(Superclass2::GetConfiguration());

  MOGOMEA_UTIL::write_generational_solutions = 0;
  configuration.ReadParameter(
    MOGOMEA_UTIL::write_generational_solutions, "WriteGenerationalSolutions", this->GetComponentLabel(), level, 0);

  MOGOMEA_UTIL::elitist_archive_size_target = 500;
  configuration.ReadParameter(
    MOGOMEA_UTIL::elitist_archive_size_target, "ElitistArchiveSizeTarget", this->GetComponentLabel(), level, 0);

  MOGOMEA_UTIL::use_constraints = 0;
  configuration.ReadParameter(MOGOMEA_UTIL::use_constraints, "UseConstraints", this->GetComponentLabel(), level, 0);

  MOGOMEA_UTIL::FOS_element_size = -6;
  configuration.ReadParameter(MOGOMEA_UTIL::FOS_element_size, "FosElementSize", this->GetComponentLabel(), level, 0);

  this->tau = 0.35;
  configuration.ReadParameter(this->tau, "Tau", this->GetComponentLabel(), level, 0);

  this->use_forced_improvement = 0;
  configuration.ReadParameter(
    this->use_forced_improvement, "UseForcedImprovement", this->GetComponentLabel(), level, 0);

  this->maximum_number_of_populations = 1;
  configuration.ReadParameter(
    this->maximum_number_of_populations, "MaximumNumberOfPopulations", this->GetComponentLabel(), level, 0);

  this->base_number_of_mixing_components = 0;
  configuration.ReadParameter(
    this->base_number_of_mixing_components, "BaseNumberOfMixingComponents", this->GetComponentLabel(), level, 0);

  this->base_population_size = 0;
  configuration.ReadParameter(this->base_population_size, "BasePopulationSize", this->GetComponentLabel(), level, 0);

  this->distribution_multiplier_decrease = 0.9;
  configuration.ReadParameter(
    this->distribution_multiplier_decrease, "DistributionMultiplierDecrease", this->GetComponentLabel(), level, 0);

  this->st_dev_ratio_threshold = 1.0;
  configuration.ReadParameter(this->st_dev_ratio_threshold, "StDevRatioThreshold", this->GetComponentLabel(), level, 0);

  this->maximum_no_improvement_stretch = 0;
  configuration.ReadParameter(
    this->maximum_no_improvement_stretch, "MaximumNoImprovementStretch", this->GetComponentLabel(), level, 0);

  this->maximum_number_of_evaluations = 0;
  configuration.ReadParameter(
    this->maximum_number_of_evaluations, "MaximumNumberOfEvaluations", this->GetComponentLabel(), level, 0);

  this->maximum_number_of_seconds = 0;
  configuration.ReadParameter(
    this->maximum_number_of_seconds, "MaximumNumberOfSeconds", this->GetComponentLabel(), level, 0);

  this->maximum_number_of_generations = 0;
  configuration.ReadParameter(
    this->maximum_number_of_generations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);

  this->maximum_number_of_pixel_evaluations = 0;
  configuration.ReadParameter(
    this->maximum_number_of_pixel_evaluations, "MaximumNumberOfPixelEvaluations", this->GetComponentLabel(), level, 0);

  this->number_of_subgenerations_per_population_factor = 8;
  configuration.ReadParameter(this->number_of_subgenerations_per_population_factor,
                              "NumberOfSubgenerationsPerPopulationFactor",
                              this->GetComponentLabel(),
                              level,
                              0);

  this->m_PartialEvaluations = true;
  configuration.ReadParameter(
    this->m_PartialEvaluations, "PartialEvaluations", this->GetComponentLabel(), level, false);

  // Get the transform pointer casted as bspline transform.
  auto transformPtr = this->GetElastix()->GetElxTransformBase()->GetAsITKBaseType();
  auto comboPtr = dynamic_cast<const CombinationTransformType *>(transformPtr);
  auto bsplinePtr = dynamic_cast<const BSplineBaseType *>(comboPtr->GetCurrentTransform());

  // Get the bspline grid size dimensions
  auto gridRegionSize = bsplinePtr->GetGridRegion().GetSize();
  MOGOMEA_UTIL::grid_region_dimensions = std::vector<int>(gridRegionSize.begin(), gridRegionSize.end());
}

template <class TElastix>
void
MOGOMEA<TElastix>::AfterEachResolution(void)
{
  std::string stopcondition;

  switch (m_StopCondition)
  {
    case StopConditionType::MaximumNumberOfEvaluationsTermination:
      stopcondition = "Maximum number of evaluations has been reached";
      break;

    case StopConditionType::MaximumNumberOfPixelEvaluationsTermination:
      stopcondition = "Maximum number of pixel evaluations has been reached";
      break;

    case StopConditionType::MaximumNumberOfSecondsTermination:
      stopcondition = "Maximum number of seconds has been reached";
      break;

    case StopConditionType::MaximumNumberOfGenerationsTermination:
      stopcondition = "Maximum number of generations has been reached";
      break;

    case StopConditionType::DistributionMultiplierTermination:
      stopcondition = "Distribution multiplier termination";
      break;

    case StopConditionType::Unknown:
      stopcondition = "Unknown";
      break;
  }

  /** Print the stopping condition */
  log::info(std::ostringstream{} << "Stopping condition: " << stopcondition << ".\n");
}

template <class TElastix>
void
MOGOMEA<TElastix>::AfterEachIteration(void)
{
  this->GetIterationInfoAt("3a:|El.Archive|")
    << std::left << std::setw(12) << std::setfill(' ') << MOGOMEA_UTIL::elitist_archive_size;
  this->GetIterationInfoAt("3b:Hypervolume") << MOGOMEA_UTIL::last_hyper_volume;
  this->GetIterationInfoAt("3c:DistMult") << this->ComputeAverageDistributionMultiplier();

  if (this->GetNewSamplesEveryIteration())
  {
    this->SelectNewSamples();
  }
}

template <class TElastix>
void
MOGOMEA<TElastix>::UpdateMetricIterationOutput()
{
  CombinationMetricType * combinationMetric = this->GetCostFunctionAsCombinationMetric();
  for (int i = 0; i < MOGOMEA_UTIL::number_of_objectives; ++i)
  {
    combinationMetric->SetMetricValue(MOGOMEA_UTIL::best_objective_values_in_elitist_archive[i], i);
  }

  MOGOMEA_UTIL::computeApproximationSet();
  MOGOMEA_UTIL::last_hyper_volume =
    MOGOMEA_UTIL::compute2DHyperVolume(MOGOMEA_UTIL::approximation_set, MOGOMEA_UTIL::approximation_set_size);
  MOGOMEA_UTIL::freeApproximationSet();
}

template <class TElastix>
void
MOGOMEA<TElastix>::InitializeRegistration()
{
  this->m_Registration->GetAsITKBaseType()->InitializeForMOGOMEA(this->base_number_of_mixing_components);
}

template <class TElastix>
void
MOGOMEA<TElastix>::SetPositionForMixingComponent(int component_index, const ParametersType & parameters)
{
  this->m_Registration->GetAsITKBaseType()->SetPositionForMixingComponent(component_index, parameters);
}

template <class TElastix>
auto
MOGOMEA<TElastix>::GetPositionForMixingComponent(int component_index) const -> const ParametersType &
{
  return this->m_Registration->GetAsITKBaseType()->GetPositionForMixingComponentOfNextLevel(component_index);
}

template <class TElastix>
auto
MOGOMEA<TElastix>::GetValueForMetric(int metric_index) const -> MeasureType
{
  const CombinationMetricType * combinationMetric = this->GetCostFunctionAsCombinationMetric();
  return combinationMetric->GetMetricValue(metric_index);
}
template <class TElastix>
void
MOGOMEA<TElastix>::WriteTransformParam(int index) const
{
  std::string filename = MOGOMEA_UTIL::output_folder + "approximation/" + std::to_string(index) + "_TransformParameters.txt";
  this->GetElastix()->CreateTransformParameterFile(filename, false);
}
} // namespace elastix

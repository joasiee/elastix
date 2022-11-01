#ifndef elxGOMEA_hxx
#define elxGOMEA_hxx

#include "elxGOMEA.h"

namespace elastix
{

template <class TElastix>
void
GOMEA<TElastix>::BeforeRegistration(void)
{
  long unsigned int randomSeed = 0;
  if (this->GetConfiguration()->ReadParameter(randomSeed, "RandomSeed", 0, false))
    this->SetRandomSeed(randomSeed);

  this->SetImageDimension(this->GetElastix()->GetFixedImage()->GetImageDimension());

  this->AddTargetCellToIterationInfo("2:Metric");
  this->AddTargetCellToIterationInfo("3a:PdPct");
  this->AddTargetCellToIterationInfo("3b:DistMult");
  this->AddTargetCellToIterationInfo("3c:Constraints");

  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3a:PdPct") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3b:DistMult") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3c:Constraints") << std::showpoint << std::fixed;


  std::string sampler;
  this->m_Configuration->ReadParameter(sampler, "ImageSampler", 0, true);

  if (sampler != "Full")
  {
    this->m_SubSampling = true;
  }

  m_outFolderProfiling = this->m_Configuration->GetCommandLineArgument("-out") + "profiling_output/";
}

template <class TElastix>
void
GOMEA<TElastix>::AfterEachIteration(void)
{
  float pdpct_mean = boost::accumulators::mean(this->m_PdPctMean);
  pdpct_mean = isnan(pdpct_mean) ? 100.0f : pdpct_mean;

  /** Print some information. */
  this->GetIterationInfoAt("2:Metric") << this->m_Value;
  this->GetIterationInfoAt("3a:PdPct") << pdpct_mean;
  this->GetIterationInfoAt("3b:DistMult") << this->GetAverageDistributionMultiplier();
  this->GetIterationInfoAt("3c:Constraints") << this->m_ConstraintValue;
  this->m_PdPctMean = {};

  this->WriteDistributionMultipliers(this->m_DistMultOutFile);

  /** Select new samples if desired. These
   * will be used in the next iteration */
  if (this->GetNewSamplesEveryIteration())
  {
    this->SelectNewSamples();
  }
}

template <class TElastix>
void
GOMEA<TElastix>::BeforeEachResolution(void)
{
  /** Get the current resolution level.*/
  unsigned int level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  /** Set MaximumNumberOfIterations.*/
  int maximumNumberOfIterations = 100;
  this->m_Configuration->ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetMaximumNumberOfIterations(maximumNumberOfIterations);

  /** Set MaxNumberOfEvaluations.*/
  unsigned long maxNumberOfEvaluations = 0L;
  this->m_Configuration->ReadParameter(
    maxNumberOfEvaluations, "MaxNumberOfEvaluations", this->GetComponentLabel(), level, 0);
  this->SetMaxNumberOfEvaluations(maxNumberOfEvaluations);

  /** Set FosElementSize.*/
  int fosElementSize = -1;
  this->m_Configuration->ReadParameter(fosElementSize, "FosElementSize", this->GetComponentLabel(), level, 0);
  this->SetFosElementSize(fosElementSize);

  /** Set StaticLinkageType.*/
  int staticLinkageType = 0;
  this->m_Configuration->ReadParameter(staticLinkageType, "StaticLinkageType", this->GetComponentLabel(), level, 0);
  this->SetStaticLinkageType(staticLinkageType);

  /** Set StaticLinkageMaxSetSize.*/
  int staticLinkageMaxSetSize = 24;
  this->m_Configuration->ReadParameter(
    staticLinkageMaxSetSize, "StaticLinkageMaxSetSize", this->GetComponentLabel(), level, 0);
  this->SetStaticLinkageMaxSetSize(staticLinkageMaxSetSize);

  /** Set MaxImprovementNoStretch.*/
  int maxNoImprovementNoStretch = 0;
  this->m_Configuration->ReadParameter(
    maxNoImprovementNoStretch, "MaxImprovementNoStretch", this->GetComponentLabel(), level, 0);
  this->SetMaxNoImprovementStretch(maxNoImprovementNoStretch);

  /** Set BasePopulationSize */
  int basePopulationSize = 0;
  this->m_Configuration->ReadParameter(basePopulationSize, "BasePopulationSize", this->GetComponentLabel(), level, 0);
  this->SetBasePopulationSize(basePopulationSize);

  /** Set MaxNumberOfPopulations */
  int maxNumberOfPopulations = 1;
  this->m_Configuration->ReadParameter(
    maxNumberOfPopulations, "MaxNumberOfPopulations", this->GetComponentLabel(), level, 0);
  this->SetMaxNumberOfPopulations(maxNumberOfPopulations);

  /** Set Tau*/
  double tau = 0.35;
  this->m_Configuration->ReadParameter(tau, "Tau", this->GetComponentLabel(), level, 0);
  this->SetTau(tau);

  /** Set DistributionMultiplierDecrease*/
  double distributionMultiplierDecrease = 0.9;
  this->m_Configuration->ReadParameter(
    distributionMultiplierDecrease, "DistributionMultiplierDecrease", this->GetComponentLabel(), level, 0);
  this->SetDistributionMultiplierDecrease(distributionMultiplierDecrease);

  /** Set StDevThreshold*/
  double stDevThreshold = 1.0;
  this->m_Configuration->ReadParameter(stDevThreshold, "StDevThreshold", this->GetComponentLabel(), level, 0);
  this->SetStDevThreshold(stDevThreshold);

  /** Set FitnessVarianceTolerance*/
  double fitnessVarianceTolerance = 1e-6;
  this->m_Configuration->ReadParameter(
    fitnessVarianceTolerance, "FitnessVarianceTolerance", this->GetComponentLabel(), level, 0);
  this->SetFitnessVarianceTolerance(fitnessVarianceTolerance);

  /** Set PartialEvaluations*/
  bool partialEvaluations = false;
  this->m_Configuration->ReadParameter(partialEvaluations, "PartialEvaluations", this->GetComponentLabel(), level, 0);
  this->SetPartialEvaluations(partialEvaluations);

  /** Set UseShrinkage*/
  bool useShrinkage = false;
  this->m_Configuration->ReadParameter(useShrinkage, "UseShrinkage", this->GetComponentLabel(), level, 0);
  this->SetUseShrinkage(useShrinkage);

  /** Set UseConstraints*/
  bool useConstraints = true;
  this->m_Configuration->ReadParameter(useConstraints, "UseConstraints", this->GetComponentLabel(), level, 0);
  this->SetUseConstraints(useConstraints);

  std::ostringstream makeFileName("");
  makeFileName << this->m_Configuration->GetCommandLineArgument("-out") << "R" << level << "_dist_mults.dat";
  std::string fileName = makeFileName.str();
  this->m_DistMultOutFile.open(fileName.c_str());

  // Get the transform pointer casted as bspline transform.
  auto transformPtr = this->GetElastix()->GetElxTransformBase()->GetAsITKBaseType();
  auto comboPtr = dynamic_cast<const CombinationTransformType *>(transformPtr);
  auto bsplinePtr = dynamic_cast<const BSplineBaseType *>(comboPtr->GetCurrentTransform());

  // Get the bspline grid size dimensions
  auto gridRegionSize = bsplinePtr->GetGridRegion().GetSize();
  m_BSplineGridRegionDimensions = std::vector<int>(gridRegionSize.begin(), gridRegionSize.end());
}

template <class TElastix>
void
GOMEA<TElastix>::AfterEachResolution(void)
{
  this->m_DistMultOutFile.close();

  std::string stopcondition;

  switch (this->GetStopCondition())
  {
    case MaximumNumberOfEvaluationsTermination:
      stopcondition = "Maximum number of evaluations has been reached";
      break;

    case MaximumNumberOfIterationsTermination:
      stopcondition = "Maximum number of iterations has been reached";
      break;

    case AverageFitnessTermination:
      stopcondition = "Average fitness delta below threshold";
      break;

    case FitnessVarianceTermination:
      stopcondition = "Fitness variance delta below threshold";
      break;

    case DistributionMultiplierTermination:
      stopcondition = "Distribution multiplier termination";
      break;

    case Unknown:
      stopcondition = "Unknown";
      break;
  }

  /** Print the stopping condition */
  elxout << "Stopping condition: " << stopcondition << ".\n";
}

template <class TElastix>
void
GOMEA<TElastix>::AfterRegistration(void)
{
  elxout << "\nFinal metric value = " << this->m_Value << "\n";
}
} // namespace elastix

#endif

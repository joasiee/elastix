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
  this->AddTargetCellToIterationInfo("3d:Evaluations");

  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3a:PdPct") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3b:DistMult") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3c:Constraints") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3d:Evaluations") << std::showpoint << std::fixed;


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
  this->GetIterationInfoAt("3d:Evaluations") << this->GetNumberOfPixelEvaluations() / 1e6;
  this->m_PdPctMean = {};

  this->WriteDistributionMultipliers(this->m_DistMultOutFile);

  if (this->GetWriteExtraOutput())
  {
    this->WriteTransformParametersWithConstraint(this->m_TransformParametersExtraOutFile);
    this->WriteMutualInformationMatrix();
  }

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
  m_CurrentResolution = level;

  /** Set MaximumNumberOfIterations.*/
  unsigned long maximumNumberOfIterations = 1000UL;
  this->m_Configuration->ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetMaximumNumberOfIterations(maximumNumberOfIterations);

  /** Set MaxNumberOfEvaluations.*/
  unsigned long maxNumberOfEvaluations = 0;
  this->m_Configuration->ReadParameter(
    maxNumberOfEvaluations, "MaxNumberOfEvaluations", this->GetComponentLabel(), level, 0);
  this->SetMaxNumberOfEvaluations(maxNumberOfEvaluations);

  /** Set MaxNumberOfPixelEvaluations.*/
  unsigned long maxNumberOfPixelEvaluations = 0L;
  this->m_Configuration->ReadParameter(
    maxNumberOfPixelEvaluations, "MaxNumberOfPixelEvaluations", this->GetComponentLabel(), level, 0);
  this->SetMaxNumberOfPixelEvaluations(maxNumberOfPixelEvaluations);

  /** Set FosElementSize.*/
  int fosElementSize = -1;
  this->m_Configuration->ReadParameter(fosElementSize, "FosElementSize", this->GetComponentLabel(), level, 0);
  this->SetFosElementSize(fosElementSize);

  /** Set StaticLinkageType.*/
  int staticLinkageType = 0;
  this->m_Configuration->ReadParameter(staticLinkageType, "StaticLinkageType", this->GetComponentLabel(), level, 0);
  this->SetStaticLinkageType(staticLinkageType);

  /** Set StaticLinkageMinSetSize.*/
  int staticLinkageMinSetSize = 3;
  this->m_Configuration->ReadParameter(
    staticLinkageMinSetSize, "StaticLinkageMinSetSize", this->GetComponentLabel(), level, 0);
  this->SetStaticLinkageMinSetSize(staticLinkageMinSetSize);

  /** Set NumberOfASGDIterations.*/
  int numberOfASGDIterations = 50;
  this->m_Configuration->ReadParameter(
    numberOfASGDIterations, "NumberOfASGDIterations", this->GetComponentLabel(), level, 0);
  this->SetNumberOfASGDIterations(numberOfASGDIterations);

  /** Set MaxNumberOfASGDIterations.*/
  int maxNumberOfASGDIterations = 200;
  this->m_Configuration->ReadParameter(
    maxNumberOfASGDIterations, "MaxNumberOfASGDIterations", this->GetComponentLabel(), level, 0);
  this->SetMaxNumberOfASGDIterations(maxNumberOfASGDIterations);

  /** Set MinNumberOfASGDIterations.*/
  int minNumberOfASGDIterations = 20;
  this->m_Configuration->ReadParameter(
    minNumberOfASGDIterations, "MinNumberOfASGDIterations", this->GetComponentLabel(), level, 0);
  this->SetMinNumberOfASGDIterations(minNumberOfASGDIterations);

  /** Set NumberOfASGDIterationsOffset.*/
  int numberOfASGDIterationsOffset = 20;
  this->m_Configuration->ReadParameter(
    numberOfASGDIterationsOffset, "NumberOfASGDIterationsOffset", this->GetComponentLabel(), level, 0);
  this->SetNumberOfASGDIterationsOffset(numberOfASGDIterationsOffset);

  /** Set StaticLinkageMaxSetSize.*/
  int staticLinkageMaxSetSize = 12;
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

  /** Set RedistributionMethod */
  int redistributionMethod = 1;
  this->m_Configuration->ReadParameter(
    redistributionMethod, "RedistributionMethod", this->GetComponentLabel(), level, 0);
  this->SetRedistributionMethod(static_cast<RedistributionMethod>(redistributionMethod));

  /** Set ASGDIterationSchedule */
  int asgdIterationSchedule = 1;
  this->m_Configuration->ReadParameter(
    asgdIterationSchedule, "ASGDIterationSchedule", this->GetComponentLabel(), level, 0);
  this->SetASGDIterationSchedule(static_cast<ASGDIterationSchedule>(asgdIterationSchedule));

  /** Set Tau*/
  double tau = 0.35;
  this->m_Configuration->ReadParameter(tau, "Tau", this->GetComponentLabel(), level, 0);
  this->SetTau(tau);

  /** Set TauASGD*/
  double tauAsgd = 0.1;
  this->m_Configuration->ReadParameter(tauAsgd, "TauASGD", this->GetComponentLabel(), level, 0);
  this->SetTauASGD(tauAsgd);

  /** Set AlphaASGD*/
  double alphaAsgd = 0.08;
  this->m_Configuration->ReadParameter(alphaAsgd, "AlphaASGD", this->GetComponentLabel(), level, 0);
  this->SetAlphaASGD(alphaAsgd);

  /** Set BetaASGD*/
  double betaAsgd = 0.1;
  this->m_Configuration->ReadParameter(betaAsgd, "BetaASGD", this->GetComponentLabel(), level, 0);
  this->SetBetaASGD(betaAsgd);

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
  double fitnessVarianceTolerance = 1e-9;
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
  bool useConstraints = false;
  this->m_Configuration->ReadParameter(useConstraints, "UseConstraints", this->GetComponentLabel(), level, 0);
  this->SetUseConstraints(useConstraints);

  /** Set UseASGD*/
  bool useASGD = false;
  this->m_Configuration->ReadParameter(useASGD, "UseASGD", this->GetComponentLabel(), level, 0);
  this->SetUseASGD(useASGD);

  /** Set WriteExtraOutput*/
  bool writeExtraOutput = false;
  this->m_Configuration->ReadParameter(writeExtraOutput, "WriteExtraOutput", this->GetComponentLabel(), level, 0);
  this->SetWriteExtraOutput(writeExtraOutput);

  if (this->GetWriteExtraOutput())
  {
    std::ostringstream makeFileName("");
    makeFileName << this->m_Configuration->GetCommandLineArgument("-out") << "R" << level
                 << "_transform_params_constraints.dat";
    std::string fileName = makeFileName.str();
    this->m_TransformParametersExtraOutFile.open(fileName.c_str());

    std::ostringstream matricesDir("");
    matricesDir << this->m_Configuration->GetCommandLineArgument("-out") << "mutual_information_matrices.R" << level
                << "/";
    this->m_MutualInformationOutDir = matricesDir.str();
    std::filesystem::create_directory(this->m_MutualInformationOutDir);
  }

  if (this->GetUseASGD())
  {
    m_ASGD = GradientDescentType::New();
    m_ASGD->SetElastix(this->GetElastix());
    m_ASGD->BeforeEachResolution();
  }

  // Open the output file for the distribution multipliers
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
  this->m_TransformParametersExtraOutFile.close();

  std::string stopcondition;

  switch (this->GetStopCondition())
  {
    case MaximumNumberOfEvaluationsTermination:
      stopcondition = "Maximum number of evaluations has been reached";
      break;

    case MaximumNumberOfPixelEvaluationsTermination:
      stopcondition = "Maximum number of pixel evaluations has been reached";
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
  log::info(std::ostringstream{}  << "Stopping condition: " << stopcondition << ".\n");
}

template <class TElastix>
void
GOMEA<TElastix>::AfterRegistration(void)
{
  log::info(std::ostringstream{}  << "\nFinal metric value = " << this->m_Value << "\n");
}

template <class TElastix>
void
GOMEA<TElastix>::OptimizeParametersWithGradientDescent(ParametersType & parameters, int iterations)
{
  m_ASGD->ResetCurrentTimeToInitialTime();
  m_ASGD->SetCurrentIteration(0);

  m_ASGD->SetInitialPosition(parameters);
  m_ASGD->SetNumberOfIterations(iterations);
  m_ASGD->StartOptimization();

  parameters = m_ASGD->GetCurrentPosition();
}

template <class TElastix>
void
GOMEA<TElastix>::RepairFoldsInTransformParameters(ParametersType & parameters)
{
  AdvancedMetricType * metric = dynamic_cast<AdvancedMetricType *>(this->GetCostFunction());
  metric->RepairFoldsInBsplineTransform(parameters);
}

template <class TElastix>
void
GOMEA<TElastix>::ZeroParametersOutsideMask(ParametersType & parameters)
{
  AdvancedMetricType * metric = dynamic_cast<AdvancedMetricType *>(this->GetCostFunction());
  if (metric->GetImageSampler()->GetUseMask())
  {
    const std::vector<bool> & mask = metric->GetParametersOutsideOfMask();

    for (unsigned int i = 0; i < mask.size(); ++i)
    {
      if (mask[i])
        parameters[i] = 0.0;
    }
  }
}

template <class TElastix>
void
GOMEA<TElastix>::WriteMutualInformationMatrix() const
{
  std::ostringstream makeFileName("");
  makeFileName << this->m_MutualInformationOutDir << "MI_matrix_" << this->m_CurrentIteration << ".dat";
  std::string   fileName = makeFileName.str();
  std::ofstream file;
  file.open(fileName.c_str());

  this->WriteMutualInformationMatrixToFile(file);

  file.close();
}

template <class TElastix>
void
GOMEA<TElastix>::InitializeASGD()
{
  m_ASGD->SetCostFunction(this->GetCostFunction());
  m_ASGD->SetHybridMode(true);
}
} // namespace elastix

#endif

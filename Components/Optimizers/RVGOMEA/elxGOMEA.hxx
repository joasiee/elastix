#ifndef elxGOMEA_hxx
#define elxGOMEA_hxx

#include "elxGOMEA.h"

namespace elastix
{

template <class TElastix>
void
GOMEA<TElastix>::BeforeRegistration(void)
{
  unsigned int randomSeed = 0;
  if (this->GetConfiguration()->ReadParameter(randomSeed, "RandomSeed", 0, false))
    this->SetRandomSeed(randomSeed);

  this->SetImageDimension(this->GetElastix()->GetFixedImage()->GetImageDimension());

  this->AddTargetCellToIterationInfo("2:Metric");
  this->AddTargetCellToIterationInfo("3:PdPct");

  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3:PdPct") << std::showpoint << std::fixed;
}

template <class TElastix>
void
GOMEA<TElastix>::AfterEachIteration(void)
{
  /** Print some information. */
  this->GetIterationInfoAt("2:Metric") << this->m_Value;
  this->GetIterationInfoAt("3:PdPct") << boost::accumulators::mean(this->m_PdPctMean);
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

  std::ostringstream makeFileName("");
  makeFileName << this->m_Configuration->GetCommandLineArgument("-out") << "R" << level << "_dist_mults.dat";
  std::string fileName = makeFileName.str();
  this->m_DistMultOutFile.open(fileName.c_str());
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
  /** Print the best metric value */
  elxout << "\n" << "Final metric value = " << this->m_Value << "\n";
}

} // namespace elastix

#endif

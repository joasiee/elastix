#ifndef elxGOMEA_hxx
#define elxGOMEA_hxx

#include "elxGOMEA.h"

namespace elastix
{

template <class TElastix>
void
GOMEA<TElastix>::StartOptimization(void)
{
  /** Check if the entered scales are correct and != [ 1 1 1 ...] */
  this->SetUseScales(false);
  const ScalesType & scales = this->GetScales();
  if (scales.GetSize() == this->GetInitialPosition().GetSize())
  {
    ScalesType unit_scales(scales.GetSize());
    unit_scales.Fill(1.0);
    if (scales != unit_scales)
    {
      /** only then: */
      this->SetUseScales(true);
    }
  }

  /** Call the superclass */
  this->Superclass1::StartOptimization();
}

template <class TElastix>
void
GOMEA<TElastix>::BeforeRegistration(void)
{
  /** Add target cells to xout[IterationInfo.*/
  this->AddTargetCellToIterationInfo("2:Metric");

  /** Format the metric and stepsize as floats */
  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
}

template <class TElastix>
void
GOMEA<TElastix>::AfterEachIteration(void)
{
  /** Print some information. */
  this->GetIterationInfoAt("2:Metric") << this->GetCurrentValue();

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

  /** Set MaximumNumberOfIterations.*/
  int maximumNumberOfEvaluations = 1000000;
  this->m_Configuration->ReadParameter(
    maximumNumberOfEvaluations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetMaximumNumberOfIterations(maximumNumberOfEvaluations);

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
  this->SetBasePopulationSize(maxNumberOfPopulations);

  /** Set Tau*/
  double tau = 0.35;
  this->m_Configuration->ReadParameter(tau, "Tau", this->GetComponentLabel(), level, 0);
  this->SetBasePopulationSize(tau);

  /** Set DistributionMultiplierDecrease*/
  double distributionMultiplierDecrease = 0.9;
  this->m_Configuration->ReadParameter(
    distributionMultiplierDecrease, "DistributionMultiplierDecrease", this->GetComponentLabel(), level, 0);
  this->SetBasePopulationSize(distributionMultiplierDecrease);

  /** Set StDevThreshold*/
  double stDevThreshold = 1.0;
  this->m_Configuration->ReadParameter(stDevThreshold, "StDevThreshold", this->GetComponentLabel(), level, 0);
  this->SetBasePopulationSize(stDevThreshold);

  /** Set FitnessVarianceTolerance*/
  double fitnessVarianceTolerance = 0.35;
  this->m_Configuration->ReadParameter(
    fitnessVarianceTolerance, "FitnessVarianceTolerance", this->GetComponentLabel(), level, 0);
  this->SetBasePopulationSize(fitnessVarianceTolerance);
}

template <class TElastix>
void
GOMEA<TElastix>::AfterEachResolution(void)
{
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
  elxout << "Stopping condition: " << stopcondition << "." << std::endl;
}

template <class TElastix>
void
GOMEA<TElastix>::AfterRegistration(void)
{
  /** Print the best metric value */
  double bestValue = this->GetCurrentValue();
  elxout << std::endl << "Final metric value  = " << bestValue << std::endl;
}

} // namespace elastix

#endif
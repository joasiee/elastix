/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef itkGOMEAOptimizer_h
#define itkGOMEAOptimizer_h

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "itkSingleValuedNonLinearOptimizer.h"
#include "gomea/Tools.h"
#include "gomea/FOS.h"

namespace itk
{

class ITKOptimizers_EXPORT GOMEAOptimizer : public SingleValuedNonLinearOptimizer
{
public:
  using Self = GOMEAOptimizer;
  using Superclass = SingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkTypeMacro(GOMEAOptimizer, SingleValuedNonLinearOptimizer);

  using Superclass::ParametersType;
  using Superclass::CostFunctionType;
  using Superclass::MeasureType;

  typedef enum
  {
    MaximumNumberOfEvaluationsTermination,
    MaximumNumberOfIterationsTermination,
    AverageFitnessTermination,
    FitnessVarianceTermination,
    DistributionMultiplierTermination,
    Unknown
  } StopConditionType;

  void
  StartOptimization() override;
  void
  ResumeOptimization();
  void
  StopOptimization();

  itkGetConstMacro(CurrentIteration, int);
  itkGetConstMacro(NumberOfEvaluations, int);
  itkGetConstMacro(CurrentValue, MeasureType);
  itkGetConstReferenceMacro(StopCondition, StopConditionType);

  itkGetConstMacro(MaximumNumberOfIterations, int);
  itkSetClampMacro(MaximumNumberOfIterations, int, 1, NumericTraits<int>::max());

  itkGetConstMacro(DistributionMultiplierDecrease, double);
  itkSetMacro(DistributionMultiplierDecrease, double);

  itkGetConstMacro(StDevThreshold, double);
  itkSetMacro(StDevThreshold, double);

  itkGetConstMacro(Tau, double);
  itkSetMacro(Tau, double);

  itkGetConstMacro(FitnessVarianceTolerance, double);
  itkSetMacro(FitnessVarianceTolerance, double);

  itkGetConstMacro(ImageDimension, int);
  itkSetMacro(ImageDimension, int);

  itkGetConstMacro(BasePopulationSize, int);
  itkSetMacro(BasePopulationSize, int);

  itkGetConstMacro(MaxNumberOfPopulations, int);
  itkSetMacro(MaxNumberOfPopulations, int);

  itkGetConstMacro(MaxNumberOfEvaluations, int);
  itkSetMacro(MaxNumberOfEvaluations, int);

  itkGetConstMacro(MaxNoImprovementStretch, int);
  itkSetMacro(MaxNoImprovementStretch, int);

  itkGetConstMacro(FosElementSize, int);
  itkSetMacro(FosElementSize, int);

  itkGetConstMacro(PartialEvaluations, bool);
  itkSetMacro(PartialEvaluations, bool);

  itkGetConstMacro(WriteOutput, bool);
  itkSetMacro(WriteOutput, bool);

  const std::string
  GetStopConditionDescription() const override;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  void
  PrintSettings(std::ostream & os, Indent indent) const;

  void
  PrintProgress(std::ostream & os, Indent indent, bool concise = true) const;

protected:
  GOMEAOptimizer();
  ~GOMEAOptimizer() override = default;

  void
  ezilaitini(void);

  int               m_NumberOfEvaluations{ 0 };
  unsigned long     m_NumberOfSubfunctionEvaluations{ 0L };
  int               m_CurrentIteration{ 0 };
  StopConditionType m_StopCondition{ Unknown };
  MeasureType       m_CurrentValue{ NumericTraits<MeasureType>::max() };
  unsigned int      m_NrOfParameters;
  int               m_ImageDimension;

private:
  void
  run(void);
  int *
  mergeSortFitness(double * objectives, int number_of_solutions);
  void
  mergeSortFitnessWithinBounds(double * objectives, int * sorted, int * tosort, int p, int q);
  void
  mergeSortFitnessMerge(double * objectives, int * sorted, int * tosort, int p, int r, int q);
  void
  checkOptions(void);
  void
  initialize(void);
  void
  initializeMemory(void);
  void
  initializeNewPopulation(void);
  void
  initializeNewPopulationMemory(int population_index);
  void
  initializeFOS(int population_index);
  void
  initializeDistributionMultipliers(int population_index);
  void
  initializePopulationAndFitnessValues(int population_index);
  void
  inheritDistributionMultipliers(GOMEA::FOS * new_FOS, GOMEA::FOS * prev_FOS, double * multipliers);
  GOMEA::FOS *
  learnLinkageTreeRVGOMEA(int population_index);
  void
  computeRanksForAllPopulations(void);
  void
  computeRanksForOnePopulation(int population_index);
  short
  checkTerminationCondition(void);
  short
  checkSubgenerationTerminationConditions(void);
  short
  checkTimeLimitTerminationCondition(void);
  short
  checkNumberOfEvaluationsTerminationCondition(void);
  short
  checkNumberOfIterationsTerminationCondition(void);
  void
  checkAverageFitnessTerminationCondition(void);
  void
  determineBestSolutionInCurrentPopulations(int * population_of_best, int * index_of_best);
  void
  checkFitnessVarianceTermination(void);
  short
  checkFitnessVarianceTerminationSinglePopulation(int population_index);
  void
  checkDistributionMultiplierTerminationCondition(void);
  void
  makeSelections(void);
  void
  makeSelectionsForOnePopulation(int population_index);
  void
  makeSelectionsForOnePopulationUsingDiversityOnRank0(int population_index);
  void
  estimateParameters(int population_index);
  void
  estimateMeanVectorML(int population_index);
  void
  estimateFullCovarianceMatrixML(int population_index);
  void
  estimateParametersML(int population_index);
  void
  estimateCovarianceMatricesML(int population_index);
  void
  initializeCovarianceMatrices(int population_index);
  void
  copyBestSolutionsToAllPopulations(void);
  void
  copyBestSolutionsToPopulation(int population_index);
  void
  getBestInPopulation(int population_index, int * individual_index);
  void
  getOverallBest(int * population_index, int * individual_index);
  void
  evaluateCompletePopulation(int population_index);
  void
  costFunctionEvaluation(ParametersType * parameters, MeasureType * obj_val);
  void
  costFunctionEvaluation(ParametersType * parameters,
                         MeasureType *    obj_val,
                         MeasureType      obj_val_previous,
                         MeasureType      obj_val_previous_partial,
                         int              setIndex);
  void
  applyDistributionMultipliersToAllPopulations(void);
  void
  applyDistributionMultipliers(int population_index);
  void
  generateAndEvaluateNewSolutionsToFillAllPopulations(void);
  void
  generateAndEvaluateNewSolutionsToFillPopulation(int population_index);
  void
  computeParametersForSampling(int population_index);
  short
  generateNewSolutionFromFOSElement(int population_index, int FOS_index, int individual_index, short apply_AMS);
  short
  applyAMS(int population_index, int individual_index);
  void
  applyForcedImprovements(int population_index, int individual_index, int donor_index);
  double *
  generateNewPartialSolutionFromFOSElement(int population_index, int FOS_index);
  short
  adaptDistributionMultipliers(int population_index, int FOS_index);
  short
  generationalImprovementForOnePopulationForFOSElement(int population_index, int FOS_index, double * st_dev_ratio);
  double
  getStDevRatioForFOSElement(int population_index, double * parameters, int FOS_index);
  void
  ezilaitiniMemory(void);
  void
  ezilaitiniDistributionMultipliers(int population_index);
  void
  ezilaitiniParametersForSampling(int population_index);
  void
  ezilaitiniParametersAllPopulations(void);
  void
  ezilaitiniCovarianceMatrices(int population_index);
  void
  generationalStepAllPopulationsRecursiveFold(int population_index_smallest, int population_index_biggest);
  void
  makePopulation(int population_index);
  void
  generationalStepAllPopulations();
  void
  runAllPopulations();
  void
  IterationWriteOutput();

  mutable std::ostringstream m_StopConditionDescription;

  template <typename T>
  using Vector1D = std::vector<T>;
  template <typename T>
  using Vector2D = std::vector<std::vector<T>>;

  Vector1D<ParametersType> mean_vectors;
  Vector1D<ParametersType> mean_shift_vector;
  Vector2D<MeasureType>    objective_values;
  Vector2D<MeasureType>    objective_values_selections;
  Vector2D<ParametersType> populations;
  Vector2D<ParametersType> selections;

  double m_Tau{ 0.35 };
  double m_DistributionMultiplierDecrease{ 0.9 };
  double m_StDevThreshold{ 1.0 };
  double m_FitnessVarianceTolerance{ 0.0 };
  double distribution_multiplier_increase;
  double eta_ams{ 1.0 };
  double eta_cov{ 1.0 };

  double **   ranks;
  double **   distribution_multipliers;
  double ***  full_covariance_matrix;
  double **** decomposed_covariance_matrices;
  double **** decomposed_cholesky_factors_lower_triangle;

  int m_MaxNumberOfPopulations{ 1 };
  int m_BasePopulationSize{ 0 };
  int m_MaximumNumberOfIterations{ 100 };
  int m_MaxNumberOfEvaluations{ 1000000 };
  int m_MaxNoImprovementStretch{ 0 };
  int m_FosElementSize{ -1 };
  int m_MovingImageBufferMisses{ 0 };
  int number_of_subgenerations_per_population_factor{ 8 };
  int number_of_populations{ 0 };

  bool m_PartialEvaluations{ false };
  bool m_WriteOutput{ false };

  short * populations_terminated;
  int *   selection_sizes;
  int *   no_improvement_stretch;
  int *   number_of_generations;
  int *   population_sizes;
  int **  samples_drawn_from_normal;
  int **  out_of_bounds_draws;
  int **  individual_NIS;

  std::ofstream outFile;

  GOMEA::FOS ** linkage_model;
};
} // namespace itk

#endif

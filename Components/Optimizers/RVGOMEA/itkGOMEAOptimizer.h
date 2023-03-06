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
#include <numeric>
#include <random>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>
#include "itkSingleValuedNonLinearOptimizer.h"
#include "./util/Tools.h"
#include "./util/FOS.h"
#include "Instrumentor.hpp"
#include "itkArray.h"
#include "itkArray2D.h"
#include "itkMatrix.h"
#include "xoutmain.h"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
using namespace boost::accumulators;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::LLT;

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
    MaximumNumberOfPixelEvaluationsTermination,
    MaximumNumberOfIterationsTermination,
    AverageFitnessTermination,
    FitnessVarianceTermination,
    DistributionMultiplierTermination,
    Unknown
  } StopConditionType;

  typedef enum
  {
    Univariate = 1,
    MarginalControlPoints = -6,
    MarginalRegions = -7,
    Full = -1,
    LinkageTree = -2,
    StaticLinkageTree = -3,
    StaticBoundedLinkageTree = -4,
    StaticBoundedRandomLinkageTree = -5
  } FOSType;

  typedef enum
  {
    Random = 0,
    BestN = 1
  } RedistributionMethod;

  typedef enum
  {
    Static = 0,
    Logarithmic = 1
  } ASGDIterationSchedule;

  void
  StartOptimization() override;
  void
  ResumeOptimization();
  void
  StopOptimization();

  itkStaticConstMacro(NumberOfGenerationsPerReevaluation, unsigned int, 50);

  itkGetConstMacro(CurrentIteration, unsigned long);
  itkGetConstMacro(NumberOfEvaluations, unsigned long);
  itkGetConstReferenceMacro(StopCondition, StopConditionType);

  itkGetConstMacro(MaximumNumberOfIterations, unsigned long);
  itkSetClampMacro(MaximumNumberOfIterations, unsigned long, 0, NumericTraits<unsigned long>::max());

  itkGetConstMacro(DistributionMultiplierDecrease, double);
  itkSetMacro(DistributionMultiplierDecrease, double);

  itkGetConstMacro(StDevThreshold, double);
  itkSetMacro(StDevThreshold, double);

  itkGetConstMacro(Tau, double);
  itkSetMacro(Tau, double);

  itkGetConstMacro(FitnessVarianceTolerance, double);
  itkSetMacro(FitnessVarianceTolerance, double);

  itkGetConstMacro(TauASGD, double);
  itkSetMacro(TauASGD, double);

  itkGetConstMacro(AlphaASGD, double);
  itkSetMacro(AlphaASGD, double);

  itkGetConstMacro(BetaASGD, double);
  itkSetMacro(BetaASGD, double);

  itkGetConstMacro(ImageDimension, int);
  itkSetMacro(ImageDimension, int);

  itkGetConstMacro(BasePopulationSize, int);
  itkSetMacro(BasePopulationSize, int);

  itkGetConstMacro(MaxNumberOfPopulations, int);
  itkSetMacro(MaxNumberOfPopulations, int);

  itkGetConstMacro(MaxNumberOfEvaluations, unsigned long);
  itkSetClampMacro(MaxNumberOfEvaluations, unsigned long, 0, NumericTraits<unsigned long>::max());

  itkGetConstMacro(MaxNumberOfPixelEvaluations, unsigned long);
  itkSetClampMacro(MaxNumberOfPixelEvaluations, unsigned long, 0, NumericTraits<unsigned long>::max());

  itkGetConstMacro(MaxNoImprovementStretch, int);
  itkSetMacro(MaxNoImprovementStretch, int);

  itkGetConstMacro(FosElementSize, int);
  itkSetMacro(FosElementSize, int);

  itkGetConstMacro(StaticLinkageType, int);
  itkSetMacro(StaticLinkageType, int);

  itkGetConstMacro(StaticLinkageMinSetSize, int);
  itkSetMacro(StaticLinkageMinSetSize, int);

  itkGetConstMacro(StaticLinkageMaxSetSize, int);
  itkSetMacro(StaticLinkageMaxSetSize, int);

  itkGetConstMacro(NumberOfASGDIterations, int);
  itkSetMacro(NumberOfASGDIterations, int);

  itkGetConstMacro(MaxNumberOfASGDIterations, int);
  itkSetMacro(MaxNumberOfASGDIterations, int);

  itkGetConstMacro(MinNumberOfASGDIterations, int);
  itkSetMacro(MinNumberOfASGDIterations, int);

  itkGetConstMacro(NumberOfASGDIterationsOffset, int);
  itkSetMacro(NumberOfASGDIterationsOffset, int);

  itkSetMacro(RedistributionMethod, RedistributionMethod);
  itkSetMacro(ASGDIterationSchedule, ASGDIterationSchedule);

  itkGetConstMacro(PartialEvaluations, bool);
  itkSetMacro(PartialEvaluations, bool);

  itkGetConstMacro(UseShrinkage, bool);
  itkSetMacro(UseShrinkage, bool);

  itkGetConstMacro(UseConstraints, bool);
  itkSetMacro(UseConstraints, bool);

  itkGetConstMacro(UseASGD, bool);
  itkSetMacro(UseASGD, bool);

  itkGetConstMacro(WriteExtraOutput, bool);
  itkSetMacro(WriteExtraOutput, bool);

  const std::string
  GetStopConditionDescription() const override;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  void
  PrintSettings() const;

  void
  WriteDistributionMultipliers(std::ofstream & outfile) const;

  void
  WriteTransformParametersWithConstraint(std::ofstream & outfile);

  void
  WriteMutualInformationMatrixToFile(std::ofstream & outfile) const;

  double
  GetAverageDistributionMultiplier() const;

protected:
  GOMEAOptimizer();
  ~GOMEAOptimizer() override = default;

  void
  SetRandomSeed(long unsigned int seed)
  {
    GOMEA::random_seed = seed;
  }

  virtual size_t
  GetNumberOfPixelEvaluations() const
  {
    return m_NumberOfEvaluations;
  }

  virtual void
  InitializeASGD()
  {}

  virtual void
  OptimizeParametersWithGradientDescent(ParametersType & params, int iterations)
  {}

  virtual void
  RepairFoldsInTransformParameters(ParametersType & params)
  {}

  virtual void
  ZeroParametersOutsideMask(ParametersType & params)
  {}

  unsigned long     m_NumberOfEvaluations{ 0L };
  unsigned long     m_CurrentIteration{ 0L };
  StopConditionType m_StopCondition{ Unknown };
  unsigned int      m_NrOfParameters;
  unsigned int      m_ImageDimension;
  unsigned int      m_CurrentResolution{ 0U };
  bool              m_SubSampling{ false };

  MeasureType m_ConstraintValue{ 0.0 };

  typedef accumulator_set<float, stats<tag::mean>> MeanAccumulator;
  mutable MeanAccumulator                          m_PdPctMean;

  std::vector<int> m_BSplineGridRegionDimensions;

  std::string m_outFolderProfiling;

private:
  void
  run(void);
  int *
  mergeSortFitness(double * objectives, double * constraints, int number_of_solutions);
  void
  mergeSortFitnessWithinBounds(double * objectives, double * constraints, int * sorted, int * tosort, int p, int q);
  void
  mergeSortFitnessMerge(double * objectives, double * constraints, int * sorted, int * tosort, int p, int r, int q);
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
  checkNumberOfPixelEvaluationTerminationCondition(void);
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
  evaluatePopulation(int population);
  void
  costFunctionEvaluation(const ParametersType & parameters,
                         int                    individual_index,
                         MeasureType &          obj_val,
                         MeasureType &          constraint_val);
  void
  costFunctionEvaluation(int           population_index,
                         int           individual_index,
                         int           fos_index,
                         MeasureType & obj_val,
                         MeasureType & constraint_val);
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
  applyASGD(int population_index);
  void
  applyForcedImprovements(int population_index, int individual_index, int donor_index);
  void
  generateNewPartialSolutionFromFOSElement(int population_index, int FOS_index, VectorXd & result);
  short
  adaptDistributionMultipliers(int population_index, int FOS_index);
  short
  generationalImprovementForOnePopulationForFOSElement(int population_index, int FOS_index, double * st_dev_ratio);
  double
  getStDevRatioForFOSElement(int population_index, double * parameters, int FOS_index);
  void
  ezilaitiniMemory(void);
  void
  generationalStepAllPopulationsRecursiveFold(int population_index_smallest, int population_index_biggest);
  void
  makePopulation(int population_index);
  void
  generationalStepAllPopulations();
  void
  runAllPopulations();
  void
  ezilaitini(void);
  void
  UpdatePosition();
  void
  saveVariancesForNextResolution();
  void
  GetValueSanityCheck(const ParametersType & parameters,
                      MeasureType            obj_val_partial,
                      MeasureType            cons_val_partial) const;
  int
  NumberOfASGDIterationsLog(int iteration_nr)
  {
    if (iteration_nr < m_NumberOfASGDIterationsOffset)
      return 0;
    
    double max_its = m_MaxNumberOfASGDIterations;
    double min_its_delta = max_its - (double) m_MinNumberOfASGDIterations;
    double its_offset = m_NumberOfASGDIterationsOffset;
    
    double alpha_x = m_AlphaASGD * ((double) iteration_nr - its_offset);
    double denom = 1.0 - alpha_x / log(m_BetaASGD);

    return static_cast<int>(max_its - min_its_delta / denom);
  }

  mutable std::ostringstream m_StopConditionDescription;

  unsigned long m_MaximumNumberOfIterations{ 1000UL };
  unsigned long m_MaxNumberOfEvaluations{ 0UL };
  unsigned long m_MaxNumberOfPixelEvaluations{ 0UL };

  double m_Tau{ 0.35 };
  double m_DistributionMultiplierDecrease{ 0.9 };
  double m_StDevThreshold{ 1.0 };
  double m_FitnessVarianceTolerance{ 0.0 };
  double m_TauASGD{ 0.1 };
  double m_AlphaASGD{ 0.08 };
  double m_BetaASGD{ 0.1 };
  double distribution_multiplier_increase;
  double eta_ams{ 1.0 };
  double eta_cov{ 1.0 };
  double max_constraint_value{ 0.0 };

  int m_MaxNumberOfPopulations{ 1 };
  int m_BasePopulationSize{ 0 };
  int m_MaxNoImprovementStretch{ 0 };
  int m_FosElementSize{ -1 };
  int m_StaticLinkageType{ 0 };
  int m_StaticLinkageMinSetSize{ 3 };
  int m_StaticLinkageMaxSetSize{ 24 };
  int m_NumberOfASGDIterations{ 50 };
  int m_MaxNumberOfASGDIterations{ 200 };
  int m_MinNumberOfASGDIterations{ 20 };
  int m_NumberOfASGDIterationsOffset{ 20 };
  int number_of_subgenerations_per_population_factor{ 8 };
  int number_of_populations{ 0 };

  RedistributionMethod m_RedistributionMethod{ RedistributionMethod::BestN };
  ASGDIterationSchedule m_ASGDIterationSchedule{ ASGDIterationSchedule::Static };

  bool m_PartialEvaluations{ false };
  bool m_UseShrinkage{ false };
  bool m_UseASGD{ false };
  bool m_UseConstraints{ true };
  bool m_WriteExtraOutput{ false };

  template <typename T>
  using Vector1D = std::vector<T>;
  template <typename T>
  using Vector2D = std::vector<std::vector<T>>;

  ParametersType variances_last_resolution;
  ParametersType max_constraint_parameters;

  Vector1D<ParametersType> mean_vectors;
  Vector1D<ParametersType> mean_shift_vector;
  Vector2D<ParametersType> populations;
  Vector2D<ParametersType> selections;

  Array<short>  populations_terminated;
  Array<int>    selection_sizes;
  Array<int>    no_improvement_stretch;
  Array<int>    number_of_generations;
  Array<int>    population_sizes;
  Vector2D<int> individual_NIS;

  Vector2D<MeasureType> objective_values;
  Vector2D<MeasureType> objective_values_selections;
  Vector2D<MeasureType> constraint_values;
  Vector2D<MeasureType> constraint_values_selections;
  Vector2D<double>      ranks;
  Vector2D<double>      distribution_multipliers;
  Vector1D<MatrixXd>    full_covariance_matrix;
  Vector2D<MatrixXd>    decomposed_covariance_matrices;
  Vector2D<MatrixXd>    decomposed_cholesky_factors_lower_triangle;

  GOMEA::FOS ** linkage_model;
};
} // namespace itk

#endif

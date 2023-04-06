#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "elxIncludes.h"
#include "itkSingleValuedNonLinearOptimizer.h"

#include "./util/Tools.h"
#include "./util/FOS.h"
#include "./util/MO_optimization.h"

namespace itk
{
class ITKOptimizers_EXPORT MOGOMEAOptimizer : public SingleValuedNonLinearOptimizer
{
public:
  using Self = MOGOMEAOptimizer;
  using Superclass = SingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkTypeMacro(MOGOMEAOptimizer, SingleValuedNonLinearOptimizer);

  using Superclass::ParametersType;
  using Superclass::CostFunctionType;
  using Superclass::MeasureType;

  typedef enum
  {
    MaximumNumberOfEvaluationsTermination,
    MaximumNumberOfPixelEvaluationsTermination,
    MaximumNumberOfGenerationsTermination,
    MaximumNumberOfSecondsTermination,
    DistributionMultiplierTermination,
    Unknown
  } StopConditionType;

  typedef enum
  {
    Univariate = 1,
    MarginalControlPoints = -6,
    Full = -1,
    LinkageTree = -2,
    StaticLinkageTree = -3,
    StaticBoundedLinkageTree = -4,
    StaticBoundedRandomLinkageTree = -5
  } FOSType;

  void
  StartOptimization() override;
  void
  ResumeOptimization();
  void
  StopOptimization();

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  void
  PrintSettings() const;

protected:
  MOGOMEAOptimizer() = default;
  ~MOGOMEAOptimizer() override;

  virtual MeasureType
  GetValueForMetric(int metric_index) const
  {
    itkExceptionMacro("GetValueForMetric() not implemented");
  }

  virtual void
  UpdateMetricIterationOutput()
  {
    itkExceptionMacro("UpdateMetricIterationOutput() not implemented");
  }

  double
  ComputeAverageDistributionMultiplier() const;

  StopConditionType m_StopCondition{ Unknown };
  std::vector<int>  m_BSplineGridRegionDimensions;
  unsigned int      m_ImageDimension;
  bool              m_PartialEvaluations;

  // Parameters
  double tau,                         /* The selection truncation percentile (in [1/population_size,1]). */
    distribution_multiplier_decrease, /* The multiplicative distribution multiplier decrease. */
    st_dev_ratio_threshold; /* The maximum ratio of the distance of the average improvement to the mean compared to the
                                 distance of one standard deviation before triggering AVS (SDR mechanism). */

  int base_number_of_mixing_components, /* The base number of mixing components. */
    base_population_size,               /* The base population size. */
    maximum_no_improvement_stretch,     /* The maximum number of subsequent generations without an improvement while the
                                             distribution multiplier is <= 1.0. */
    maximum_number_of_populations,      /* The maximum number of populations. */
    number_of_subgenerations_per_population_factor; /* The number of subgenerations per population factor. */

  size_t maximum_number_of_evaluations, /* The maximum number of evaluations. */
    maximum_number_of_generations,      /* The maximum number of generations. */
    maximum_number_of_seconds;          /* The maximum number of seconds. */

  // Options
  short use_forced_improvement; /* Use forced improvement. */

  // Defined in ./util/*.h:
  // random_seed, write_generational_statistics, write_generational_solutions,
  // number_of_populations, use_constraints, number_of_parameters;


private:
  void
  run(void);
  void
  checkOptions(void);
  void
  initialize(void);
  void
  initializeNewPopulation(void);
  void
  initializeMemory(void);
  void
  initializeNewPopulationMemory(int population_index);
  void
  initializeCovarianceMatrices(int population_index);
  void
  initializeDistributionMultipliers(int population_index);
  void
  initializePopulationAndFitnessValues(int population_index);
  void
  computeRanks(int population_index);
  void
  computeObjectiveRanges(int population_index);
  short
  checkTerminationConditionAllPopulations(void);
  short
  checkTerminationConditionOnePopulation(int population_index);
  short
  checkNumberOfEvaluationsTerminationCondition(void);
  short
  checkNumberOfGenerationsTerminationCondition(void);
  short
  checkDistributionMultiplierTerminationCondition(int population_index);
  short
  checkTimeLimitTerminationCondition(void);
  void
  makeSelection(int population_index);
  int *
  completeSelectionBasedOnDiversityInLastSelectedRank(int   population_index,
                                                      int   start_index,
                                                      int   number_to_select,
                                                      int * sorted);
  int *
  greedyScatteredSubsetSelection(double ** points,
                                 int       number_of_points,
                                 int       number_of_dimensions,
                                 int       number_to_select);
  void
  makePopulation(int population_index);
  void
  estimateParameters(int population_index);
  void
  estimateFullCovarianceMatrixML(int population_index, int cluster_index);
  void
  initializeFOS(int population_index, int cluster_index);
  MOGOMEA_UTIL::FOS *
  learnLinkageTreeRVGOMEA(int population_index, int cluster_index);
  void
  inheritDistributionMultipliers(MOGOMEA_UTIL::FOS * new_FOS, MOGOMEA_UTIL::FOS * prev_FOS, double * multipliers);
  void
  evaluateIndividual(int population_index, int individual_index, int FOS_index);
  void
  evaluateCompletePopulation(int population_index);
  void
  copyBestSolutionsToPopulation(int population_index, double ** objective_values_selection_scaled);
  void
  applyDistributionMultipliers(int population_index);
  void
  generateAndEvaluateNewSolutionsToFillPopulationAndUpdateElitistArchive(int population_index);
  short
  applyAMS(int population_index, int individual_index, int cluster_index);
  void
  applyForcedImprovements(int population_index, int individual_index, short * improved);
  void
  computeParametersForSampling(int population_index, int cluster_index);
  short
  generateNewSolutionFromFOSElement(int population_index, int cluster_index, int FOS_index, int individual_index);
  double *
  generateNewPartialSolutionFromFOSElement(int population_index, int cluster_index, int FOS_index);
  void
  adaptDistributionMultipliers(int population_index, int cluster_index, int FOS_index);
  void
  runAllPopulations();
  void
  generationalStepAllPopulations();
  void
  generationalStepAllPopulationsRecursiveFold(int population_index_smallest, int population_index_biggest);
  short
  generationalImprovementForOneClusterForFOSElement(int      population_index,
                                                    int      cluster_index,
                                                    int      FOS_index,
                                                    double * st_dev_ratio);
  double
  getStDevRatioForOneClusterForFOSElement(int population_index, int cluster_index, int FOS_index, double * parameters);
  short
  solutionWasImprovedByFOSElement(int population_index, int cluster_index, int FOS_index, int individual_index);
  void
  ezilaitini(void);
  void
  ezilaitiniMemory(void);
  void
  ezilaitiniMemoryOnePopulation(int population_index);
  void
  ezilaitiniDistributionMultipliers(int population_index);
  void
  ezilaitiniCovarianceMatrices(int population_index);
  void
  ezilaitiniParametersForSampling(int population_index);

  ParametersType previous_params;

  int **cluster_index_for_population, *selection_sizes, /* The size of the selection. */
    *cluster_sizes,                                     /* The size of the clusters. */
    number_of_cluster_failures,
    **selection_indices, /* Indices of corresponding individuals in population for all selected individuals. */
    ***selection_indices_of_cluster_members,          /* The indices pertaining to the selection of cluster members. */
    ***selection_indices_of_cluster_members_previous, /* The indices pertaining to the selection of cluster members in
                                                         the previous generation. */
    **pop_indices_selected, **single_objective_clusters, **num_individuals_in_cluster,
    *number_of_mixing_components,                 /* The number of components in the mixture distribution. */
    number_of_nearest_neighbours_in_registration, /* The number of nearest neighbours to consider in cluster
                                                     registration */
    samples_current_cluster,                      /* The number of samples generated for the current cluster. */
    *no_improvement_stretch, /* The number of subsequent generations without an improvement while the distribution
                                multiplier is <= 1.0. */
    **number_of_elitist_solutions_copied, /* The number of solutions from the elitist archive copied to the population.
                                           */
    **sorted_ranks;
  double delta_AMS,                         /* The adaptation length for AMS (anticipated mean shift). */
    **objective_ranges,                     /* Ranges of objectives observed in the current population. */
    ***objective_values_selection_previous, /* Objective values of selected solutions in the previous generation,
                                               required for cluster registration. */
    **ranks_selection,                      /* Ranks of the selected solutions. */
    ***distribution_multipliers,            /* Distribution multipliers (AVS mechanism) */
    distribution_multiplier_increase,       /* The multiplicative distribution multiplier increase. */
    ***mean_vectors,                        /* The mean vectors, one for each population. */
    ***mean_vectors_previous,               /* The mean vectors of the previous generation, one for each population. */
    ***objective_means_scaled, /* The means of the clusters in the objective space, linearly scaled according to the
                                  observed ranges. */
    *****decomposed_covariance_matrices,             /* The covariance matrices to be used for the sampling. */
    *****decomposed_cholesky_factors_lower_triangle, /* The unique lower triangular matrix of the Cholesky factorization
                                                        for every FOS element. */
    ****full_covariance_matrix;
  clock_t               start, end;
  MOGOMEA_UTIL::FOS *** linkage_model;

  bool is_initialized{ false };
  /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
};


} // namespace itk

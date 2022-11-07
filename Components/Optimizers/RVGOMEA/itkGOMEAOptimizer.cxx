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

#include "itkGOMEAOptimizer.h"

namespace itk
{
using namespace GOMEA;

GOMEAOptimizer::GOMEAOptimizer()
{
  itkDebugMacro("Constructor");
}

void
GOMEAOptimizer::PrintSelf(std::ostream & os, Indent indent) const
{
  os << indent << this->GetNameOfClass() << ":\n";
}

void
GOMEAOptimizer::PrintSettings() const
{
  Indent indent = *itk::Indent::New();
  std::cout << indent << "Settings: \n";
  std::cout << indent.GetNextIndent() << "FOS setting: " << m_FosElementSize << "\n";
  std::cout << indent.GetNextIndent() << "Nr. of parameters: " << m_NrOfParameters << "\n";
  std::cout << indent.GetNextIndent() << "Tau: " << m_Tau << "\n";
  std::cout << indent.GetNextIndent() << "Number of populations: " << m_MaxNumberOfPopulations << "\n";
  std::cout << indent.GetNextIndent() << "Population size: " << m_BasePopulationSize << "\n";
}

void
GOMEAOptimizer::StartOptimization()
{
  PROFILE_BEGIN_SESSION("Startup", m_outFolderProfiling + "startup");
  PROFILE_FUNCTION();
  itkDebugMacro("StartOptimization");

  this->m_NrOfParameters = this->GetCostFunction()->GetNumberOfParameters();
  this->SetCurrentPosition(this->GetInitialPosition());
  this->m_CurrentIteration = 0;
  this->m_Value = NumericTraits<MeasureType>::max();
  this->m_NumberOfEvaluations = 0L;
  this->m_StopCondition = Unknown;
  this->number_of_populations = 0;
  this->initialize();
  this->ResumeOptimization();
}

void
GOMEAOptimizer::ResumeOptimization()
{
  PROFILE_FUNCTION();
  itkDebugMacro("ResumeOptimization");
  InvokeEvent(StartEvent());
  this->run();
}

void
GOMEAOptimizer::StopOptimization()
{
  itkDebugMacro("StopOptimization");
  this->saveVariancesForNextResolution();
  InvokeEvent(EndEvent());
}

void
GOMEAOptimizer::initialize(void)
{
  PROFILE_FUNCTION();

  if (m_BasePopulationSize == 0)
  {
    if (m_MaxNumberOfPopulations == 1)
    {
      switch (m_FosElementSize)
      {
        case FOSType::Full:
          m_BasePopulationSize = m_NrOfParameters * 10;
          break;
        case FOSType::StaticLinkageTree:
          m_BasePopulationSize = m_StaticLinkageMaxSetSize * 10;
          break;
        default:
          m_BasePopulationSize = static_cast<int>(10 + 12.18 * log2((double)m_NrOfParameters));
          break;
      }
    }
    else
      m_BasePopulationSize = 10;
  }

  if (m_MaxNoImprovementStretch == 0L)
    m_MaxNoImprovementStretch = 25 + m_NrOfParameters;

  // FOS init
  use_univariate_FOS = 0;
  learn_linkage_tree = 0;
  static_linkage_tree = 0;
  random_linkage_tree = 0;
  bspline_marginal_cp = 0;
  GOMEA::number_of_parameters = m_NrOfParameters;
  FOS_element_ub = m_NrOfParameters;

  if (m_FosElementSize == Full)
    m_FosElementSize = m_NrOfParameters;
  if (m_FosElementSize == LinkageTree)
    learn_linkage_tree = 1;
  if (m_FosElementSize == StaticLinkageTree)
    static_linkage_tree = 1;
  if (m_FosElementSize == StaticBoundedLinkageTree)
  {
    static_linkage_tree = 1;
    FOS_element_ub = 100;
  }
  if (m_FosElementSize == StaticBoundedRandomLinkageTree)
  {
    random_linkage_tree = 1;
    static_linkage_tree = 1;
    FOS_element_ub = 100;
  }
  if (m_FosElementSize == MarginalControlPoints)
    bspline_marginal_cp = 1;
  if (m_FosElementSize == Univariate)
    use_univariate_FOS = 1;

  GOMEA::FOS_element_size = m_FosElementSize;
  GOMEA::grid_region_dimensions = std::move(m_BSplineGridRegionDimensions);
  GOMEA::static_linkage_type = static_cast<BSplineStaticLinkageType>(m_StaticLinkageType);

  // finish initialization
  this->checkOptions();
  initializeRandomNumberGenerator();
  this->initializeMemory();
  this->Modified();
}

/**
 * Sorts an array of objectives and constraints
 * using constraint domination and returns the
 * sort-order (small to large).
 */
int *
GOMEAOptimizer::mergeSortFitness(double * objectives, double * constraints, int number_of_solutions)
{
  int i, *sorted, *tosort;

  sorted = (int *)Malloc(number_of_solutions * sizeof(int));
  tosort = (int *)Malloc(number_of_solutions * sizeof(int));
  for (i = 0; i < number_of_solutions; i++)
    tosort[i] = i;

  if (number_of_solutions == 1)
    sorted[0] = 0;
  else
    mergeSortFitnessWithinBounds(objectives, constraints, sorted, tosort, 0, number_of_solutions - 1);

  free(tosort);

  return (sorted);
}

/**
 * Subroutine of merge sort, sorts the part of the objectives and
 * constraints arrays between p and q.
 */
void
GOMEAOptimizer::mergeSortFitnessWithinBounds(double * objectives,
                                             double * constraints,
                                             int *    sorted,
                                             int *    tosort,
                                             int      p,
                                             int      q)
{
  int r;

  if (p < q)
  {
    r = (p + q) / 2;
    mergeSortFitnessWithinBounds(objectives, constraints, sorted, tosort, p, r);
    mergeSortFitnessWithinBounds(objectives, constraints, sorted, tosort, r + 1, q);
    mergeSortFitnessMerge(objectives, constraints, sorted, tosort, p, r + 1, q);
  }
}

/**
 * Subroutine of merge sort, merges the results of two sorted parts.
 */
void
GOMEAOptimizer::mergeSortFitnessMerge(double * objectives,
                                      double * constraints,
                                      int *    sorted,
                                      int *    tosort,
                                      int      p,
                                      int      r,
                                      int      q)
{
  int i, j, k, first;

  i = p;
  j = r;
  for (k = p; k <= q; k++)
  {
    first = 0;
    if (j <= q)
    {
      if (i < r)
      {
        if (betterFitness(objectives[tosort[i]], constraints[tosort[i]], objectives[tosort[j]], constraints[tosort[j]]))
          first = 1;
      }
    }
    else
      first = 1;

    if (first)
    {
      sorted[k] = tosort[i];
      i++;
    }
    else
    {
      sorted[k] = tosort[j];
      j++;
    }
  }

  for (k = p; k <= q; k++)
    tosort[k] = sorted[k];
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/**
 * Checks whether the selected options are feasible.
 */
void
GOMEAOptimizer::checkOptions(void)
{
  if (m_NrOfParameters < 1)
  {
    printf("\n");
    printf("Error: number of parameters < 1 (read: %d). Require number of parameters >= 1.", m_NrOfParameters);
    printf("\n\n");

    exit(0);
  }

  if (((int)(m_Tau * m_BasePopulationSize)) <= 0 || m_Tau >= 1)
  {
    printf("\n");
    printf("Error: tau not in range (read: %e). Require tau in [1/pop,1] (read: [%e,%e]).",
           m_Tau,
           1.0 / ((double)m_BasePopulationSize),
           1.0);
    printf("\n\n");

    exit(0);
  }

  if (m_BasePopulationSize < 1)
  {
    printf("\n");
    printf("Error: population size < 1 (read: %d). Require population size >= 1.", m_BasePopulationSize);
    printf("\n\n");

    exit(0);
  }

  if (m_MaxNumberOfPopulations < 1)
  {
    printf("\n");
    printf("Error: number of populations < 1 (read: %d). Require number of populations >= 1.", number_of_populations);
    printf("\n\n");

    exit(0);
  }

  if (FOS_element_size > 1 && (unsigned)FOS_element_size > m_NrOfParameters)
  {
    printf("\n");
    printf("Error: invalid FOS element size (read %d). Must be <= %d.", FOS_element_size, m_NrOfParameters);
    printf("\n\n");

    exit(0);
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Initialize -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Initializes the memory.
 */
void
GOMEAOptimizer::initializeMemory(void)
{
  PROFILE_FUNCTION();
  mean_vectors = Vector1D<ParametersType>(m_MaxNumberOfPopulations, ParametersType(m_NrOfParameters, 0.0));
  mean_shift_vector = Vector1D<ParametersType>(m_MaxNumberOfPopulations, ParametersType(m_NrOfParameters, 0.0));
  full_covariance_matrix =
    Vector1D<MatrixXd>(m_MaxNumberOfPopulations, MatrixXd::Zero(m_NrOfParameters, m_NrOfParameters));

  populations = Vector2D<ParametersType>(m_MaxNumberOfPopulations);
  selections = Vector2D<ParametersType>(m_MaxNumberOfPopulations);
  decomposed_cholesky_factors_lower_triangle = Vector2D<MatrixXd>(m_MaxNumberOfPopulations);
  decomposed_covariance_matrices = Vector2D<MatrixXd>(m_MaxNumberOfPopulations);
  individual_NIS = Vector2D<int>(m_MaxNumberOfPopulations);
  objective_values = Vector2D<MeasureType>(m_MaxNumberOfPopulations);
  objective_values_selections = Vector2D<MeasureType>(m_MaxNumberOfPopulations);
  constraint_values = Vector2D<MeasureType>(m_MaxNumberOfPopulations);
  constraint_values_selections = Vector2D<MeasureType>(m_MaxNumberOfPopulations);
  ranks = Vector2D<double>(m_MaxNumberOfPopulations);
  distribution_multipliers = Vector2D<double>(m_MaxNumberOfPopulations);

  population_sizes = Array<int>(m_MaxNumberOfPopulations);
  selection_sizes = Array<int>(m_MaxNumberOfPopulations);
  populations_terminated = Array<short>(m_MaxNumberOfPopulations);
  no_improvement_stretch = Array<int>(m_MaxNumberOfPopulations);
  number_of_generations = Array<int>(m_MaxNumberOfPopulations);

  linkage_model = (FOS **)Malloc(m_MaxNumberOfPopulations * sizeof(FOS *));
}

void
GOMEAOptimizer::initializeNewPopulationMemory(int population_index)
{
  PROFILE_FUNCTION();
  int i;

  if (population_index == 0)
    population_sizes[population_index] = m_BasePopulationSize;
  else
    population_sizes[population_index] = 2 * population_sizes[population_index - 1];

  selection_sizes[population_index] = (double)(m_Tau * population_sizes[population_index]);

  ParametersType zeroParam(m_NrOfParameters, 0.0);
  populations[population_index] = Vector1D<ParametersType>(population_sizes[population_index], zeroParam);
  selections[population_index] = Vector1D<ParametersType>(selection_sizes[population_index], zeroParam);

  objective_values[population_index] = Vector1D<MeasureType>(population_sizes[population_index], 0.0);
  constraint_values[population_index] = Vector1D<MeasureType>(population_sizes[population_index], 0.0);
  objective_values_selections[population_index] = Vector1D<MeasureType>(selection_sizes[population_index], 0.0);
  constraint_values_selections[population_index] = Vector1D<MeasureType>(selection_sizes[population_index], 0.0);
  ranks[population_index] = Vector1D<double>(population_sizes[population_index], 0.0);

  individual_NIS[population_index] = Vector1D<int>(population_sizes[population_index], 0);

  if (learn_linkage_tree)
  {
    distribution_multipliers[population_index] = Vector1D<double>(1);
    linkage_model[population_index] = (FOS *)Malloc(sizeof(FOS));
    linkage_model[population_index]->length = 1;
    linkage_model[population_index]->sets = (int **)Malloc(linkage_model[population_index]->length * sizeof(int *));
    linkage_model[population_index]->set_length = (int *)Malloc(linkage_model[population_index]->length * sizeof(int));
    for (i = 0; i < linkage_model[population_index]->length; i++)
      linkage_model[population_index]->sets[i] = (int *)Malloc(1 * sizeof(int));
  }
  else
    this->initializeFOS(population_index);

  populations_terminated[population_index] = 0;
  no_improvement_stretch[population_index] = 0;
  number_of_generations[population_index] = 0;
}

void
GOMEAOptimizer::initializeNewPopulation()
{
  PROFILE_FUNCTION();
  this->initializeNewPopulationMemory(number_of_populations);

  if (m_PartialEvaluations)
    this->GetCostFunction()->InitSubfunctionSamplers(population_sizes[0]);

  this->initializePopulationAndFitnessValues(number_of_populations);

  if (!learn_linkage_tree)
  {
    this->initializeCovarianceMatrices(number_of_populations);
    this->initializeDistributionMultipliers(number_of_populations);
  }

  this->computeRanksForOnePopulation(number_of_populations);

  ++number_of_populations;
}

/**
 * Initializes the linkage tree
 */
void
GOMEAOptimizer::initializeFOS(int population_index)
{
  PROFILE_FUNCTION();
  int    i;
  FILE * file;
  FOS *  new_FOS;

  fflush(stdout);
  file = fopen("FOS.in", "r");
  if (file != NULL)
  {
    if (population_index == 0)
      new_FOS = readFOSFromFile(file);
    else
      new_FOS = copyFOS(linkage_model[0]);
  }
  else if (static_linkage_tree)
  {
    if (population_index == 0)
    {
      new_FOS = this->learnLinkageTreeRVGOMEA(population_index);
      if (static_linkage_type == BSplineStaticLinkageType::EuclideanSimilarity)
        filterFOS(new_FOS, 3, m_StaticLinkageMaxSetSize);
    }
    else
      new_FOS = copyFOS(linkage_model[0]);
  }
  else if (bspline_marginal_cp)
  {
    new_FOS = (FOS *)Malloc(sizeof(FOS));
    new_FOS->length = m_NrOfParameters / m_ImageDimension;
    new_FOS->sets = (int **)Malloc(new_FOS->length * sizeof(int *));
    new_FOS->set_length = (int *)Malloc(new_FOS->length * sizeof(int));
    for (i = 0; i < new_FOS->length; i++)
    {
      new_FOS->sets[i] = (int *)Malloc(m_ImageDimension * sizeof(int));
      new_FOS->set_length[i] = m_ImageDimension;
    }
    for (i = 0; (unsigned)i < m_NrOfParameters; i++)
    {
      new_FOS->sets[i % new_FOS->length][i / new_FOS->length] = i;
    }
  }
  else
  {
    new_FOS = (FOS *)Malloc(sizeof(FOS));
    new_FOS->length = (m_NrOfParameters + FOS_element_size - 1) / FOS_element_size;
    new_FOS->sets = (int **)Malloc(new_FOS->length * sizeof(int *));
    new_FOS->set_length = (int *)Malloc(new_FOS->length * sizeof(int));
    for (i = 0; i < new_FOS->length; i++)
    {
      new_FOS->sets[i] = (int *)Malloc(FOS_element_size * sizeof(int));
      new_FOS->set_length[i] = 0;
    }

    for (i = 0; (unsigned)i < m_NrOfParameters; i++)
    {
      new_FOS->sets[i / FOS_element_size][i % FOS_element_size] = i;
      new_FOS->set_length[i / FOS_element_size]++;
    }
  }
  linkage_model[population_index] = new_FOS;
}

/**
 * Initializes the distribution multipliers.
 */
void
GOMEAOptimizer::initializeDistributionMultipliers(int population_index)
{
  PROFILE_FUNCTION();
  distribution_multipliers[population_index] = Vector1D<double>(linkage_model[population_index]->length, 1.0);

  distribution_multiplier_increase = 1.0 / m_DistributionMultiplierDecrease;
}

/**
 * Initializes the populations and the fitness values.
 */
void
GOMEAOptimizer::initializePopulationAndFitnessValues(int population_index)
{
  PROFILE_FUNCTION();
  int j, k;

  for (j = 0; j < population_sizes[population_index]; j++)
  {
    individual_NIS[population_index][j] = 0;
    for (k = 0; (unsigned)k < m_NrOfParameters; k++)
    {
      if (m_CurrentResolution > 0 && variances_last_resolution.size() == m_NrOfParameters)
        populations[population_index][j][k] = m_CurrentPosition[k] + (j > 0) * random1DNormalUnit() * sqrt(variances_last_resolution[k]);
      else
        populations[population_index][j][k] = m_CurrentPosition[k] + (j > 0) * random1DNormalUnit();
    }
    this->costFunctionEvaluation(populations[population_index][j],
                                 j,
                                 objective_values[population_index][j],
                                 constraint_values[population_index][j]);
    this->SavePartialEvaluation(j);
  }
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

FOS *
GOMEAOptimizer::learnLinkageTreeRVGOMEA(int population_index)
{
  PROFILE_FUNCTION();
  int   i;
  FOS * new_FOS;

  new_FOS = learnLinkageTree(full_covariance_matrix[population_index]);
  if (learn_linkage_tree && number_of_generations[population_index] > 0)
  {
    this->inheritDistributionMultipliers(
      new_FOS, linkage_model[population_index], distribution_multipliers[population_index].data());
  }

  if (learn_linkage_tree)
  {
    for (i = 0; i < linkage_model[population_index]->length; i++)
      free(linkage_model[population_index]->sets[i]);
    free(linkage_model[population_index]->sets);
    free(linkage_model[population_index]->set_length);
    free(linkage_model[population_index]);
  }
  return (new_FOS);
}

void
GOMEAOptimizer::inheritDistributionMultipliers(FOS * new_FOS, FOS * prev_FOS, double * multipliers)
{
  PROFILE_FUNCTION();
  int      i, *permutation;
  double * multipliers_copy;

  multipliers_copy = (double *)Malloc(new_FOS->length * sizeof(double));
  for (i = 0; i < new_FOS->length; i++)
    multipliers_copy[i] = multipliers[i];

  permutation = matchFOSElements(new_FOS, prev_FOS);

  for (i = 0; i < new_FOS->length; i++)
    multipliers[permutation[i]] = multipliers_copy[i];

  free(multipliers_copy);
  free(permutation);
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Ranking -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Computes the ranks of the solutions in all populations.
 */
void
GOMEAOptimizer::computeRanksForAllPopulations(void)
{
  PROFILE_FUNCTION();
  int i;

  for (i = 0; i < number_of_populations; i++)
    this->computeRanksForOnePopulation(i);
}

/**
 * Computes the ranks of the solutions in one population.
 */
void
GOMEAOptimizer::computeRanksForOnePopulation(int population_index)
{
  PROFILE_FUNCTION();
  int i, *sorted, rank;

  if (!populations_terminated[population_index])
  {
    sorted = this->mergeSortFitness(objective_values[population_index].data(),
                                    constraint_values[population_index].data(),
                                    population_sizes[population_index]);

    rank = 0;
    ranks[population_index][sorted[0]] = rank;
    selections[population_index][rank] = populations[population_index][sorted[0]];
    for (i = 1; i < population_sizes[population_index]; i++)
    {
      if (objective_values[population_index][sorted[i]] != objective_values[population_index][sorted[i - 1]])
        rank++;

      ranks[population_index][sorted[i]] = rank;
    }

    free(sorted);
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Termination -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns 1 if termination should be enforced, 0 otherwise.
 */
short
GOMEAOptimizer::checkTerminationCondition(void)
{
  PROFILE_FUNCTION();
  short allTrue;
  int   i;

  if (m_CurrentIteration == 0)
    return (0);

  if (this->checkSubgenerationTerminationConditions())
    return (1);

  this->checkAverageFitnessTerminationCondition();

  this->checkFitnessVarianceTermination();

  this->checkDistributionMultiplierTerminationCondition();

  if (number_of_populations < m_MaxNumberOfPopulations)
    return (0);

  allTrue = 1;
  for (i = 0; i < number_of_populations; i++)
  {
    if (!populations_terminated[i])
    {
      allTrue = 0;
      break;
    }
  }

  return (allTrue);
}

short
GOMEAOptimizer::checkSubgenerationTerminationConditions(void)
{
  PROFILE_FUNCTION();
  if (this->checkNumberOfEvaluationsTerminationCondition())
    return (1);

  if (this->checkNumberOfPixelEvaluationTerminationCondition())
    return (1);

  if (this->checkNumberOfIterationsTerminationCondition())
    return (1);

  return (0);
}

/**
 * Returns 1 if the maximum number of evaluations
 * has been reached, 0 otherwise.
 */
short
GOMEAOptimizer::checkNumberOfEvaluationsTerminationCondition(void)
{
  PROFILE_FUNCTION();
  if (m_MaxNumberOfEvaluations > 0 && m_NumberOfEvaluations >= m_MaxNumberOfEvaluations)
  {
    this->m_StopCondition = MaximumNumberOfEvaluationsTermination;
    return (1);
  }

  return (0);
}

short
GOMEAOptimizer::checkNumberOfPixelEvaluationTerminationCondition(void)
{
  PROFILE_FUNCTION();
  if (m_MaxNumberOfPixelEvaluations > 0 && this->GetNumberOfPixelEvaluations() >= m_MaxNumberOfPixelEvaluations)
  {
    this->m_StopCondition = MaximumNumberOfPixelEvaluationsTermination;
    return (1);
  }

  return (0);
}

/**
 * Returns 1 if the maximum number of evaluations
 * has been reached, 0 otherwise.
 */
short
GOMEAOptimizer::checkNumberOfIterationsTerminationCondition(void)
{
  PROFILE_FUNCTION();
  if (m_CurrentIteration >= m_MaximumNumberOfIterations && m_MaximumNumberOfIterations > 0)
  {
    this->m_StopCondition = MaximumNumberOfIterationsTermination;
    return (1);
  }

  return (0);
}

void
GOMEAOptimizer::checkAverageFitnessTerminationCondition(void)
{
  PROFILE_FUNCTION();
  int     i, j;
  double *average_objective_values, *average_constraint_values;

  average_objective_values = (double *)Malloc(number_of_populations * sizeof(double));
  average_constraint_values = (double *)Malloc(number_of_populations * sizeof(double));
  for (i = number_of_populations - 1; i >= 0; i--)
  {
    average_objective_values[i] = 0;
    average_constraint_values[i] = 0;
    for (j = 0; j < population_sizes[i]; j++)
    {
      average_objective_values[i] += objective_values[i][j];
      average_constraint_values[i] += constraint_values[i][j];
    }
    average_objective_values[i] /= population_sizes[i];
    average_constraint_values[i] /= population_sizes[i];
    if (i < number_of_populations - 1 && betterFitness(average_objective_values[i + 1],
                                                       average_constraint_values[i + 1],
                                                       average_objective_values[i],
                                                       average_constraint_values[i]))
    {
      for (j = i; j >= 0; j--)
        populations_terminated[j] = 1;
      break;
    }
  }
  free(average_objective_values);
  free(average_constraint_values);
}

/**
 * Determines which solution is the best of all solutions
 * in all current populations.
 */
void
GOMEAOptimizer::determineBestSolutionInCurrentPopulations(int * population_of_best, int * index_of_best)
{
  PROFILE_FUNCTION();
  int i, j;

  (*population_of_best) = 0;
  (*index_of_best) = 0;
  for (i = 0; i < number_of_populations; i++)
  {
    for (j = 0; j < population_sizes[i]; j++)
    {
      if (betterFitness(objective_values[i][j],
                        constraint_values[i][j],
                        objective_values[(*population_of_best)][(*index_of_best)],
                        constraint_values[(*population_of_best)][(*index_of_best)]))
      {
        (*population_of_best) = i;
        (*index_of_best) = j;
      }
    }
  }
}

/**
 * Checks whether the fitness variance in any population
 * has become too small (user-defined tolerance).
 */
void
GOMEAOptimizer::checkFitnessVarianceTermination(void)
{
  PROFILE_FUNCTION();
  int i;

  for (i = 0; i < number_of_populations; i++)
  {
    if (!populations_terminated[i] && this->checkFitnessVarianceTerminationSinglePopulation(i))
    {
      populations_terminated[i] = 1;
      this->m_StopCondition = FitnessVarianceTermination;
    }
  }
}

/**
 * Returns 1 if the fitness variance in a specific population
 * has become too small (user-defined tolerance).
 */
short
GOMEAOptimizer::checkFitnessVarianceTerminationSinglePopulation(int population_index)
{
  PROFILE_FUNCTION();
  int    i;
  double objective_avg, objective_var;

  objective_avg = 0.0;
  for (i = 0; i < population_sizes[population_index]; i++)
    objective_avg += objective_values[population_index][i];
  objective_avg = objective_avg / ((double)population_sizes[population_index]);

  objective_var = 0.0;
  for (i = 0; i < population_sizes[population_index]; i++)
    objective_var +=
      (objective_values[population_index][i] - objective_avg) * (objective_values[population_index][i] - objective_avg);
  objective_var = objective_var / ((double)population_sizes[population_index]);

  if (objective_var <= 0.0)
    objective_var = 0.0;

  if (objective_var <= m_FitnessVarianceTolerance)
    return (1);

  return (0);
}

/**
 * Checks whether the distribution multiplier in any population
 * has become too small (1e-10).
 */
void
GOMEAOptimizer::checkDistributionMultiplierTerminationCondition(void)
{
  PROFILE_FUNCTION();
  int   i, j;
  short converged;

  for (i = 0; i < number_of_populations; i++)
  {
    if (!populations_terminated[i])
    {
      converged = 1;
      for (j = 0; j < linkage_model[i]->length; j++)
      {
        if (distribution_multipliers[i][j] > 1e-10)
        {
          converged = 0;
          break;
        }
      }

      if (converged)
      {
        populations_terminated[i] = 1;
        this->m_StopCondition = DistributionMultiplierTermination;
      }
    }
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Selection =-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Makes a set of selected solutions for each population.
 */
void
GOMEAOptimizer::makeSelections(void)
{
  PROFILE_FUNCTION();
  int i;

  for (i = 0; i < number_of_populations; i++)
    if (!populations_terminated[i])
      this->makeSelectionsForOnePopulation(i);
}

/**
 * Performs truncation selection on a single population.
 */
void
GOMEAOptimizer::makeSelectionsForOnePopulation(int population_index)
{
  PROFILE_FUNCTION();
  int i, j, *sorted;

  sorted = mergeSort(ranks[population_index].data(), population_sizes[population_index]);

  if (ranks[population_index][sorted[selection_sizes[population_index] - 1]] == 0)
  {
    this->makeSelectionsForOnePopulationUsingDiversityOnRank0(population_index);
  }
  else
  {
    for (i = 0; i < selection_sizes[population_index]; i++)
    {
      for (j = 0; (unsigned)j < m_NrOfParameters; j++)
        selections[population_index][i][j] = populations[population_index][sorted[i]][j];

      objective_values_selections[population_index][i] = objective_values[population_index][sorted[i]];
      constraint_values_selections[population_index][i] = constraint_values[population_index][sorted[i]];
    }
  }

  free(sorted);
}

/**
 * Performs selection from all solutions that have rank 0
 * based on diversity.
 */
void
GOMEAOptimizer::makeSelectionsForOnePopulationUsingDiversityOnRank0(int population_index)
{
  PROFILE_FUNCTION();
  int i, j, number_of_rank0_solutions, *preselection_indices, *selection_indices, index_of_farthest,
    number_selected_so_far;
  double *nn_distances, distance_of_farthest, value;

  number_of_rank0_solutions = 0;
  for (i = 0; i < population_sizes[population_index]; i++)
  {
    if (ranks[population_index][i] == 0)
      number_of_rank0_solutions++;
  }

  preselection_indices = (int *)Malloc(number_of_rank0_solutions * sizeof(int));
  j = 0;
  for (i = 0; i < population_sizes[population_index]; i++)
  {
    if (ranks[population_index][i] == 0)
    {
      preselection_indices[j] = i;
      j++;
    }
  }

  index_of_farthest = 0;
  distance_of_farthest = objective_values[population_index][preselection_indices[0]];
  for (i = 1; i < number_of_rank0_solutions; i++)
  {
    if (objective_values[population_index][preselection_indices[i]] > distance_of_farthest)
    {
      index_of_farthest = i;
      distance_of_farthest = objective_values[population_index][preselection_indices[i]];
    }
  }

  number_selected_so_far = 0;
  selection_indices = (int *)Malloc(selection_sizes[population_index] * sizeof(int));
  selection_indices[number_selected_so_far] = preselection_indices[index_of_farthest];
  preselection_indices[index_of_farthest] = preselection_indices[number_of_rank0_solutions - 1];
  number_of_rank0_solutions--;
  number_selected_so_far++;

  nn_distances = (double *)Malloc(number_of_rank0_solutions * sizeof(double));
  for (i = 0; i < number_of_rank0_solutions; i++)
    nn_distances[i] =
      distanceEuclidean(populations[population_index][preselection_indices[i]].data_block(),
                        populations[population_index][selection_indices[number_selected_so_far - 1]].data_block(),
                        m_NrOfParameters);

  while (number_selected_so_far < selection_sizes[population_index])
  {
    index_of_farthest = 0;
    distance_of_farthest = nn_distances[0];
    for (i = 1; i < number_of_rank0_solutions; i++)
    {
      if (nn_distances[i] > distance_of_farthest)
      {
        index_of_farthest = i;
        distance_of_farthest = nn_distances[i];
      }
    }

    selection_indices[number_selected_so_far] = preselection_indices[index_of_farthest];
    preselection_indices[index_of_farthest] = preselection_indices[number_of_rank0_solutions - 1];
    nn_distances[index_of_farthest] = nn_distances[number_of_rank0_solutions - 1];
    number_of_rank0_solutions--;
    number_selected_so_far++;

    for (i = 0; i < number_of_rank0_solutions; i++)
    {
      value =
        distanceEuclidean(populations[population_index][preselection_indices[i]].data_block(),
                          populations[population_index][selection_indices[number_selected_so_far - 1]].data_block(),
                          m_NrOfParameters);
      if (value < nn_distances[i])
        nn_distances[i] = value;
    }
  }

  for (i = 0; i < selection_sizes[population_index]; i++)
  {
    for (j = 0; (unsigned)j < m_NrOfParameters; j++)
      selections[population_index][i][j] = populations[population_index][selection_indices[i]][j];

    objective_values_selections[population_index][i] = objective_values[population_index][selection_indices[i]];
    constraint_values_selections[population_index][i] = constraint_values[population_index][selection_indices[i]];
  }

  free(nn_distances);
  free(selection_indices);
  free(preselection_indices);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Variation -==-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * First estimates the parameters of a normal distribution in the
 * parameter space from the selected sets of solutions (a separate
 * normal distribution for each population). Then copies the single
 * best selected solutions to their respective populations. Finally
 * fills up each population, after the variances have been scaled,
 * by drawing new samples from the normal distributions and applying
 * AMS to several of these new solutions. Then, the fitness ranks
 * are recomputed. Finally, the distribution multipliers are adapted
 * according to the SDR-AVS mechanism.
 */
void
GOMEAOptimizer::makePopulation(int population_index)
{
  PROFILE_FUNCTION();
  if (populations_terminated[population_index])
    return;

  this->estimateParameters(population_index);

  this->copyBestSolutionsToPopulation(population_index);

  this->applyDistributionMultipliers(population_index);

  this->generateAndEvaluateNewSolutionsToFillPopulation(population_index);

  this->computeRanksForOnePopulation(population_index);
}

/**
 * Estimates the paramaters of the multivariate
 * normal distribution for a specified population.
 */
void
GOMEAOptimizer::estimateParameters(int population_index)
{
  PROFILE_FUNCTION();
  if (!populations_terminated[population_index])
  {
    this->estimateMeanVectorML(population_index);

    if (learn_linkage_tree)
    {
      this->estimateFullCovarianceMatrixML(population_index);

      linkage_model[population_index] = learnLinkageTreeRVGOMEA(population_index);

      this->initializeCovarianceMatrices(population_index);

      if (number_of_generations[population_index] == 0)
        this->initializeDistributionMultipliers(population_index);
    }

    if (m_PartialEvaluations)
    {
      if (learn_linkage_tree || number_of_generations[population_index] == 0)
      {
        this->GetCostFunction()->InitFOSMapping(linkage_model[population_index]->sets,
                                                linkage_model[population_index]->set_length,
                                                linkage_model[population_index]->length);
      }
    }

    this->estimateParametersML(population_index);
  }
}

void
GOMEAOptimizer::estimateParametersML(int population_index)
{
  PROFILE_FUNCTION();
  int i, j;

  /* Change the focus of the search to the best solution */
  for (i = 0; i < linkage_model[population_index]->length; i++)
    if (distribution_multipliers[population_index][i] < 1.0)
      for (j = 0; j < linkage_model[population_index]->set_length[i]; j++)
        mean_vectors[population_index][linkage_model[population_index]->sets[i][j]] =
          selections[population_index][0][linkage_model[population_index]->sets[i][j]];

  this->estimateCovarianceMatricesML(population_index);
}

/**
 * Computes the sample mean for a specified population.
 */
void
GOMEAOptimizer::estimateMeanVectorML(int population_index)
{
  PROFILE_FUNCTION();
  unsigned int i, j;
  double       new_mean;

  for (i = 0; i < m_NrOfParameters; i++)
  {
    new_mean = 0.0;
    for (j = 0; j < selection_sizes[population_index]; j++)
      new_mean += selections[population_index][j][i];
    new_mean /= (double)selection_sizes[population_index];

    if (number_of_generations[population_index] > 0)
      mean_shift_vector[population_index][i] = new_mean - mean_vectors[population_index][i];

    mean_vectors[population_index][i] = new_mean;
  }
}

/**
 * Computes the matrix of sample covariances for
 * a specified population.
 *
 * It is important that the pre-condition must be satisified:
 * estimateMeanVector was called first.
 */
void
GOMEAOptimizer::estimateFullCovarianceMatrixML(int population_index)
{
  PROFILE_FUNCTION();
  int    i, j, m;
  double cov;

  full_covariance_matrix[population_index] = MatrixXd(m_NrOfParameters, m_NrOfParameters);
  /* First do the maximum-likelihood estimate from data */
  for (i = 0; (unsigned)i < m_NrOfParameters; i++)
  {
    for (j = 0; (unsigned)j < m_NrOfParameters; j++)
    {
      cov = 0.0;
      for (m = 0; m < selection_sizes[population_index]; m++)
      {
        cov += (selections[population_index][m][i] - mean_vectors[population_index][i]) *
               (selections[population_index][m][j] - mean_vectors[population_index][j]);
      }

      cov /= (double)selection_sizes[population_index];
      full_covariance_matrix[population_index](i, j) = cov;
      full_covariance_matrix[population_index](j, i) = cov;
    }
  }
}

void
GOMEAOptimizer::estimateCovarianceMatricesML(int population_index)
{
  PROFILE_FUNCTION();
  int    vara, varb, i, j, k, m;
  double cov;

  /* First do the maximum-likelihood estimate from data */
  // for each fos set:
  for (i = 0; i < linkage_model[population_index]->length; ++i)
  {
    // for each parameter index in fos set:
    for (j = 0; j < linkage_model[population_index]->set_length[i]; ++j)
    {
      vara = linkage_model[population_index]->sets[i][j];
      for (k = j; k < linkage_model[population_index]->set_length[i]; ++k)
      {
        // for each pair (vara, varb) of parameters:
        varb = linkage_model[population_index]->sets[i][k];

        if (learn_linkage_tree)
        {
          cov = full_covariance_matrix[population_index](vara, varb);
        }
        else
        {
          cov = 0.0;
          for (m = 0; m < selection_sizes[population_index]; ++m)
          {
            cov += (selections[population_index][m][vara] - mean_vectors[population_index][vara]) *
                   (selections[population_index][m][varb] - mean_vectors[population_index][varb]);
          }

          cov /= static_cast<double>(selection_sizes[population_index]);
        }
        decomposed_covariance_matrices[population_index][i](j, k) =
          (1 - eta_cov) * decomposed_covariance_matrices[population_index][i](j, k) + eta_cov * cov;
        decomposed_covariance_matrices[population_index][i](k, j) =
          decomposed_covariance_matrices[population_index][i](j, k);
      }
    }
    // if (this->m_UseShrinkage && linkage_model[population_index]->set_length[i] * 10 >
    // selection_sizes[population_index])
    //   shrunkCovarianceOAS(decomposed_covariance_matrices[population_index][i], selection_sizes[population_index]);
  }
}

void
GOMEAOptimizer::initializeCovarianceMatrices(int population_index)
{
  PROFILE_FUNCTION();
  int j;

  decomposed_covariance_matrices[population_index] = Vector1D<MatrixXd>(linkage_model[population_index]->length);
  decomposed_cholesky_factors_lower_triangle[population_index] =
    Vector1D<MatrixXd>(linkage_model[population_index]->length);

  for (j = 0; j < linkage_model[population_index]->length; j++)
  {
    decomposed_covariance_matrices[population_index][j] =
      MatrixXd(linkage_model[population_index]->set_length[j], linkage_model[population_index]->set_length[j]);
    decomposed_covariance_matrices[population_index][j].fill(1.0);

    decomposed_cholesky_factors_lower_triangle[population_index][j] =
      MatrixXd(linkage_model[population_index]->set_length[j], linkage_model[population_index]->set_length[j]);
  }
}

void
GOMEAOptimizer::copyBestSolutionsToAllPopulations(void)
{
  PROFILE_FUNCTION();
  int i;

  for (i = 0; i < number_of_populations; i++)
    this->copyBestSolutionsToPopulation(i);
}

/**
 * Copies the single very best of the selected solutions
 * to their respective populations.
 */
void
GOMEAOptimizer::copyBestSolutionsToPopulation(int population_index)
{
  PROFILE_FUNCTION();
  int k;

  if (!populations_terminated[population_index])
  {
    for (k = 0; (unsigned)k < m_NrOfParameters; k++)
      populations[population_index][0][k] = selections[population_index][0][k];

    objective_values[population_index][0] = objective_values_selections[population_index][0];
    constraint_values[population_index][0] = constraint_values_selections[population_index][0];
  }
}

void
GOMEAOptimizer::getBestInPopulation(int population_index, int * individual_index)
{
  PROFILE_FUNCTION();
  int i;

  *individual_index = 0;
  for (i = 0; i < population_sizes[population_index]; i++)
    if (betterFitness(objective_values[population_index][i],
                      constraint_values[population_index][i],
                      objective_values[population_index][*individual_index],
                      constraint_values[population_index][*individual_index]))
      *individual_index = i;
}

void
GOMEAOptimizer::getOverallBest(int * population_index, int * individual_index)
{
  PROFILE_FUNCTION();
  int i, best_individual_index;

  *population_index = 0;
  this->getBestInPopulation(0, &best_individual_index);
  *individual_index = best_individual_index;
  for (i = 0; i < number_of_populations; i++)
  {
    getBestInPopulation(i, &best_individual_index);
    if (betterFitness(objective_values[i][best_individual_index],
                      constraint_values[i][best_individual_index],
                      objective_values[*population_index][*individual_index],
                      constraint_values[*population_index][*individual_index]))
    {
      *population_index = i;
      *individual_index = best_individual_index;
    }
  }
}

void
GOMEAOptimizer::evaluatePopulation(int population)
{
  PROFILE_FUNCTION();
  int i;

  for (i = 0; i < population_sizes[population]; ++i)
  {
    this->costFunctionEvaluation(
      populations[population][i], i, objective_values[population][i], constraint_values[population][i]);
    this->SavePartialEvaluation(i);
  }
}

void
GOMEAOptimizer::costFunctionEvaluation(const ParametersType & parameters,
                                       int                    individual_index,
                                       MeasureType &          obj_val,
                                       MeasureType &          constraint_val)
{
  PROFILE_FUNCTION();
  obj_val = this->m_PartialEvaluations ? this->GetValue(parameters, -1, individual_index, constraint_val)
                                       : this->GetValue(parameters, constraint_val);
  constraint_val *= m_UseConstraints;

  ++m_NumberOfEvaluations;
}

void
GOMEAOptimizer::costFunctionEvaluation(int           population_index,
                                       int           individual_index,
                                       int           fos_index,
                                       MeasureType & obj_val,
                                       MeasureType & constraint_val)
{
  PROFILE_FUNCTION();
  if (!(this->m_PartialEvaluations))
  {
    this->costFunctionEvaluation(
      populations[population_index][individual_index], individual_index, obj_val, constraint_val);
    return;
  }

  obj_val =
    this->GetValue(populations[population_index][individual_index], fos_index, individual_index, constraint_val);
  constraint_val *= m_UseConstraints;

  // this->GetValueSanityCheck(populations[population_index][individual_index]);

  ++m_NumberOfEvaluations;
}

void
GOMEAOptimizer::GetValueSanityCheck(const ParametersType & parameters) const
{
  Array<double> derivative(m_NrOfParameters);
  MeasureType   constraint_val, constraint_val2;
  MeasureType   obj_val3;

  MeasureType obj_val = this->GetValue(parameters, constraint_val);
  MeasureType obj_val2 = this->GetValue(parameters, -1, 0, constraint_val2);
  this->GetCostFunction()->GetValueAndDerivative(parameters, obj_val3, derivative);

  if (abs(obj_val - obj_val2) > 1e-8 || abs(constraint_val - constraint_val2) > 1e-8)
    std::cout << "WTF0\n";

  if (abs(obj_val - obj_val3) > 1e-8)
    std::cout << "WTF1\n";
}

/**
 * Applies the distribution multipliers.
 */
void
GOMEAOptimizer::applyDistributionMultipliersToAllPopulations(void)
{
  PROFILE_FUNCTION();
  int i;

  for (i = 0; i < number_of_populations; i++)
    this->applyDistributionMultipliers(i);
}

void
GOMEAOptimizer::applyDistributionMultipliers(int population_index)
{
  PROFILE_FUNCTION();
  int j, k, m;

  if (!populations_terminated[population_index])
  {
    for (j = 0; j < linkage_model[population_index]->length; j++)
      for (k = 0; k < linkage_model[population_index]->set_length[j]; k++)
        for (m = 0; m < linkage_model[population_index]->set_length[j]; m++)
          decomposed_covariance_matrices[population_index][j](k, m) *= distribution_multipliers[population_index][j];
  }
}

void
GOMEAOptimizer::generateAndEvaluateNewSolutionsToFillAllPopulations(void)
{
  PROFILE_FUNCTION();
  int i;

  for (i = 0; i < number_of_populations; i++)
    this->generateAndEvaluateNewSolutionsToFillPopulation(i);
}

/**
 * Generates new solutions for each
 * of the populations in turn.
 */
void
GOMEAOptimizer::generateAndEvaluateNewSolutionsToFillPopulation(int population_index)
{
  PROFILE_FUNCTION();
  short generationalImprovement, *FOS_element_caused_improvement, all_multipliers_leq_one, *individual_improved,
    apply_AMS;
  int    oj, i, j, k, *fos_order, number_of_AMS_solutions, best_individual_index;
  double alpha_AMS;

  this->computeParametersForSampling(population_index);

  if (!populations_terminated[population_index])
  {
    FOS_element_caused_improvement = (short *)Malloc(linkage_model[population_index]->length * sizeof(short));
    individual_improved = (short *)Malloc(population_sizes[population_index] * sizeof(short));
    for (k = 1; k < population_sizes[population_index]; k++)
      individual_improved[k] = 0;

    alpha_AMS =
      0.5 * m_Tau * (((double)population_sizes[population_index]) / ((double)(population_sizes[population_index] - 1)));
    number_of_AMS_solutions = (int)(alpha_AMS * (population_sizes[population_index] - 1));
    fos_order = randomPermutation(linkage_model[population_index]->length);
    for (oj = 0; oj < linkage_model[population_index]->length; oj++)
    {
      j = fos_order[oj];

      FOS_element_caused_improvement[j] = 0;

      apply_AMS = 1;
      for (k = 1; k < population_sizes[population_index]; k++)
      {
        if (k > number_of_AMS_solutions)
          apply_AMS = 0;
        individual_improved[k] |= this->generateNewSolutionFromFOSElement(population_index, j, k, apply_AMS);
      }

      FOS_element_caused_improvement[j] = this->adaptDistributionMultipliers(population_index, j);
    }
    free(fos_order);

    if (number_of_generations[population_index] > 0)
    {
      for (k = 1; k <= number_of_AMS_solutions; k++)
        individual_improved[k] |= this->applyAMS(population_index, k);
    }

    for (i = 1; i < population_sizes[population_index]; i++)
      if (!individual_improved[i])
        individual_NIS[population_index][i]++;
      else
        individual_NIS[population_index][i] = 0;

    this->getBestInPopulation(population_index, &best_individual_index);
    for (k = 1; k < population_sizes[population_index]; k++)
      if (individual_NIS[population_index][k] > m_MaxNoImprovementStretch)
        this->applyForcedImprovements(population_index, k, best_individual_index);

    generationalImprovement = 0;
    for (j = 0; j < linkage_model[population_index]->length; j++)
      if (FOS_element_caused_improvement[j])
        generationalImprovement = 1;

    if (generationalImprovement)
      no_improvement_stretch[population_index] = 0;
    else
    {
      all_multipliers_leq_one = 1;
      for (j = 0; j < linkage_model[population_index]->length; j++)
        if (distribution_multipliers[population_index][j] > 1.0)
        {
          all_multipliers_leq_one = 0;
          break;
        }

      if (all_multipliers_leq_one)
        (no_improvement_stretch[population_index])++;
    }

    free(individual_improved);
    free(FOS_element_caused_improvement);
  }
}

/**
 * Computes the Cholesky decompositions required for sampling
 * the multivariate normal distribution.
 */
void
GOMEAOptimizer::computeParametersForSampling(int population_index)
{
  PROFILE_FUNCTION();
  int i;

  if (!use_univariate_FOS)
  {
    for (i = 0; i < linkage_model[population_index]->length; i++)
    {
      this->m_PdPctMean(choleskyDecomposition(decomposed_cholesky_factors_lower_triangle[population_index][i],
                                              decomposed_covariance_matrices[population_index][i],
                                              linkage_model[population_index]->set_length[i]));
    }
  }
}

/**
 * Generates and returns a single new solution by drawing
 * a sample for the variables in the selected FOS element
 * and inserting this into the population.
 */
void
GOMEAOptimizer::generateNewPartialSolutionFromFOSElement(int population_index, int FOS_index, VectorXd & result)
{
  PROFILE_FUNCTION();
  short    ready;
  int      i, num_indices, *indices;
  VectorXd z;

  num_indices = linkage_model[population_index]->set_length[FOS_index];
  indices = linkage_model[population_index]->sets[FOS_index];

  // // TODO: EigenRand?
  // VectorXd rz = VectorXd(num_indices).unaryExpr([&](float dummy) { return dist(gen); });
  // z = rz;

  z = VectorXd(num_indices);
  for (i = 0; i < num_indices; i++)
    z(i) = random1DNormalUnit();

  if (use_univariate_FOS)
  {
    result = VectorXd(1);
    result[0] = z[0] * sqrt(decomposed_covariance_matrices[population_index][FOS_index](0, 0)) +
                mean_vectors[population_index][indices[0]];
  }
  else
  {
    result = decomposed_cholesky_factors_lower_triangle[population_index][FOS_index].triangularView<Eigen::Lower>() * z;
    // TODO: mean vectors as eigen vector -> add using eigen operator.
    for (i = 0; i < num_indices; i++)
      result[i] += mean_vectors[population_index][indices[i]];
  }
}

/**
 * Generates and returns a single new solution by drawing
 * a single sample from a specified model.
 */
short
GOMEAOptimizer::generateNewSolutionFromFOSElement(int   population_index,
                                                  int   FOS_index,
                                                  int   individual_index,
                                                  short apply_AMS)
{
  PROFILE_FUNCTION();
  int      j, m, im, *indices, num_indices, *touched_indices, num_touched_indices;
  double * individual_backup, obj_val, cons_val{ 0.0 }, delta_AMS, shrink_factor;
  short    improvement, out_of_range;
  VectorXd result;

  delta_AMS = 2.0;
  improvement = 0;
  num_indices = linkage_model[population_index]->set_length[FOS_index];
  indices = linkage_model[population_index]->sets[FOS_index];
  num_touched_indices = num_indices;
  touched_indices = indices;
  individual_backup = (double *)Malloc(num_touched_indices * sizeof(double));

  if (m_PartialEvaluations)
  {
    this->PreloadPartialEvaluation(populations[population_index][individual_index], FOS_index);
  }

  for (j = 0; j < num_touched_indices; j++)
    individual_backup[j] = populations[population_index][individual_index][touched_indices[j]];

  this->generateNewPartialSolutionFromFOSElement(population_index, FOS_index, result);

  for (j = 0; j < num_indices; j++)
    populations[population_index][individual_index][indices[j]] = result[j];

  if (apply_AMS && (number_of_generations[population_index] > 0))
  {
    for (m = 0; m < num_indices; m++)
    {
      im = indices[m];
      result[m] = populations[population_index][individual_index][im] +
                  shrink_factor * delta_AMS * distribution_multipliers[population_index][FOS_index] *
                    mean_shift_vector[population_index][im];
    }
    for (m = 0; m < num_indices; m++)
    {
      populations[population_index][individual_index][indices[m]] = result[m];
    }
  }

  this->costFunctionEvaluation(population_index, individual_index, FOS_index, obj_val, cons_val);
  improvement = betterFitness(obj_val,
                              cons_val,
                              objective_values[population_index][individual_index],
                              constraint_values[population_index][individual_index]);
  if (improvement || randomRealUniform01() < 0.05)
  {
    objective_values[population_index][individual_index] = obj_val;
    constraint_values[population_index][individual_index] = cons_val;
    this->SavePartialEvaluation(individual_index);
  }
  else
  {
    for (j = 0; j < num_touched_indices; j++)
      populations[population_index][individual_index][touched_indices[j]] = individual_backup[j];
  }

  free(individual_backup);
  return (improvement);
}

short
GOMEAOptimizer::applyAMS(int population_index, int individual_index)
{
  PROFILE_FUNCTION();
  short          out_of_range, improvement;
  double         shrink_factor, delta_AMS, obj_val, cons_val{ 0.0 };
  ParametersType solution_AMS;
  int            m, k;

  delta_AMS = 2;
  shrink_factor = 2;
  improvement = 0;
  solution_AMS = ParametersType(m_NrOfParameters);
  for (m = 0; (unsigned)m < m_NrOfParameters; m++)
  {
    solution_AMS[m] = populations[population_index][individual_index][m] +
                      shrink_factor * delta_AMS *
                        mean_shift_vector[population_index][m]; //*distribution_multipliers[population_index][FOS_index]
  }

  this->costFunctionEvaluation(solution_AMS, individual_index, obj_val, cons_val);
  improvement = betterFitness(obj_val,
                              cons_val,
                              objective_values[population_index][individual_index],
                              constraint_values[population_index][individual_index]);
  if (improvement || randomRealUniform01() < 0.05)
  {
    objective_values[population_index][individual_index] = obj_val;
    constraint_values[population_index][individual_index] = cons_val;
    this->SavePartialEvaluation(individual_index);
    for (m = 0; (unsigned)m < m_NrOfParameters; m++)
      populations[population_index][individual_index][m] = solution_AMS[m];
    improvement = 1;
  }
  return (improvement);
}

void
GOMEAOptimizer::applyForcedImprovements(int population_index, int individual_index, int donor_index)
{
  PROFILE_FUNCTION();
  int     i, io, j, *order, *touched_indices, num_touched_indices;
  double *FI_backup, obj_val, cons_val{ 0.0 }, alpha;
  short   improvement;

  improvement = 0;
  alpha = 1.0;

  while (alpha >= 0.01)
  {
    alpha *= 0.5;
    order = randomPermutation(linkage_model[population_index]->length);
    for (io = 0; io < linkage_model[population_index]->length; io++)
    {
      i = order[io];
      if (m_PartialEvaluations)
      {
        this->PreloadPartialEvaluation(populations[population_index][individual_index], i);
      }
      touched_indices = linkage_model[population_index]->sets[i];
      num_touched_indices = linkage_model[population_index]->set_length[i];
      FI_backup = (double *)Malloc(num_touched_indices * sizeof(double));
      for (j = 0; j < num_touched_indices; j++)
      {
        FI_backup[j] = populations[population_index][individual_index][touched_indices[j]];
        populations[population_index][individual_index][touched_indices[j]] =
          alpha * populations[population_index][individual_index][touched_indices[j]] +
          (1 - alpha) * populations[population_index][donor_index][touched_indices[j]];
      }
      this->costFunctionEvaluation(population_index, individual_index, i, obj_val, cons_val);
      improvement = betterFitness(obj_val,
                                  cons_val,
                                  objective_values[population_index][individual_index],
                                  constraint_values[population_index][individual_index]);
      // printf("alpha=%.1e\tf=%.30e\n",alpha,obj_val);

      if (!improvement)
        for (j = 0; j < num_touched_indices; j++)
          populations[population_index][individual_index][touched_indices[j]] = FI_backup[j];
      else
      {
        objective_values[population_index][individual_index] = obj_val;
        constraint_values[population_index][individual_index] = cons_val;
        this->SavePartialEvaluation(individual_index);
      }

      free(FI_backup);

      if (improvement)
        break;
    }
    free(order);
    if (improvement)
      break;
  }

  if (improvement)
  {
    objective_values[population_index][individual_index] = obj_val;
    constraint_values[population_index][individual_index] = cons_val;
  }
  else
  {
    for (i = 0; (unsigned)i < m_NrOfParameters; i++)
      populations[population_index][individual_index][i] = populations[population_index][donor_index][i];
    objective_values[population_index][individual_index] = objective_values[population_index][donor_index];
    constraint_values[population_index][individual_index] = constraint_values[population_index][donor_index];
    this->CopyPartialEvaluation(donor_index, individual_index);
  }
}

/**
 * Adapts distribution multipliers according to SDR-AVS mechanism.
 * Returns whether the FOS element with index FOS_index has caused
 * an improvement in population_index.
 */
short
GOMEAOptimizer::adaptDistributionMultipliers(int population_index, int FOS_index)
{
  PROFILE_FUNCTION();
  short  improvementForFOSElement;
  int    i, j;
  double st_dev_ratio, increase_for_FOS_element, decrease_for_FOS_element;

  i = population_index;
  j = FOS_index;
  improvementForFOSElement = 0;
  increase_for_FOS_element = distribution_multiplier_increase;
  decrease_for_FOS_element = 1.0 / increase_for_FOS_element;
  if (!populations_terminated[i])
  {
    improvementForFOSElement = this->generationalImprovementForOnePopulationForFOSElement(i, j, &st_dev_ratio);

    if (improvementForFOSElement)
    {
      if (distribution_multipliers[i][j] < 1.0)
        distribution_multipliers[i][j] = 1.0;

      if (st_dev_ratio > m_StDevThreshold)
        distribution_multipliers[i][j] *= increase_for_FOS_element;
    }
    else
    {
      if ((distribution_multipliers[i][j] > 1.0) || (no_improvement_stretch[i] >= m_MaxNoImprovementStretch))
        distribution_multipliers[i][j] *= decrease_for_FOS_element;

      if (no_improvement_stretch[i] < m_MaxNoImprovementStretch && distribution_multipliers[i][j] < 1.0)
        distribution_multipliers[i][j] = 1.0;
    }
  }
  return (improvementForFOSElement);
}

/**
 * Determines whether an improvement is found for a specified
 * population. Returns 1 in case of an improvement, 0 otherwise.
 * The standard-deviation ratio required by the SDR-AVS
 * mechanism is computed and returned in the pointer variable.
 */
short
GOMEAOptimizer::generationalImprovementForOnePopulationForFOSElement(int      population_index,
                                                                     int      FOS_index,
                                                                     double * st_dev_ratio)
{
  PROFILE_FUNCTION();
  int      i, j, index_best_population, number_of_improvements, *indices, num_indices;
  double * average_parameters_of_improvements;
  short    generationalImprovement;

  generationalImprovement = 0;
  indices = linkage_model[population_index]->sets[FOS_index];
  num_indices = linkage_model[population_index]->set_length[FOS_index];

  // Determine best in the population and the average improvement parameters
  average_parameters_of_improvements = (double *)Malloc(num_indices * sizeof(double));
  for (i = 0; i < num_indices; i++)
    average_parameters_of_improvements[i] = 0.0;

  index_best_population = 0;
  number_of_improvements = 0;
  for (i = 0; i < population_sizes[population_index]; i++)
  {
    if (betterFitness(objective_values[population_index][i],
                      constraint_values[population_index][i],
                      objective_values[population_index][index_best_population],
                      constraint_values[population_index][index_best_population]))
      index_best_population = i;

    if (betterFitness(objective_values[population_index][i],
                      constraint_values[population_index][i],
                      objective_values_selections[population_index][0],
                      constraint_values_selections[population_index][0]))
    {
      number_of_improvements++;
      for (j = 0; j < num_indices; j++)
        average_parameters_of_improvements[j] += populations[population_index][i][indices[j]];
    }
  }

  // Determine st.dev. ratio
  *st_dev_ratio = 0.0;
  if (number_of_improvements > 0)
  {
    for (i = 0; i < num_indices; i++)
      average_parameters_of_improvements[i] /= (double)number_of_improvements;

    *st_dev_ratio = this->getStDevRatioForFOSElement(population_index, average_parameters_of_improvements, FOS_index);
    generationalImprovement = 1;
  }

  free(average_parameters_of_improvements);

  return (generationalImprovement);
}

/**
 * Computes and returns the standard-deviation-ratio
 * of a given point for a given model.
 */
double
GOMEAOptimizer::getStDevRatioForFOSElement(int population_index, double * parameters, int FOS_index)
{
  PROFILE_FUNCTION();
  int      i, *indices, num_indices;
  double   result;
  VectorXd x_min_mu, z;

  indices = linkage_model[population_index]->sets[FOS_index];
  num_indices = linkage_model[population_index]->set_length[FOS_index];
  x_min_mu = VectorXd(num_indices);

  for (i = 0; i < num_indices; i++)
    x_min_mu[i] = parameters[i] - mean_vectors[population_index][indices[i]];
  result = 0.0;

  if (use_univariate_FOS)
  {
    result = fabs(x_min_mu[0] / sqrt(decomposed_covariance_matrices[population_index][FOS_index](0, 0)));
  }
  else
  {
    MatrixXd inverse =
      decomposed_cholesky_factors_lower_triangle[population_index][FOS_index].triangularView<Eigen::Lower>();
    lowerTriangularMatrixInverse(inverse, num_indices);
    z = inverse * x_min_mu;

    for (i = 0; i < num_indices; i++)
    {
      if (fabs(z[i]) > result)
        result = fabs(z[i]);
    }
  }

  return (result);
}

void
GOMEAOptimizer::UpdatePosition()
{
  PROFILE_FUNCTION();
  this->SetCurrentPosition(selections[0][0]);
  this->costFunctionEvaluation(this->GetCurrentPosition(), 0, this->m_Value, this->m_ConstraintValue);

  m_CurrentIteration++;
  this->InvokeEvent(IterationEvent());
}

void
GOMEAOptimizer::saveVariancesForNextResolution()
{
  estimateFullCovarianceMatrixML(0);

  variances_last_resolution = ParametersType(number_of_parameters);
  for (unsigned int i = 0; i < number_of_parameters; i++)
    variances_last_resolution[i] = full_covariance_matrix[0](i, i);
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Ezilaitini -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Undoes initialization procedure by freeing up memory.
 */
void
GOMEAOptimizer::ezilaitini(void)
{
  this->ezilaitiniMemory();
}

/**
 * Undoes initialization procedure by freeing up memory.
 */
void
GOMEAOptimizer::ezilaitiniMemory(void)
{
  int i;

  for (i = 0; i < number_of_populations; i++)
  {
    ezilaitiniFOS(linkage_model[i]);
  }

  free(linkage_model);
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Run -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void
GOMEAOptimizer::generationalStepAllPopulations()
{
  int population_index_smallest, population_index_biggest;

  population_index_biggest = number_of_populations - 1;
  population_index_smallest = 0;
  while (population_index_smallest <= population_index_biggest)
  {
    if (!populations_terminated[population_index_smallest])
      break;

    population_index_smallest++;
  }

  this->generationalStepAllPopulationsRecursiveFold(population_index_smallest, population_index_biggest);
}

void
GOMEAOptimizer::generationalStepAllPopulationsRecursiveFold(int population_index_smallest, int population_index_biggest)
{
  int i, j, population_index;

  for (i = 0; i < number_of_subgenerations_per_population_factor - 1; i++)
  {
    for (population_index = population_index_smallest; population_index <= population_index_biggest; population_index++)
    {
      if (!populations_terminated[population_index])
      {
#ifdef ELASTIX_ENABLE_PROFILING
        std::string profileName = "Generation " + std::to_string(number_of_generations[population_index]);
        std::string profileFile =
          m_outFolderProfiling + "gen_" + std::to_string(number_of_generations[population_index]);
#endif
        PROFILE_BEGIN_SESSION(profileName, profileFile);

        this->makeSelectionsForOnePopulation(population_index);

        this->makePopulation(population_index);

        number_of_generations[population_index]++;
        this->UpdatePosition();

        if (this->m_SubSampling || number_of_generations[population_index] % NumberOfGenerationsPerReevaluation == 0)
          this->evaluatePopulation(population_index);

        PROFILE_END_SESSION();

        if (this->checkSubgenerationTerminationConditions())
        {
          for (j = 0; j < number_of_populations; j++)
            populations_terminated[j] = 1;
          return;
        }
      }
    }

    for (population_index = population_index_smallest; population_index < population_index_biggest; population_index++)
      this->generationalStepAllPopulationsRecursiveFold(population_index_smallest, population_index);
  }
}

void
GOMEAOptimizer::runAllPopulations()
{
  PROFILE_FUNCTION();
  while (!this->checkTerminationCondition())
  {
    if (number_of_populations < m_MaxNumberOfPopulations)
      this->initializeNewPopulation();

    PROFILE_END_SESSION();

    this->generationalStepAllPopulations();
  }
}

const std::string
GOMEAOptimizer::GetStopConditionDescription() const
{
  PROFILE_FUNCTION();
  m_StopConditionDescription.str("");
  switch (m_StopCondition)
  {
    case MaximumNumberOfEvaluationsTermination:
      m_StopConditionDescription << "Maximum number of evaluations termination";
      break;
    case MaximumNumberOfIterationsTermination:
      m_StopConditionDescription << "Maximum number of iterations termination";
      break;
    case AverageFitnessTermination:
      m_StopConditionDescription << "Average fitness termination";
      break;
    case FitnessVarianceTermination:
      m_StopConditionDescription << "Fitness variance termination";
      break;
    case DistributionMultiplierTermination:
      m_StopConditionDescription << "Distribution multiplier termination";
      break;
    case Unknown:
      m_StopConditionDescription << "Unknown termination";
      break;
  }
  return m_StopConditionDescription.str();
}

double
GOMEAOptimizer::GetAverageDistributionMultiplier() const
{
  PROFILE_FUNCTION();
  if (this->distribution_multipliers.size() > 0)
  {
    const int pop = number_of_populations - 1;
    double    sum =
      std::accumulate(this->distribution_multipliers[pop].begin(), this->distribution_multipliers[pop].end(), 0.0);
    return sum / this->distribution_multipliers[pop].size();
  }
  return 1.0;
}

void
GOMEAOptimizer::WriteDistributionMultipliers(std::ofstream & outfile) const
{
  PROFILE_FUNCTION();
  if (this->distribution_multipliers.size() > 0)
  {
    for (const double & distm : this->distribution_multipliers[0])
    {
      outfile << distm << " ";
    }
    outfile << "\n";
    outfile.flush();
  }
}

/**
 * Runs the IDEA.
 */
void
GOMEAOptimizer::run(void)
{
  PROFILE_FUNCTION();
  this->PrintSettings();
  this->runAllPopulations();
  this->ezilaitini();
  this->StopOptimization();
}
} // namespace itk

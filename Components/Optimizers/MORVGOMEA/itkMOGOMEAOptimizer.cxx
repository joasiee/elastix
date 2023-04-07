#include "itkMOGOMEAOptimizer.h"

namespace itk
{
using namespace MOGOMEA_UTIL;

/*-=-=-=-=-=-=-=-=-=-=- Section Interpret Command Line -=-=-=-=-=-=-=-=-=-=-*/
/**
 * Checks whether the selected options are feasible.
 */
void
MOGOMEAOptimizer::checkOptions(void)
{
  if (number_of_parameters < 1)
  {
    printf("\n");
    printf("Error: number of parameters < 1 (read: %d). Require number of parameters >= 1.", number_of_parameters);
    printf("\n\n");

    exit(0);
  }

  if (((int)(tau * base_population_size)) <= 0 || tau >= 1)
  {
    printf("\n");
    printf("Error: tau not in range (read: %e). Require tau in [1/pop,1] (read: [%e,%e]).",
           tau,
           1.0 / ((double)base_population_size),
           1.0);
    printf("\n\n");

    exit(0);
  }

  if (base_population_size < 1)
  {
    printf("\n");
    printf("Error: population size < 1 (read: %d). Require population size >= 1.", base_population_size);
    printf("\n\n");

    exit(0);
  }

  if (base_number_of_mixing_components < 1)
  {
    printf("\n");
    printf("Error: number of mixing components < 1 (read: %d). Require number of mixture components >= 1.",
           base_number_of_mixing_components);
    printf("\n\n");

    exit(0);
  }

  if (elitist_archive_size_target < 1)
  {
    printf("\n");
    printf("Error: elitist archive size target < 1 (read: %d).", elitist_archive_size_target);
    printf("\n\n");

    exit(0);
  }

  if (!learn_linkage_tree && !static_linkage_tree && !bspline_marginal_tree && FOS_element_size < 1)
  {
    printf("\n");
    printf("Error: FOS element size invalid (read: %d). Require FOS element size >= 1 or [-2, -6].", FOS_element_size);
    printf("\n\n");

    exit(0);
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Initialize -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Performs initializations that are required before starting a run.
 */
void
MOGOMEAOptimizer::initialize(void)
{
  int i, j;
  startTimer();

  if (maximum_number_of_populations == 1)
  {
    base_number_of_mixing_components = base_number_of_mixing_components == 0 ? 20 : base_number_of_mixing_components;
    base_population_size =
      base_population_size == 0
        ? (int)((0.5 * base_number_of_mixing_components) * (36.1 + 7.58 * log2((double)number_of_parameters)))
        : base_population_size;
  }
  else
  {
    base_number_of_mixing_components =
      base_number_of_mixing_components == 0 ? 1 + number_of_objectives : base_number_of_mixing_components;
    base_population_size = base_population_size == 0 ? 10 * base_number_of_mixing_components : base_population_size;
  }

  maximum_no_improvement_stretch =
    maximum_no_improvement_stretch == 0
      ? (int)(2.0 + ((double)(25 + number_of_parameters)) / ((double)base_number_of_mixing_components))
      : maximum_no_improvement_stretch;


  number_of_populations = 0;
  total_number_of_generations = 0;
  number_of_evaluations = 0;
  number_of_full_evaluations = 0;
  distribution_multiplier_increase = 1.0 / distribution_multiplier_decrease;
  objective_discretization_in_effect = 0;
  delta_AMS = 2.0;
  statistics_file_existed = 0;

  bspline_marginal_tree = 0;
  static_linkage_tree = 0;
  random_linkage_tree = 0;
  FOS_element_ub = number_of_parameters;

  if (FOS_element_size == FOSType::Full)
    FOS_element_size = number_of_parameters;
  if (FOS_element_size == FOSType::LinkageTree)
    learn_linkage_tree = 1;
  if (FOS_element_size == FOSType::StaticLinkageTree)
    static_linkage_tree = 1;
  if (FOS_element_size == FOSType::StaticBoundedLinkageTree)
  {
    static_linkage_tree = 1;
    FOS_element_ub = 100;
  }
  if (FOS_element_size == FOSType::StaticBoundedRandomLinkageTree)
  {
    random_linkage_tree = 1;
    static_linkage_tree = 1;
    FOS_element_ub = 100;
  }
  if (FOS_element_size == FOSType::MarginalControlPoints)
  {
    bspline_marginal_tree = 1;
  }
  if (FOS_element_size == FOSType::Univariate)
    use_univariate_FOS = 1;

  param_helper = ParametersType{ number_of_parameters };

  checkOptions();
  initializeMemory();
  this->InitializeRegistration();

  is_initialized = true;
}

void
MOGOMEAOptimizer::initializeNewPopulation(void)
{
  current_population_index = number_of_populations;

  initializeNewPopulationMemory(number_of_populations);

  if (m_PartialEvaluations)
    this->GetCostFunction()->InitSubfunctionSamplers(population_sizes[0]);

  initializePopulationAndFitnessValues(number_of_populations);

  if (!learn_linkage_tree)
  {
    initializeCovarianceMatrices(number_of_populations);

    initializeDistributionMultipliers(number_of_populations);
  }

  computeObjectiveRanges(number_of_populations);

  computeRanks(number_of_populations);

  number_of_populations++;
}

void
MOGOMEAOptimizer::initializeMemory()
{
  int i;

  elitist_archive_size = 0;
  elitist_archive_capacity = 10;
  populations = (individual ***)Malloc(maximum_number_of_populations * sizeof(individual **));
  selection = (individual ***)Malloc(maximum_number_of_populations * sizeof(individual **));
  population_sizes = (int *)Malloc(maximum_number_of_populations * sizeof(int));
  populations_terminated = (bool *)Malloc(maximum_number_of_populations * sizeof(bool));
  selection_sizes = (int *)Malloc(maximum_number_of_populations * sizeof(int));
  cluster_sizes = (int *)Malloc(maximum_number_of_populations * sizeof(int));
  cluster_index_for_population = (int **)Malloc(maximum_number_of_populations * sizeof(int *));
  ranks = (double **)Malloc(maximum_number_of_populations * sizeof(double *));
  sorted_ranks = (int **)Malloc(maximum_number_of_populations * sizeof(int *));
  objective_ranges = (double **)Malloc(maximum_number_of_populations * sizeof(double *));
  selection_indices = (int **)Malloc(maximum_number_of_populations * sizeof(int *));
  objective_values_selection_previous = (double ***)Malloc(maximum_number_of_populations * sizeof(double **));
  ranks_selection = (double **)Malloc(maximum_number_of_populations * sizeof(double *));
  pop_indices_selected = (int **)Malloc(maximum_number_of_populations * sizeof(int *));
  number_of_elitist_solutions_copied = (int **)Malloc(maximum_number_of_populations * sizeof(int *));
  objective_means_scaled = (double ***)Malloc(maximum_number_of_populations * sizeof(double **));
  mean_vectors = (double ***)Malloc(maximum_number_of_populations * sizeof(double **));
  mean_vectors_previous = (double ***)Malloc(maximum_number_of_populations * sizeof(double **));
  decomposed_cholesky_factors_lower_triangle =
    (double *****)Malloc(maximum_number_of_populations * sizeof(double ****));
  selection_indices_of_cluster_members = (int ***)Malloc(maximum_number_of_populations * sizeof(int **));
  selection_indices_of_cluster_members_previous = (int ***)Malloc(maximum_number_of_populations * sizeof(int **));
  single_objective_clusters = (int **)Malloc(maximum_number_of_populations * sizeof(int *));
  num_individuals_in_cluster = (int **)Malloc(maximum_number_of_populations * sizeof(int *));
  number_of_mixing_components = (int *)Malloc(maximum_number_of_populations * sizeof(int));
  distribution_multipliers = (double ***)Malloc(maximum_number_of_populations * sizeof(double **));
  decomposed_covariance_matrices = (double *****)Malloc(maximum_number_of_populations * sizeof(double ****));
  full_covariance_matrix = (double ****)Malloc(maximum_number_of_populations * sizeof(double ***));
  number_of_generations = (int *)Malloc(maximum_number_of_populations * sizeof(int));
  no_improvement_stretch = (int *)Malloc(maximum_number_of_populations * sizeof(int));
  linkage_model = (FOS ***)Malloc(maximum_number_of_populations * sizeof(FOS **));

  objective_discretization = (double *)Malloc(number_of_objectives * sizeof(double));
  elitist_archive = (individual **)Malloc(elitist_archive_capacity * sizeof(individual *));
  best_objective_values_in_elitist_archive = (double *)Malloc(number_of_objectives * sizeof(double));
  worst_objective_values_in_elitist_archive = (double *)Malloc(number_of_objectives * sizeof(double));
  elitist_archive_indices_inactive = (bool *)Malloc(elitist_archive_capacity * sizeof(bool));

  for (i = 0; i < elitist_archive_capacity; i++)
  {
    elitist_archive[i] = initializeIndividual();
    elitist_archive_indices_inactive[i] = 0;
  }

  for (i = 0; i < number_of_objectives; i++)
  {
    best_objective_values_in_elitist_archive[i] = 1e+308;
    worst_objective_values_in_elitist_archive[i] = -1e+308;
  }

  for (i = 0; i < maximum_number_of_populations; i++)
  {
    distribution_multipliers[i] = NULL;
  }
}


/**
 * Initializes the memory.
 */
void
MOGOMEAOptimizer::initializeNewPopulationMemory(int population_index)
{
  int i;

  if (population_index == 0)
  {
    population_sizes[population_index] = base_population_size;
    number_of_mixing_components[population_index] = base_number_of_mixing_components;
  }
  else
  {
    population_sizes[population_index] = 2 * population_sizes[population_index - 1];
    number_of_mixing_components[population_index] = number_of_mixing_components[population_index - 1] + 1;
  }
  selection_sizes[population_index] =
    population_sizes[population_index]; // HACK(int) (tau*(population_sizes[population_index]));
  cluster_sizes[population_index] =
    (2 * ((int)(tau * (population_sizes[population_index])))) /
    number_of_mixing_components[population_index]; // HACK(2*selection_size)/number_of_mixing_components;
  number_of_generations[population_index] = 0;
  populations_terminated[population_index] = 0;
  no_improvement_stretch[population_index] = 0;

  populations[population_index] = (individual **)Malloc(population_sizes[population_index] * sizeof(individual *));
  selection[population_index] = (individual **)Malloc(selection_sizes[population_index] * sizeof(individual *));
  cluster_index_for_population[population_index] = (int *)Malloc(population_sizes[population_index] * sizeof(int));
  ranks[population_index] = (double *)Malloc(population_sizes[population_index] * sizeof(double));
  sorted_ranks[population_index] = (int *)Malloc(population_sizes[population_index] * sizeof(int));
  objective_ranges[population_index] = (double *)Malloc(population_sizes[population_index] * sizeof(double));
  selection_indices[population_index] = (int *)Malloc(selection_sizes[population_index] * sizeof(int));
  objective_values_selection_previous[population_index] =
    (double **)Malloc(selection_sizes[population_index] * sizeof(double *));
  ranks_selection[population_index] = (double *)Malloc(selection_sizes[population_index] * sizeof(double));
  pop_indices_selected[population_index] = (int *)Malloc(population_sizes[population_index] * sizeof(int));
  number_of_elitist_solutions_copied[population_index] =
    (int *)Malloc(number_of_mixing_components[population_index] * sizeof(int));
  objective_means_scaled[population_index] =
    (double **)Malloc(number_of_mixing_components[population_index] * sizeof(double *));
  mean_vectors[population_index] = (double **)Malloc(number_of_mixing_components[population_index] * sizeof(double *));
  mean_vectors_previous[population_index] =
    (double **)Malloc(number_of_mixing_components[population_index] * sizeof(double *));
  decomposed_cholesky_factors_lower_triangle[population_index] =
    (double ****)Malloc(number_of_mixing_components[population_index] * sizeof(double ***));
  selection_indices_of_cluster_members[population_index] =
    (int **)Malloc(number_of_mixing_components[population_index] * sizeof(int *));
  selection_indices_of_cluster_members_previous[population_index] =
    (int **)Malloc(number_of_mixing_components[population_index] * sizeof(int *));
  single_objective_clusters[population_index] =
    (int *)Malloc(number_of_mixing_components[population_index] * sizeof(int));
  num_individuals_in_cluster[population_index] =
    (int *)Malloc(number_of_mixing_components[population_index] * sizeof(int));
  linkage_model[population_index] = (FOS **)Malloc(number_of_mixing_components[population_index] * sizeof(FOS *));

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    linkage_model[population_index][i] = NULL;

  for (i = 0; i < population_sizes[population_index]; i++)
    populations[population_index][i] = initializeIndividual();

  for (i = 0; i < selection_sizes[population_index]; i++)
  {
    selection[population_index][i] = initializeIndividual();

    objective_values_selection_previous[population_index][i] = (double *)Malloc(number_of_objectives * sizeof(double));
  }

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    mean_vectors[population_index][i] = (double *)Malloc(number_of_parameters * sizeof(double));

    mean_vectors_previous[population_index][i] = (double *)Malloc(number_of_parameters * sizeof(double));

    selection_indices_of_cluster_members[population_index][i] = NULL;

    selection_indices_of_cluster_members_previous[population_index][i] = NULL;

    objective_means_scaled[population_index][i] = (double *)Malloc(number_of_objectives * sizeof(double));
  }

  if (learn_linkage_tree)
  {
    full_covariance_matrix[population_index] =
      (double ***)Malloc(number_of_mixing_components[population_index] * sizeof(double **));
  }
  else
  {
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      initializeFOS(population_index, i);
  }
}

void
MOGOMEAOptimizer::initializeCovarianceMatrices(int population_index)
{
  int i, j, k, m;

  decomposed_covariance_matrices[population_index] =
    (double ****)Malloc(number_of_mixing_components[population_index] * sizeof(double ***));
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    decomposed_covariance_matrices[population_index][i] =
      (double ***)Malloc(linkage_model[population_index][i]->length * sizeof(double **));
    for (j = 0; j < linkage_model[population_index][i]->length; j++)
    {
      decomposed_covariance_matrices[population_index][i][j] =
        (double **)Malloc(linkage_model[population_index][i]->set_length[j] * sizeof(double *));
      for (k = 0; k < linkage_model[population_index][i]->set_length[j]; k++)
      {
        decomposed_covariance_matrices[population_index][i][j][k] =
          (double *)Malloc(linkage_model[population_index][i]->set_length[j] * sizeof(double));
        for (m = 0; m < linkage_model[population_index][i]->set_length[j]; m++)
        {
          decomposed_covariance_matrices[population_index][i][j][k][m] = 1;
        }
      }
    }
  }
}

/**
 * Initializes the distribution multipliers.
 */
void
MOGOMEAOptimizer::initializeDistributionMultipliers(int population_index)
{
  int i, j;

  distribution_multipliers[population_index] =
    (double **)Malloc(number_of_mixing_components[population_index] * sizeof(double *));
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    distribution_multipliers[population_index][i] =
      (double *)Malloc(linkage_model[population_index][i]->length * sizeof(double));
    for (j = 0; j < linkage_model[population_index][i]->length; j++)
      distribution_multipliers[population_index][i][j] = 1.0;
  }
}

/**
 * Initializes the population and the fitness values.
 */
void
MOGOMEAOptimizer::initializePopulationAndFitnessValues(int population_index)
{
  int i, j;

  for (i = 0; i < population_sizes[population_index]; i++)
  {
    const double scaleFactor = randomRealUniform01() * 0.25;
    const int    mixing_index = i % number_of_mixing_components[population_index];

    const ParametersType & base_parameters =
      m_CurrentResolution == 0 ? this->GetCurrentPosition() : this->GetPositionForMixingComponent(mixing_index);
    for (j = 0; j < number_of_parameters; j++)
    {
      populations[population_index][i]->parameters[j] =
        base_parameters[j] + (i > 0) * scaleFactor * random1DNormalUnit();
    }

    evaluateIndividual(population_index, i, -1);
    updateElitistArchive(populations[population_index][i]);
    this->SavePartialEvaluation(i);
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Ranking -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Computes the ranks of the solutions in all populations.
 */
void
MOGOMEAOptimizer::computeRanks(int population_index)
{
  bool **domination_matrix, is_illegal;
  int    i, j, k, *being_dominated_count, rank, number_of_solutions_ranked, *indices_in_this_rank;

  for (i = 0; i < population_sizes[population_index]; i++)
  {
    is_illegal = 0;
    for (j = 0; j < number_of_objectives; j++)
    {
      if (isnan(populations[population_index][i]->objective_values[j]))
      {
        is_illegal = 1;
        break;
      }
    }
    if (isnan(populations[population_index][i]->constraint_value))
      is_illegal = 1;

    if (is_illegal)
    {
      for (j = 0; j < number_of_objectives; j++)
        populations[population_index][i]->objective_values[j] = 1e+308;
      populations[population_index][i]->constraint_value = 1e+308;
    }
  }

  /* The domination matrix stores for each solution i
   * whether it dominates solution j, i.e. domination[i][j] = 1. */
  domination_matrix = (bool **)Malloc(population_sizes[population_index] * sizeof(bool *));
  for (i = 0; i < population_sizes[population_index]; i++)
    domination_matrix[i] = (bool *)Malloc(population_sizes[population_index] * sizeof(bool));

  being_dominated_count = (int *)Malloc(population_sizes[population_index] * sizeof(int));

  for (i = 0; i < population_sizes[population_index]; i++)
  {
    being_dominated_count[i] = 0;
    for (j = 0; j < population_sizes[population_index]; j++)
      domination_matrix[i][j] = 0;
  }

  for (i = 0; i < population_sizes[population_index]; i++)
  {
    for (j = 0; j < population_sizes[population_index]; j++)
    {
      if (i != j)
      {
        if (constraintParetoDominates(populations[population_index][i]->objective_values,
                                      populations[population_index][i]->constraint_value,
                                      populations[population_index][j]->objective_values,
                                      populations[population_index][j]->constraint_value))
        {
          domination_matrix[i][j] = 1;
          being_dominated_count[j]++;
        }
      }
    }
  }

  /* Compute ranks from the domination matrix */
  rank = 0;
  number_of_solutions_ranked = 0;
  indices_in_this_rank = (int *)Malloc(population_sizes[population_index] * sizeof(int));
  while (number_of_solutions_ranked < population_sizes[population_index])
  {
    k = 0;
    for (i = 0; i < population_sizes[population_index]; i++)
    {
      if (being_dominated_count[i] == 0)
      {
        ranks[population_index][i] = rank;
        indices_in_this_rank[k] = i;
        k++;
        being_dominated_count[i]--;
        number_of_solutions_ranked++;
      }
    }

    for (i = 0; i < k; i++)
    {
      for (j = 0; j < population_sizes[population_index]; j++)
      {
        if (domination_matrix[indices_in_this_rank[i]][j] == 1)
          being_dominated_count[j]--;
      }
    }

    rank++;
  }

  free(indices_in_this_rank);

  free(being_dominated_count);

  for (i = 0; i < population_sizes[population_index]; i++)
    free(domination_matrix[i]);
  free(domination_matrix);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Output =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Computes the ranges of all fitness values
 * of all solutions currently in the populations.
 */
void
MOGOMEAOptimizer::computeObjectiveRanges(int population_index)
{
  int    i, j;
  double low, high;

  for (j = 0; j < number_of_objectives; j++)
  {
    low = 1e+308;
    high = -1e+308;

    for (i = 0; i < population_sizes[population_index]; i++)
    {
      if (populations[population_index][i]->objective_values[j] < low)
        low = populations[population_index][i]->objective_values[j];
      if (populations[population_index][i]->objective_values[j] > high &&
          (populations[population_index][i]->objective_values[j] <= 1e+300))
        high = populations[population_index][i]->objective_values[j];
    }

    objective_ranges[population_index][j] = high - low;
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Termination -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns 1 if termination should be enforced, 0 otherwise.
 */
bool
MOGOMEAOptimizer::checkTerminationConditionAllPopulations(void)
{
  int i;

  if (number_of_populations == 0)
    return (0);

  if (checkNumberOfEvaluationsTerminationCondition())
    return (1);

  if (checkNumberOfGenerationsTerminationCondition())
    return (1);

  if (checkTimeLimitTerminationCondition())
    return (1);

  for (i = 0; i < number_of_populations; i++)
    if (checkDistributionMultiplierTerminationCondition(i))
      populations_terminated[i] = 1;

  return (0);
}

bool
MOGOMEAOptimizer::checkTerminationConditionOnePopulation(int population_index)
{
  if (number_of_populations == 0)
    return (0);

  if (checkNumberOfEvaluationsTerminationCondition())
    return (1);

  if (checkNumberOfGenerationsTerminationCondition())
    return (1);

  if (checkTimeLimitTerminationCondition())
    return (1);

  if (checkDistributionMultiplierTerminationCondition(population_index))
    populations_terminated[population_index] = 1;

  return (0);
}

/**
 * Returns 1 if the maximum number of evaluations
 * has been reached, 0 otherwise.
 */
bool
MOGOMEAOptimizer::checkNumberOfEvaluationsTerminationCondition(void)
{
  if (number_of_evaluations >= maximum_number_of_evaluations && maximum_number_of_evaluations > 0)
  {
    m_StopCondition = StopConditionType::MaximumNumberOfEvaluationsTermination;
    return (1);
  }

  return (0);
}

/**
 * Checks whether the distribution multiplier for any mixture component
 * has become too small (0.5).
 */
bool
MOGOMEAOptimizer::checkDistributionMultiplierTerminationCondition(int population_index)
{
  int i, j;

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    for (j = 0; j < linkage_model[population_index][i]->length; j++)
      if (distribution_multipliers[population_index][i][j] > 5e-1)
        return (0);
  }

  m_StopCondition = StopConditionType::DistributionMultiplierTermination;

  return (1);
}

bool
MOGOMEAOptimizer::checkTimeLimitTerminationCondition(void)
{
  if (maximum_number_of_seconds > 0 && getTimer() > maximum_number_of_seconds)
  {
    m_StopCondition = StopConditionType::MaximumNumberOfSecondsTermination;
    return (1);
  }
  return (0);
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Selection =-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Makes a set of selected solutions by taking the solutions from all
 * ranks up to the rank of the solution at the selection-size boundary.
 * The selection is filled using a diverse selection from the final rank.
 */
void
MOGOMEAOptimizer::makeSelection(int population_index)
{
  int i, j, k, individuals_selected, individuals_to_select, last_selected_rank, elitist_solutions_copied;

  for (i = 0; i < selection_sizes[population_index]; i++)
    for (j = 0; j < number_of_objectives; j++)
      objective_values_selection_previous[population_index][i][j] = selection[population_index][i]->objective_values[j];

  for (i = 0; i < population_sizes[population_index]; i++)
    pop_indices_selected[population_index][i] = -1;

  free(sorted_ranks[population_index]);
  sorted_ranks[population_index] = mergeSort(ranks[population_index], population_sizes[population_index]);

  // Copy elitist archive to selection
  elitist_solutions_copied = 0;
  individuals_selected = elitist_solutions_copied;
  individuals_to_select = ((int)(tau * population_sizes[population_index])) - elitist_solutions_copied;
  last_selected_rank = (int)ranks[population_index][sorted_ranks[population_index][individuals_to_select - 1]];

  i = 0;
  while (((int)ranks[population_index][sorted_ranks[population_index][i]]) != last_selected_rank)
  {
    for (j = 0; j < number_of_parameters; j++)
      selection[population_index][individuals_selected]->parameters[j] =
        populations[population_index][sorted_ranks[population_index][i]]->parameters[j];
    for (j = 0; j < number_of_objectives; j++)
      selection[population_index][individuals_selected]->objective_values[j] =
        populations[population_index][sorted_ranks[population_index][i]]->objective_values[j];
    selection[population_index][individuals_selected]->constraint_value =
      populations[population_index][sorted_ranks[population_index][i]]->constraint_value;
    ranks_selection[population_index][individuals_selected] =
      ranks[population_index][sorted_ranks[population_index][i]];
    selection_indices[population_index][individuals_selected] = sorted_ranks[population_index][i];
    pop_indices_selected[population_index][sorted_ranks[population_index][i]] = individuals_selected;

    i++;
    individuals_selected++;
  }

  int *selected_indices, start_index;
  selected_indices = NULL;
  if (individuals_selected < individuals_to_select)
    selected_indices = completeSelectionBasedOnDiversityInLastSelectedRank(
      population_index, i, individuals_to_select - individuals_selected, sorted_ranks[population_index]);

  if (selected_indices)
  {
    start_index = i;
    for (j = 0; individuals_selected < individuals_to_select; individuals_selected++, j++)
      pop_indices_selected[population_index][sorted_ranks[population_index][selected_indices[j] + start_index]] =
        individuals_selected;
  }

  j = individuals_to_select;
  for (i = 0; i < population_sizes[population_index]; i++)
  {
    if (pop_indices_selected[population_index][i] == -1)
    {
      for (k = 0; k < number_of_parameters; k++)
        selection[population_index][j]->parameters[k] = populations[population_index][i]->parameters[k];
      for (k = 0; k < number_of_objectives; k++)
        selection[population_index][j]->objective_values[k] = populations[population_index][i]->objective_values[k];
      selection[population_index][j]->constraint_value = populations[population_index][i]->constraint_value;
      ranks_selection[population_index][j] = ranks[population_index][i];
      selection_indices[population_index][j] = i;
      pop_indices_selected[population_index][i] = j;
      j++;
    }
  }

  if (selected_indices)
    free(selected_indices);
}

/**
 * Fills up the selection by using greedy diversity selection
 * in the last selected rank.
 */
int *
MOGOMEAOptimizer::completeSelectionBasedOnDiversityInLastSelectedRank(int   population_index,
                                                                      int   start_index,
                                                                      int   number_to_select,
                                                                      int * sorted)
{
  int       i, j, *selected_indices, number_of_points, number_of_dimensions;
  double ** points;

  /* Determine the solutions to select from */
  number_of_points = 0;
  while (ranks[population_index][sorted[start_index + number_of_points]] ==
         ranks[population_index][sorted[start_index]])
  {
    number_of_points++;
    if ((start_index + number_of_points) == population_sizes[population_index])
      break;
  }

  points = (double **)Malloc(number_of_points * sizeof(double *));
  for (i = 0; i < number_of_points; i++)
    points[i] = (double *)Malloc(number_of_objectives * sizeof(double));
  for (i = 0; i < number_of_points; i++)
    for (j = 0; j < number_of_objectives; j++)
      points[i][j] = populations[population_index][sorted[start_index + i]]->objective_values[j] /
                     objective_ranges[population_index][j];

  /* Select */
  number_of_dimensions = number_of_objectives;
  selected_indices = greedyScatteredSubsetSelection(points, number_of_points, number_of_dimensions, number_to_select);

  /* Copy to selection */
  for (i = 0; i < number_to_select; i++)
  {
    for (j = 0; j < number_of_parameters; j++)
      selection[population_index][i + start_index]->parameters[j] =
        populations[population_index][sorted[selected_indices[i] + start_index]]->parameters[j];
    for (j = 0; j < number_of_objectives; j++)
      selection[population_index][i + start_index]->objective_values[j] =
        populations[population_index][sorted[selected_indices[i] + start_index]]->objective_values[j];
    selection[population_index][i + start_index]->constraint_value =
      populations[population_index][sorted[selected_indices[i] + start_index]]->constraint_value;
    ranks_selection[population_index][i + start_index] =
      ranks[population_index][sorted[selected_indices[i] + start_index]];
    selection_indices[population_index][i + start_index] = sorted[selected_indices[i] + start_index];
  }

  for (i = 0; i < number_of_points; i++)
    free(points[i]);
  free(points);

  return (selected_indices);
}

bool
MOGOMEAOptimizer::checkNumberOfGenerationsTerminationCondition(void)
{
  if ((total_number_of_generations >= maximum_number_of_generations) && (maximum_number_of_generations > 0))
  {
    m_StopCondition = StopConditionType::MaximumNumberOfGenerationsTermination;
    return (1);
  }

  return (0);
}

/**
 * Selects n points from a set of points. A
 * greedy heuristic is used to find a good
 * scattering of the selected points. First,
 * a point is selected with a maximum value
 * in a randomly selected dimension. The
 * remaining points are selected iteratively.
 * In each iteration, the point selected is
 * the one that maximizes the minimal distance
 * to the points selected so far.
 */
int *
MOGOMEAOptimizer::greedyScatteredSubsetSelection(double ** points,
                                                 int       number_of_points,
                                                 int       number_of_dimensions,
                                                 int       number_to_select)
{
  int     i, index_of_farthest, random_dimension_index, number_selected_so_far, *indices_left, *result;
  double *nn_distances, distance_of_farthest, value;

  if (number_to_select > number_of_points)
  {
    printf("\n");
    printf("Error: greedyScatteredSubsetSelection asked to select %d solutions from set of size %d.",
           number_to_select,
           number_of_points);
    printf("\n\n");

    exit(0);
  }

  result = (int *)Malloc(number_to_select * sizeof(int));

  indices_left = (int *)Malloc(number_of_points * sizeof(int));
  for (i = 0; i < number_of_points; i++)
    indices_left[i] = i;

  /* Find the first point: maximum value in a randomly chosen dimension */
  random_dimension_index = randomInt(number_of_dimensions);

  index_of_farthest = 0;
  distance_of_farthest = points[indices_left[index_of_farthest]][random_dimension_index];
  for (i = 1; i < number_of_points; i++)
  {
    if (points[indices_left[i]][random_dimension_index] > distance_of_farthest)
    {
      index_of_farthest = i;
      distance_of_farthest = points[indices_left[i]][random_dimension_index];
    }
  }

  number_selected_so_far = 0;
  result[number_selected_so_far] = indices_left[index_of_farthest];
  indices_left[index_of_farthest] = indices_left[number_of_points - number_selected_so_far - 1];
  number_selected_so_far++;

  /* Then select the rest of the solutions: maximum minimum
   * (i.e. nearest-neighbour) distance to so-far selected points */
  nn_distances = (double *)Malloc(number_of_points * sizeof(double));
  for (i = 0; i < number_of_points - number_selected_so_far; i++)
    nn_distances[i] =
      distanceEuclidean(points[indices_left[i]], points[result[number_selected_so_far - 1]], number_of_dimensions);

  while (number_selected_so_far < number_to_select)
  {
    index_of_farthest = 0;
    distance_of_farthest = nn_distances[0];
    for (i = 1; i < number_of_points - number_selected_so_far; i++)
    {
      if (nn_distances[i] > distance_of_farthest)
      {
        index_of_farthest = i;
        distance_of_farthest = nn_distances[i];
      }
    }

    result[number_selected_so_far] = indices_left[index_of_farthest];
    indices_left[index_of_farthest] = indices_left[number_of_points - number_selected_so_far - 1];
    nn_distances[index_of_farthest] = nn_distances[number_of_points - number_selected_so_far - 1];
    number_selected_so_far++;

    for (i = 0; i < number_of_points - number_selected_so_far; i++)
    {
      value =
        distanceEuclidean(points[indices_left[i]], points[result[number_selected_so_far - 1]], number_of_dimensions);
      if (value < nn_distances[i])
        nn_distances[i] = value;
    }
  }

  free(nn_distances);
  free(indices_left);

  return (result);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Variation -==-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * First estimates the parameters of a normal mixture distribution
 * in the parameter space from the selected solutions. Then copies
 * the best selected solutions. Finally fills up the population,
 * after the variances of the mixture components have been scaled,
 * by drawing new samples from normal mixture distribution and applying
 * AMS to several of these new solutions. Then, the fitness ranks
 * are recomputed. Finally, the distribution multipliers for the
 * mixture components are adapted according to the SDR-AVS mechanism.
 */
void
MOGOMEAOptimizer::makePopulation(int population_index)
{
  current_population_index = population_index;

  estimateParameters(population_index);

  applyDistributionMultipliers(population_index);

  generateAndEvaluateNewSolutionsToFillPopulationAndUpdateElitistArchive(population_index);

  computeRanks(population_index);

  computeObjectiveRanges(population_index);

  adaptObjectiveDiscretization();

  ezilaitiniParametersForSampling(population_index);
}

/**
 * Estimates the parameters of the multivariate normal
 * mixture distribution.
 */
void
MOGOMEAOptimizer::estimateParameters(int population_index)
{
  bool *clusters_now_already_registered, *clusters_previous_already_registered;
  int   i, j, k, m, q, i_min, j_min, *selection_indices_of_leaders, number_of_dimensions, number_to_select,
    **selection_indices_of_cluster_members_before_registration, *k_means_cluster_sizes,
    **selection_indices_of_cluster_members_k_means, *nearest_neighbour_choice_best, number_of_clusters_left_to_register,
    *sorted, *r_nearest_neighbours_now, *r_nearest_neighbours_previous, number_of_clusters_to_register_by_permutation,
    number_of_cluster_permutations, **all_cluster_permutations;
  double **objective_values_selection_scaled, **objective_values_selection_previous_scaled, distance, distance_smallest,
    distance_largest, **objective_means_scaled_new, **objective_means_scaled_previous, *distances_to_cluster,
    **distance_cluster_i_now_to_cluster_j_previous, **distance_cluster_i_now_to_cluster_j_now,
    **distance_cluster_i_previous_to_cluster_j_previous, epsilon;

  /* Determine the leaders */
  objective_values_selection_scaled = (double **)Malloc(selection_sizes[population_index] * sizeof(double *));
  for (i = 0; i < selection_sizes[population_index]; i++)
    objective_values_selection_scaled[i] = (double *)Malloc(number_of_objectives * sizeof(double));
  for (i = 0; i < selection_sizes[population_index]; i++)
    for (j = 0; j < number_of_objectives; j++)
      objective_values_selection_scaled[i][j] =
        selection[population_index][i]->objective_values[j] / objective_ranges[population_index][j];

  /* Heuristically find k far-apart leaders, taken from an artificial selection */
  int leader_selection_size;

  leader_selection_size = tau * population_sizes[population_index];

  number_of_dimensions = number_of_objectives;
  number_to_select = number_of_mixing_components[population_index];
  selection_indices_of_leaders = greedyScatteredSubsetSelection(
    objective_values_selection_scaled, leader_selection_size, number_of_dimensions, number_to_select);

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    for (j = 0; j < number_of_objectives; j++)
      objective_means_scaled[population_index][i][j] =
        selection[population_index][selection_indices_of_leaders[i]]->objective_values[j] /
        objective_ranges[population_index][j];

  /* Perform k-means clustering with leaders as initial mean guesses */
  objective_means_scaled_new = (double **)Malloc(number_of_mixing_components[population_index] * sizeof(double *));
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    objective_means_scaled_new[i] = (double *)Malloc(number_of_objectives * sizeof(double));

  objective_means_scaled_previous = (double **)Malloc(number_of_mixing_components[population_index] * sizeof(double *));
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    objective_means_scaled_previous[i] = (double *)Malloc(number_of_objectives * sizeof(double));

  selection_indices_of_cluster_members_k_means =
    (int **)Malloc(number_of_mixing_components[population_index] * sizeof(int *));
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    selection_indices_of_cluster_members_k_means[i] = (int *)Malloc(selection_sizes[population_index] * sizeof(int));

  k_means_cluster_sizes = (int *)Malloc(number_of_mixing_components[population_index] * sizeof(int));

  for (j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++)
    single_objective_clusters[population_index][j] = -1;
  for (j = 0; j < number_of_objectives; j++)
    single_objective_clusters[population_index]
                             [number_of_mixing_components[population_index] - number_of_objectives + j] = j;

  epsilon = 1e+308;
  // BEGIN HACK: This essentially causes the code to skip k-means clustering
  epsilon = 0;
  for (j = 0; j < number_of_mixing_components[population_index]; j++)
    k_means_cluster_sizes[j] = 0;
  // END HACK
  while (epsilon > 1e-10)
  {
    for (j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++)
    {
      k_means_cluster_sizes[j] = 0;
      for (k = 0; k < number_of_objectives; k++)
        objective_means_scaled_new[j][k] = 0.0;
    }

    for (i = 0; i < selection_sizes[population_index]; i++)
    {
      j_min = -1;
      distance_smallest = -1;
      for (j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++)
      {
        distance = distanceEuclidean(
          objective_values_selection_scaled[i], objective_means_scaled[population_index][j], number_of_objectives);
        if ((distance_smallest < 0) || (distance < distance_smallest))
        {
          j_min = j;
          distance_smallest = distance;
        }
      }
      selection_indices_of_cluster_members_k_means[j_min][k_means_cluster_sizes[j_min]] = i;
      for (k = 0; k < number_of_objectives; k++)
        objective_means_scaled_new[j_min][k] += objective_values_selection_scaled[i][k];
      k_means_cluster_sizes[j_min]++;
    }

    for (j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++)
      for (k = 0; k < number_of_objectives; k++)
        objective_means_scaled_new[j][k] /= (double)k_means_cluster_sizes[j];

    epsilon = 0;
    for (j = 0; j < number_of_mixing_components[population_index] - number_of_objectives; j++)
    {
      epsilon += distanceEuclidean(
        objective_means_scaled[population_index][j], objective_means_scaled_new[j], number_of_objectives);
      for (k = 0; k < number_of_objectives; k++)
        objective_means_scaled[population_index][j][k] = objective_means_scaled_new[j][k];
    }
  }

  /* Do leader-based distance assignment */
  distances_to_cluster = (double *)Malloc(selection_sizes[population_index] * sizeof(double));
  for (i = 0; i < number_of_mixing_components[population_index] - number_of_objectives; i++)
  {
    for (j = 0; j < selection_sizes[population_index]; j++)
      distances_to_cluster[j] = distanceEuclidean(
        objective_values_selection_scaled[j], objective_means_scaled[population_index][i], number_of_objectives);
    for (j = leader_selection_size; j < selection_sizes[population_index]; j++)
      distances_to_cluster[j] = 1e+308; // HACK

    if (selection_indices_of_cluster_members_previous[population_index][i] != NULL)
      free(selection_indices_of_cluster_members_previous[population_index][i]);
    selection_indices_of_cluster_members_previous[population_index][i] =
      selection_indices_of_cluster_members[population_index][i];
    selection_indices_of_cluster_members[population_index][i] =
      mergeSort(distances_to_cluster, selection_sizes[population_index]);
  }

  // For k-th objective, create a cluster consisting of only the best solutions in that objective (from the overall
  // selection)
  for (j = number_of_mixing_components[population_index] - number_of_objectives;
       j < number_of_mixing_components[population_index];
       j++)
  {
    double *individual_objectives, worst;

    individual_objectives = (double *)Malloc(selection_sizes[population_index] * sizeof(double));

    worst = -1e+308;
    for (i = 0; i < selection_sizes[population_index]; i++)
    {
      individual_objectives[i] =
        selection[population_index][i]
          ->objective_values[j - (number_of_mixing_components[population_index] - number_of_objectives)];
      if (individual_objectives[i] > worst)
        worst = individual_objectives[i];
    }
    for (i = 0; i < selection_sizes[population_index]; i++)
    {
      if (selection[population_index][i]->constraint_value != 0)
        individual_objectives[i] = worst + selection[population_index][i]->constraint_value;
    }

    if (selection_indices_of_cluster_members_previous[population_index][j] != NULL)
      free(selection_indices_of_cluster_members_previous[population_index][j]);
    selection_indices_of_cluster_members_previous[population_index][j] =
      selection_indices_of_cluster_members[population_index][j];
    selection_indices_of_cluster_members[population_index][j] =
      mergeSort(individual_objectives, selection_sizes[population_index]);
    free(individual_objectives);
  }

  /* Re-assign cluster indices to achieve cluster registration,
   * i.e. make cluster i in this generation to be the cluster that is
   * closest to cluster i of the previous generation. The
   * algorithm first computes all distances between clusters in
   * the current generation and the previous generation. It also
   * computes all distances between the clusters in the current
   * generation and all distances between the clusters in the
   * previous generation. Then it determines the two clusters
   * that are the farthest apart. It randomly takes one of
   * these two far-apart clusters and its r nearest neighbours.
   * It also finds the closest cluster among those of the previous
   * generation and its r nearest neighbours. All permutations
   * are then considered to register these two sets. Subset
   * registration continues in this fashion until all clusters
   * are registered. */
  if (number_of_generations[population_index] > 0)
  {
    number_of_nearest_neighbours_in_registration = 7;

    objective_values_selection_previous_scaled =
      (double **)Malloc(selection_sizes[population_index] * sizeof(double *));
    for (i = 0; i < selection_sizes[population_index]; i++)
      objective_values_selection_previous_scaled[i] = (double *)Malloc(number_of_objectives * sizeof(double));

    for (i = 0; i < selection_sizes[population_index]; i++)
      for (j = 0; j < number_of_objectives; j++)
        objective_values_selection_previous_scaled[i][j] =
          objective_values_selection_previous[population_index][i][j] / objective_ranges[population_index][j];

    selection_indices_of_cluster_members_before_registration =
      (int **)Malloc(number_of_mixing_components[population_index] * sizeof(int *));
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      selection_indices_of_cluster_members_before_registration[i] =
        selection_indices_of_cluster_members[population_index][i];

    distance_cluster_i_now_to_cluster_j_previous =
      (double **)Malloc(number_of_mixing_components[population_index] * sizeof(double *));
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      distance_cluster_i_now_to_cluster_j_previous[i] =
        (double *)Malloc(number_of_mixing_components[population_index] * sizeof(double));

    /* Compute distances between clusters */
    ////// START DISTANCE COMPUTATION
    /* OLD: distance between clusters is the smallest distance between pairs of points.
     */
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
    {
      for (j = 0; j < number_of_mixing_components[population_index]; j++)
      {
        distance_cluster_i_now_to_cluster_j_previous[i][j] = 0;
        for (k = 0; k < cluster_sizes[population_index]; k++)
        {
          distance_smallest = -1;
          for (q = 0; q < cluster_sizes[population_index]; q++)
          {
            distance = distanceEuclidean(
              objective_values_selection_scaled[selection_indices_of_cluster_members_before_registration[i][k]],
              objective_values_selection_previous_scaled[selection_indices_of_cluster_members_previous[population_index]
                                                                                                      [j][q]],
              number_of_objectives);
            if ((distance_smallest < 0) || (distance < distance_smallest))
              distance_smallest = distance;
          }
          distance_cluster_i_now_to_cluster_j_previous[i][j] += distance_smallest;
        }
      }
    }

    distance_cluster_i_now_to_cluster_j_now =
      (double **)Malloc(number_of_mixing_components[population_index] * sizeof(double *));
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      distance_cluster_i_now_to_cluster_j_now[i] =
        (double *)Malloc(number_of_mixing_components[population_index] * sizeof(double));

    for (i = 0; i < number_of_mixing_components[population_index]; i++)
    {
      for (j = 0; j < number_of_mixing_components[population_index]; j++)
      {
        distance_cluster_i_now_to_cluster_j_now[i][j] = 0;
        if (i != j)
        {
          for (k = 0; k < cluster_sizes[population_index]; k++)
          {
            distance_smallest = -1;
            for (q = 0; q < cluster_sizes[population_index]; q++)
            {
              distance = distanceEuclidean(
                objective_values_selection_scaled[selection_indices_of_cluster_members_before_registration[i][k]],
                objective_values_selection_scaled[selection_indices_of_cluster_members_before_registration[j][q]],
                number_of_objectives);
              if ((distance_smallest < 0) || (distance < distance_smallest))
                distance_smallest = distance;
            }
            distance_cluster_i_now_to_cluster_j_now[i][j] += distance_smallest;
          }
        }
      }
    }

    distance_cluster_i_previous_to_cluster_j_previous =
      (double **)Malloc(number_of_mixing_components[population_index] * sizeof(double *));
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      distance_cluster_i_previous_to_cluster_j_previous[i] =
        (double *)Malloc(number_of_mixing_components[population_index] * sizeof(double));

    for (i = 0; i < number_of_mixing_components[population_index]; i++)
    {
      for (j = 0; j < number_of_mixing_components[population_index]; j++)
      {
        distance_cluster_i_previous_to_cluster_j_previous[i][j] = 0;
        if (i != j)
        {
          for (k = 0; k < cluster_sizes[population_index]; k++)
          {
            distance_smallest = -1;
            for (q = 0; q < cluster_sizes[population_index]; q++)
            {
              distance = distanceEuclidean(objective_values_selection_previous_scaled
                                             [selection_indices_of_cluster_members_previous[population_index][i][k]],
                                           objective_values_selection_previous_scaled
                                             [selection_indices_of_cluster_members_previous[population_index][j][q]],
                                           number_of_objectives);
              if ((distance_smallest < 0) || (distance < distance_smallest))
                distance_smallest = distance;
            }
            distance_cluster_i_previous_to_cluster_j_previous[i][j] += distance_smallest;
          }
        }
      }
    }
    /* NEW: distance between clusters is the distance between the cluster means in scaled-objective space.
        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled[population_index][j][k] = 0.0;

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            for( q = 0; q < cluster_sizes[population_index]; q++ )
              objective_means_scaled[population_index][j][k] +=
       objective_values_selection_scaled[selection_indices_of_cluster_members[population_index][j][q]][k];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled[population_index][j][k] /= (double) cluster_sizes[population_index];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled_previous[j][k] = 0.0;

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            for( q = 0; q < cluster_sizes[population_index]; q++ )
              objective_means_scaled_previous[j][k] +=
       objective_values_selection_previous_scaled[selection_indices_of_cluster_members_previous[population_index][j][q]][k];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled_previous[j][k] /= (double) cluster_sizes[population_index];

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = i; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_now_to_cluster_j_previous[i][j] = distanceEuclidean(
       objective_means_scaled[population_index][i], objective_means_scaled_previous[j], number_of_objectives );
            distance_cluster_i_now_to_cluster_j_previous[j][i] = distance_cluster_i_now_to_cluster_j_previous[i][j];
          }
        }

        distance_cluster_i_now_to_cluster_j_now = (double **) Malloc(
       number_of_mixing_components[population_index]*sizeof( double * ) ); for( i = 0; i <
       number_of_mixing_components[population_index]; i++ ) distance_cluster_i_now_to_cluster_j_now[i] = (double *)
       Malloc( number_of_mixing_components[population_index]*sizeof( double ) );

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = i; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_now_to_cluster_j_now[i][j] = 0;
            if( i != j )
            {
              distance_cluster_i_now_to_cluster_j_now[i][j] = distanceEuclidean(
       objective_means_scaled[population_index][i], objective_means_scaled[population_index][j], number_of_objectives );
              distance_cluster_i_now_to_cluster_j_now[j][i] = distance_cluster_i_now_to_cluster_j_now[i][j];
            }
          }
        }

        distance_cluster_i_previous_to_cluster_j_previous = (double **) Malloc(
       number_of_mixing_components[population_index]*sizeof( double * ) ); for( i = 0; i < number_of_mixing_componen
              for( k = 0; k < number_of_parameters; k++ )ts; i++ )
          distance_cluster_i_previous_to_cluster_j_previous[i] = (double *) Malloc(
       number_of_mixing_components[population_index]*sizeof( double ) );

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_previous_to_cluster_j_previous[i][j] = 0;
            if( i != j )
            {
              distance_cluster_i_previous_to_cluster_j_previous[i][j] = distanceEuclidean(
       objective_means_scaled_previous[i], objective_means_scaled_previous[j], number_of_objectives );
              distance_cluster_i_previou- number_of_objectivess_to_cluster_j_previous[j][i] =
       distance_cluster_i_previous_to_cluster_j_previous[i][j];
            }
          }
        }
    */
    /* NEWNEW: distance between clusters is RANDOM
        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled[population_index][j][k] = 0.0;

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            for( q = 0; q < cluster_sizes[population_index]; q++ )
              objective_means_scaled[population_index][j][k] +=
       objective_values_selection_scaled[selection_indices_of_cluster_members[population_index][j][q]][k];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled[population_index][j][k] /= (double) cluster_sizes[population_index];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled_previous[j][k] = 0.0;

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            for( q = 0; q < cluster_sizes[population_index]; q++ )
              objective_means_scaled_previous[j][k] +=
       objective_values_selection_previous_scaled[selection_indices_of_cluster_members_previous[population_index][j][q]][k];

        for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          for( k = 0; k < number_of_objectives; k++ )
            objective_means_scaled_previous[j][k] /= (double) cluster_sizes[population_index];

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = i; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_now_to_cluster_j_previous[i][j] = randomRealUniform01();
            distance_cluster_i_now_to_cluster_j_previous[j][i] = distance_cluster_i_now_to_cluster_j_previous[i][j];
          }
        }

        distance_cluster_i_now_to_cluster_j_now = (double **) Malloc(
       number_of_mixing_components[population_index]*sizeof( double * ) ); for( i = 0; i <
       number_of_mixing_components[population_index]; i++ ) distance_cluster_i_now_to_cluster_j_now[i] = (double *)
       Malloc( number_of_mixing_components[population_index]*sizeof( double ) );

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = i; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_now_to_cluster_j_now[i][j] = 0;
            if( i != j )
            {
              distance_cluster_i_now_to_cluster_j_now[i][j] = randomRealUniform01();
              distance_cluster_i_now_to_cluster_j_now[j][i] = distance_cluster_i_now_to_cluster_j_now[i][j];
            }
          }
        }

        distance_cluster_i_previous_to_cluster_j_previous = (double **) Malloc(
       number_of_mixing_components[population_index]*sizeof( double * ) ); for( i = 0; i <
       number_of_mixing_components[population_index]; i++ ) distance_cluster_i_previous_to_cluster_j_previous[i] =
       (double *) Malloc( number_of_mixing_components[population_index]*sizeof( double ) );

        for( i = 0; i < number_of_mixing_components[population_index]; i++ )
        {
          for( j = 0; j < number_of_mixing_components[population_index]; j++ )
          {
            distance_cluster_i_previous_to_cluster_j_previous[i][j] = 0;
            if( i != j )
            {
              distance_cluster_i_previous_to_cluster_j_previous[i][j] = randomRealUniform01();
              distance_cluster_i_previous_to_cluster_j_previous[j][i] =
       distance_cluster_i_previous_to_cluster_j_previous[i][j];
            }
          }
        }
    */
    ////// END DISTANCE COMPUTATION

    clusters_now_already_registered = (bool *)Malloc(number_of_mixing_components[population_index] * sizeof(bool));
    clusters_previous_already_registered = (bool *)Malloc(number_of_mixing_components[population_index] * sizeof(bool));
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
    {
      clusters_now_already_registered[i] = 0;
      clusters_previous_already_registered[i] = 0;
    }

    r_nearest_neighbours_now = (int *)Malloc((number_of_nearest_neighbours_in_registration + 1) * sizeof(int));
    r_nearest_neighbours_previous = (int *)Malloc((number_of_nearest_neighbours_in_registration + 1) * sizeof(int));
    nearest_neighbour_choice_best = (int *)Malloc((number_of_nearest_neighbours_in_registration + 1) * sizeof(int));

    number_of_clusters_left_to_register = number_of_mixing_components[population_index];
    while (number_of_clusters_left_to_register > 0)
    {
      /* Find the two clusters in the current generation that are farthest apart and haven't been registered yet */
      i_min = -1;
      j_min = -1;
      distance_largest = -1;
      for (i = 0; i < number_of_mixing_components[population_index]; i++)
      {
        if (clusters_now_already_registered[i] == 0)
        {
          for (j = 0; j < number_of_mixing_components[population_index]; j++)
          {
            if ((i != j) && (clusters_now_already_registered[j] == 0))
            {
              distance = distance_cluster_i_now_to_cluster_j_now[i][j];
              if ((distance_largest < 0) || (distance > distance_largest))
              {
                distance_largest = distance;
                i_min = i;
                j_min = j;
              }
            }
          }
        }
      }

      if (i_min == -1)
      {
        for (i = 0; i < number_of_mixing_components[population_index]; i++)
          if (clusters_now_already_registered[i] == 0)
          {
            i_min = i;
            break;
          }
      }

      /* Find the r nearest clusters of one of the two far-apart clusters that haven't been registered yet */
      sorted = mergeSort(distance_cluster_i_now_to_cluster_j_now[i_min], number_of_mixing_components[population_index]);
      j = 0;
      for (i = 0; i < number_of_mixing_components[population_index]; i++)
      {
        if (clusters_now_already_registered[sorted[i]] == 0)
        {
          r_nearest_neighbours_now[j] = sorted[i];
          clusters_now_already_registered[sorted[i]] = 1;
          j++;
        }
        if (j == number_of_nearest_neighbours_in_registration && number_of_clusters_left_to_register - j != 1)
          break;
      }
      number_of_clusters_to_register_by_permutation = j;
      free(sorted);

      /* Find the closest cluster from the previous generation */
      j_min = -1;
      distance_smallest = -1;
      for (j = 0; j < number_of_mixing_components[population_index]; j++)
      {
        if (clusters_previous_already_registered[j] == 0)
        {
          distance = distance_cluster_i_now_to_cluster_j_previous[i_min][j];
          if ((distance_smallest < 0) || (distance < distance_smallest))
          {
            distance_smallest = distance;
            j_min = j;
          }
        }
      }

      /* Find the r nearest clusters of one of the the closest cluster from the previous generation */
      sorted = mergeSort(distance_cluster_i_previous_to_cluster_j_previous[j_min],
                         number_of_mixing_components[population_index]);
      j = 0;
      for (i = 0; i < number_of_mixing_components[population_index]; i++)
      {
        if (clusters_previous_already_registered[sorted[i]] == 0)
        {
          r_nearest_neighbours_previous[j] = sorted[i];
          clusters_previous_already_registered[sorted[i]] = 1;
          j++;
        }
        if (j == number_of_clusters_to_register_by_permutation)
          break;
      }
      free(sorted);

      /* Register the r selected clusters from the current and the previous generation */
      all_cluster_permutations =
        allPermutations(number_of_clusters_to_register_by_permutation, &number_of_cluster_permutations);
      distance_smallest = -1;
      for (i = 0; i < number_of_cluster_permutations; i++)
      {
        distance = 0;
        for (j = 0; j < number_of_clusters_to_register_by_permutation; j++)
          distance +=
            distance_cluster_i_now_to_cluster_j_previous[r_nearest_neighbours_now[j]]
                                                        [r_nearest_neighbours_previous[all_cluster_permutations[i][j]]];
        if ((distance_smallest < 0) || (distance < distance_smallest))
        {
          distance_smallest = distance;
          for (j = 0; j < number_of_clusters_to_register_by_permutation; j++)
            nearest_neighbour_choice_best[j] = r_nearest_neighbours_previous[all_cluster_permutations[i][j]];
        }
      }
      for (i = 0; i < number_of_cluster_permutations; i++)
        free(all_cluster_permutations[i]);
      free(all_cluster_permutations);

      for (i = 0; i < number_of_clusters_to_register_by_permutation; i++)
      {
        selection_indices_of_cluster_members[population_index][nearest_neighbour_choice_best[i]] =
          selection_indices_of_cluster_members_before_registration[r_nearest_neighbours_now[i]];
        if (r_nearest_neighbours_now[i] >= number_of_mixing_components[population_index] - number_of_objectives)
        {
          single_objective_clusters[population_index][nearest_neighbour_choice_best[i]] =
            r_nearest_neighbours_now[i] - (number_of_mixing_components[population_index] - number_of_objectives);
          single_objective_clusters[population_index][r_nearest_neighbours_now[i]] = -1;
        }
      }

      number_of_clusters_left_to_register -= number_of_clusters_to_register_by_permutation;
    }

    free(nearest_neighbour_choice_best);
    free(r_nearest_neighbours_previous);
    free(r_nearest_neighbours_now);
    free(clusters_now_already_registered);
    free(clusters_previous_already_registered);
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      free(distance_cluster_i_previous_to_cluster_j_previous[i]);
    free(distance_cluster_i_previous_to_cluster_j_previous);
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      free(distance_cluster_i_now_to_cluster_j_now[i]);
    free(distance_cluster_i_now_to_cluster_j_now);
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      free(distance_cluster_i_now_to_cluster_j_previous[i]);
    free(distance_cluster_i_now_to_cluster_j_previous);
    free(selection_indices_of_cluster_members_before_registration);
    for (i = 0; i < selection_sizes[population_index]; i++)
      free(objective_values_selection_previous_scaled[i]);
    free(objective_values_selection_previous_scaled);
  }

  // Compute objective means
  for (j = 0; j < number_of_mixing_components[population_index]; j++)
    for (k = 0; k < number_of_objectives; k++)
      objective_means_scaled[population_index][j][k] = 0.0;

  for (j = 0; j < number_of_mixing_components[population_index]; j++)
    for (k = 0; k < number_of_objectives; k++)
      for (q = 0; q < cluster_sizes[population_index]; q++)
        objective_means_scaled[population_index][j][k] +=
          objective_values_selection_scaled[selection_indices_of_cluster_members[population_index][j][q]][k];

  for (j = 0; j < number_of_mixing_components[population_index]; j++)
    for (k = 0; k < number_of_objectives; k++)
      objective_means_scaled[population_index][j][k] /= (double)cluster_sizes[population_index];

  int **   full_rankings = (int **)Malloc(number_of_mixing_components[population_index] * sizeof(int *));
  double * distances = (double *)Malloc(selection_sizes[population_index] * sizeof(double));
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    for (j = 0; j < selection_sizes[population_index]; j++)
      distances[j] = distanceEuclidean(
        objective_values_selection_scaled[j], objective_means_scaled[population_index][i], number_of_objectives);
    full_rankings[i] = mergeSort(distances, selection_sizes[population_index]);
  }

  // Assign exactly 'cluster_size' individuals of the population to each cluster
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    num_individuals_in_cluster[population_index][i] = 0;
  for (i = 0; i < population_sizes[population_index]; i++)
    cluster_index_for_population[population_index][i] = -1;

  for (j = 0; j < cluster_sizes[population_index]; j++)
  {
    for (i = number_of_mixing_components[population_index] - 1; i >= 0; i--)
    {
      int inc = 0;
      int individual_index =
        selection_indices[population_index][selection_indices_of_cluster_members[population_index][i][j]];
      while (cluster_index_for_population[population_index][individual_index] != -1)
      {
        individual_index = selection_indices[population_index][full_rankings[i][j + inc]];
        inc++;
      }
      cluster_index_for_population[population_index][individual_index] = i;
      num_individuals_in_cluster[population_index][i]++;
    }
  }
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    free(full_rankings[i]);
  free(full_rankings);
  free(distances);

  double * objective_values_scaled = (double *)Malloc(number_of_objectives * sizeof(double));
  int      index_smallest;
  for (i = 0; i < population_sizes[population_index]; i++)
  {
    if (cluster_index_for_population[population_index][i] != -1)
      continue;

    for (j = 0; j < number_of_objectives; j++)
      objective_values_scaled[j] =
        populations[population_index][i]->objective_values[j] / objective_ranges[population_index][j];

    distance_smallest = -1;
    index_smallest = -1;
    for (j = 0; j < number_of_mixing_components[population_index]; j++)
    {
      distance =
        distanceEuclidean(objective_values_scaled, objective_means_scaled[population_index][j], number_of_objectives);
      if ((distance_smallest < 0) || (distance < distance_smallest))
      {
        index_smallest = j;
        distance_smallest = distance;
      }
    }
    cluster_index_for_population[population_index][i] = index_smallest;
    num_individuals_in_cluster[population_index][index_smallest]++;
  }
  free(objective_values_scaled);

  /* Elitism, must be done here, before possibly changing the focus of each cluster to an elitist solution */
  copyBestSolutionsToPopulation(population_index, objective_values_selection_scaled);

  /* Estimate the parameters */
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    /* Means */
    if (number_of_generations[population_index] > 0)
    {
      for (j = 0; j < number_of_parameters; j++)
        mean_vectors_previous[population_index][i][j] = mean_vectors[population_index][i][j];
    }

    for (j = 0; j < number_of_parameters; j++)
    {
      mean_vectors[population_index][i][j] = 0.0;

      for (k = 0; k < cluster_sizes[population_index]; k++)
        mean_vectors[population_index][i][j] +=
          selection[population_index][selection_indices_of_cluster_members[population_index][i][k]]->parameters[j];

      mean_vectors[population_index][i][j] /= (double)cluster_sizes[population_index];
    }
  }

  if (learn_linkage_tree)
  {
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
    {
      estimateFullCovarianceMatrixML(population_index, i);

      linkage_model[population_index][i] = learnLinkageTreeRVGOMEA(population_index, i);

      for (j = 0; j < number_of_parameters; j++)
        free(full_covariance_matrix[population_index][i][j]);
      free(full_covariance_matrix[population_index][i]);
    }

    initializeCovarianceMatrices(population_index);

    if (number_of_generations[population_index] == 0)
      initializeDistributionMultipliers(population_index);
  }

  if (m_PartialEvaluations)
  {
    if (learn_linkage_tree || number_of_generations[population_index] == 0)
    {
      this->GetCostFunction()->InitFOSMapping(linkage_model[population_index][0]->sets,
                                              linkage_model[population_index][0]->set_length,
                                              linkage_model[population_index][0]->length);
    }
  }

  int    vara, varb, cluster_index;
  double cov;
  for (cluster_index = 0; cluster_index < number_of_mixing_components[population_index]; cluster_index++)
  {
    /* First do the maximum-likelihood estimate from data */
    for (i = 0; i < linkage_model[population_index][cluster_index]->length; i++)
    {
      for (j = 0; j < linkage_model[population_index][cluster_index]->set_length[i]; j++)
      {
        vara = linkage_model[population_index][cluster_index]->sets[i][j];
        for (k = j; k < linkage_model[population_index][cluster_index]->set_length[i]; k++)
        {
          varb = linkage_model[population_index][cluster_index]->sets[i][k];
          cov = 0.0;

          for (m = 0; m < cluster_sizes[population_index]; m++)
            cov +=
              (selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][m]]
                 ->parameters[vara] -
               mean_vectors[population_index][cluster_index][vara]) *
              (selection[population_index][selection_indices_of_cluster_members[population_index][cluster_index][m]]
                 ->parameters[varb] -
               mean_vectors[population_index][cluster_index][varb]);

          cov /= (double)cluster_sizes[population_index];
          decomposed_covariance_matrices[population_index][cluster_index][i][j][k] = cov;
          decomposed_covariance_matrices[population_index][cluster_index][i][k][j] = cov;
        }
      }
    }
  }

  free(distances_to_cluster);
  free(k_means_cluster_sizes);
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    free(selection_indices_of_cluster_members_k_means[i]);
  free(selection_indices_of_cluster_members_k_means);
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    free(objective_means_scaled_new[i]);
  free(objective_means_scaled_new);
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    free(objective_means_scaled_previous[i]);
  free(objective_means_scaled_previous);
  for (i = 0; i < selection_sizes[population_index]; i++)
    free(objective_values_selection_scaled[i]);
  free(objective_values_selection_scaled);
  free(selection_indices_of_leaders);
}

/**
 * Elitism: copies at most 1/k*tau*n solutions per cluster
 * from the elitist archive.
 */
void
MOGOMEAOptimizer::copyBestSolutionsToPopulation(int population_index, double ** objective_values_selection_scaled)
{
  int i, j, j_min, k, index, **elitist_archive_indices_per_cluster, so_index,
    *number_of_elitist_archive_indices_per_cluster, max, *diverse_indices, skipped;
  double distance, distance_smallest, *objective_values_scaled, **points;

  number_of_elitist_archive_indices_per_cluster =
    (int *)Malloc(number_of_mixing_components[population_index] * sizeof(int));
  elitist_archive_indices_per_cluster = (int **)Malloc(number_of_mixing_components[population_index] * sizeof(int *));
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    number_of_elitist_archive_indices_per_cluster[i] = 0;
    number_of_elitist_solutions_copied[population_index][i] = 0;
    elitist_archive_indices_per_cluster[i] = (int *)Malloc(elitist_archive_size * sizeof(int));
  }
  objective_values_scaled = (double *)Malloc(number_of_objectives * sizeof(double));

  for (i = 0; i < elitist_archive_size; i++)
  {
    if (elitist_archive_indices_inactive[i])
      continue;
    for (j = 0; j < number_of_objectives; j++)
      objective_values_scaled[j] = elitist_archive[i]->objective_values[j] / objective_ranges[population_index][j];
    j_min = -1;
    distance_smallest = -1;
    for (j = 0; j < number_of_mixing_components[population_index]; j++)
    {
      distance =
        distanceEuclidean(objective_values_scaled, objective_means_scaled[population_index][j], number_of_objectives);
      if ((distance_smallest < 0) || (distance < distance_smallest))
      {
        j_min = j;
        distance_smallest = distance;
      }
    }

    elitist_archive_indices_per_cluster[j_min][number_of_elitist_archive_indices_per_cluster[j_min]] = i;
    number_of_elitist_archive_indices_per_cluster[j_min]++;
  }

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    max = (int)(tau * num_individuals_in_cluster[population_index][i]);
    skipped = 0;
    if (number_of_elitist_archive_indices_per_cluster[i] <= max)
    {
      for (j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++)
      {
        index = sorted_ranks[population_index][population_sizes[population_index] - 1 - skipped]; // BLA
        while (cluster_index_for_population[population_index][index] != i &&
               (population_sizes[population_index] - 1 - skipped) > 0)
          index = sorted_ranks[population_index][population_sizes[population_index] - 1 - (++skipped)];
        if (cluster_index_for_population[population_index][index] != i)
          break;
        so_index = single_objective_clusters[population_index][i];
        if (so_index != -1 && populations[population_index][index]->objective_values[so_index] <
                                elitist_archive[elitist_archive_indices_per_cluster[i][j]]->objective_values[so_index])
          continue;
        copyIndividual(elitist_archive[elitist_archive_indices_per_cluster[i][j]],
                       populations[population_index][index]);
        evaluateIndividual(population_index, index, -1);
        this->SavePartialEvaluation(index);
        
        populations[population_index][index]->NIS = 0;
        skipped++;
      }
      number_of_elitist_solutions_copied[population_index][i] = j;
    }
    else
    {
      points = (double **)Malloc(number_of_elitist_archive_indices_per_cluster[i] * sizeof(double *));
      for (j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++)
        points[j] = (double *)Malloc(number_of_objectives * sizeof(double));
      for (j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++)
      {
        for (k = 0; k < number_of_objectives; k++)
          points[j][k] = elitist_archive[elitist_archive_indices_per_cluster[i][j]]->objective_values[k] /
                         objective_ranges[population_index][k];
      }
      diverse_indices = greedyScatteredSubsetSelection(
        points, number_of_elitist_archive_indices_per_cluster[i], number_of_objectives, max);
      for (j = 0; j < max; j++)
      {
        index = sorted_ranks[population_index][population_sizes[population_index] - 1 - skipped]; // BLA
        while (cluster_index_for_population[population_index][index] != i &&
               (population_sizes[population_index] - 1 - skipped) > 0)
          index = sorted_ranks[population_index][population_sizes[population_index] - 1 - (++skipped)];
        if (cluster_index_for_population[population_index][index] != i)
          break;
        so_index = single_objective_clusters[population_index][i];
        if (so_index != -1 && populations[population_index][index]->objective_values[so_index] <
                                elitist_archive[elitist_archive_indices_per_cluster[i][j]]->objective_values[so_index])
          continue;
        copyIndividual(elitist_archive[elitist_archive_indices_per_cluster[i][diverse_indices[j]]],
                       populations[population_index][index]);
        evaluateIndividual(population_index, index, -1);
        this->SavePartialEvaluation(index);
        populations[population_index][index]->NIS = 0;
        skipped++;
      }
      number_of_elitist_solutions_copied[population_index][i] = j;
      free(diverse_indices);
      for (j = 0; j < number_of_elitist_archive_indices_per_cluster[i]; j++)
        free(points[j]);
      free(points);
    }
  }

  free(objective_values_scaled);
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    free(elitist_archive_indices_per_cluster[i]);
  free(elitist_archive_indices_per_cluster);
  free(number_of_elitist_archive_indices_per_cluster);
}

/**
 * Initializes the FOS
 */
void
MOGOMEAOptimizer::initializeFOS(int population_index, int cluster_index)
{
  int    i;
  FILE * file;
  FOS *  new_FOS;

  fflush(stdout);
  file = fopen("FOS.in", "r");
  if (file != NULL)
  {
    if (population_index == 0 && cluster_index == 0)
      new_FOS = readFOSFromFile(file);
    else
      new_FOS = copyFOS(linkage_model[0][0]);
  }
  else
  {
    if (static_linkage_tree)
    {
      if (population_index == 0 && cluster_index == 0)
      {
        new_FOS = learnLinkageTree(NULL);
      }
      else
        new_FOS = copyFOS(linkage_model[0][0]);
    }
    else if (bspline_marginal_tree)
    {
      new_FOS = (FOS *)Malloc(sizeof(FOS));
      new_FOS->length = number_of_parameters / m_ImageDimension;
      new_FOS->sets = (int **)Malloc(new_FOS->length * sizeof(int *));
      new_FOS->set_length = (int *)Malloc(new_FOS->length * sizeof(int));
      for (i = 0; i < new_FOS->length; i++)
      {
        new_FOS->sets[i] = (int *)Malloc(m_ImageDimension * sizeof(int));
        new_FOS->set_length[i] = m_ImageDimension;
      }
      for (i = 0; (unsigned)i < number_of_parameters; i++)
      {
        new_FOS->sets[i % new_FOS->length][i / new_FOS->length] = i;
      }
    }
    else
    {
      new_FOS = (FOS *)Malloc(sizeof(FOS));
      new_FOS->length = (int)((number_of_parameters + FOS_element_size - 1) / FOS_element_size);
      new_FOS->sets = (int **)Malloc(new_FOS->length * sizeof(int *));
      new_FOS->set_length = (int *)Malloc(new_FOS->length * sizeof(int));
      for (i = 0; i < new_FOS->length; i++)
      {
        new_FOS->sets[i] = (int *)Malloc(FOS_element_size * sizeof(int));
        new_FOS->set_length[i] = 0;
      }

      for (i = 0; i < number_of_parameters; i++)
      {
        new_FOS->sets[i / FOS_element_size][i % FOS_element_size] = i;
        new_FOS->set_length[i / FOS_element_size]++;
      }
    }
  }
  linkage_model[population_index][cluster_index] = new_FOS;
}

FOS *
MOGOMEAOptimizer::learnLinkageTreeRVGOMEA(int population_index, int cluster_index)
{
  int   i;
  FOS * new_FOS;

  new_FOS = learnLinkageTree(full_covariance_matrix[population_index][cluster_index]);
  if (learn_linkage_tree && number_of_generations[population_index] > 0)
    inheritDistributionMultipliers(new_FOS,
                                   linkage_model[population_index][cluster_index],
                                   distribution_multipliers[population_index][cluster_index]);

  if (learn_linkage_tree && number_of_generations[population_index] > 0)
  {
    for (i = 0; i < linkage_model[population_index][cluster_index]->length; i++)
      free(linkage_model[population_index][cluster_index]->sets[i]);
    free(linkage_model[population_index][cluster_index]->sets);
    free(linkage_model[population_index][cluster_index]->set_length);
    free(linkage_model[population_index][cluster_index]);
  }
  return (new_FOS);
}

void
MOGOMEAOptimizer::evaluateIndividual(int population_index, int individual_index, int FOS_index)
{
  int                   i;

  individual * ind = populations[population_index][individual_index];
  param_helper.MoveDataPointer(ind->parameters);
  MeasureType constraint_value = NumericTraits<MeasureType>::Zero;

  if (m_PartialEvaluations)
    this->GetValue(param_helper, FOS_index, individual_index, constraint_value);
  else
    this->GetValue(param_helper, constraint_value);

  for (i = 0; i < number_of_objectives; i++)
  {
    ind->objective_values[i] = this->GetValueForMetric(i);
  }
  ind->constraint_value = constraint_value;

  number_of_full_evaluations += FOS_index == -1 ? 1 : 0;
  number_of_evaluations++;

  // Enable this to check if the partial evaluations are done correctly
  // this->getValueSanityCheck(ind);
}

void
MOGOMEAOptimizer::getValueSanityCheck(individual * ind)
{
  MeasureType constraint_value = NumericTraits<MeasureType>::Zero;
  this->GetValue(param_helper, -1, 0, constraint_value);

  for (int i = 0; i < number_of_objectives; i++)
  {
    if (abs(ind->objective_values[i] - this->GetValueForMetric(i)) > 1e-6)
    {
      itkWarningMacro("getValueSanityCheck: objective value mismatch for objective "
                      << i << ". Partial: " << ind->objective_values[i] << " Full: " << this->GetValueForMetric(i)
                      << ".");
    }
  }

  if (abs(ind->constraint_value - constraint_value) > 1e-6)
  {
    itkWarningMacro("getValueSanityCheck: constraint value mismatch. Partial: " << ind->constraint_value << " Full: "
                                                                                << constraint_value << ".");
  }
}

void
MOGOMEAOptimizer::inheritDistributionMultipliers(FOS * new_FOS, FOS * prev_FOS, double * multipliers)
{
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

void
MOGOMEAOptimizer::estimateFullCovarianceMatrixML(int population_index, int cluster_index)
{
  int    i, j, k, q;
  double cov;

  i = cluster_index;
  full_covariance_matrix[population_index][i] = (double **)Malloc(number_of_parameters * sizeof(double *));
  for (k = 0; k < number_of_parameters; k++)
    full_covariance_matrix[population_index][i][k] = (double *)Malloc(number_of_parameters * sizeof(double));

  /* Covariance matrices */
  for (j = 0; j < number_of_parameters; j++)
  {
    for (q = j; q < number_of_parameters; q++)
    {
      cov = 0.0;
      for (k = 0; k < cluster_sizes[population_index]; k++)
        cov +=
          (selection[population_index][selection_indices_of_cluster_members[population_index][i][k]]->parameters[j] -
           mean_vectors[population_index][i][j]) *
          (selection[population_index][selection_indices_of_cluster_members[population_index][i][k]]->parameters[q] -
           mean_vectors[population_index][i][q]);
      cov /= (double)cluster_sizes[population_index];

      full_covariance_matrix[population_index][i][j][q] = cov;
      full_covariance_matrix[population_index][i][q][j] = cov;
    }
  }
}

void
MOGOMEAOptimizer::evaluateCompletePopulation(int population_index)
{
  int i;
  for (i = 0; i < population_sizes[population_index]; i++)
    evaluateIndividual(population_index, i, -1);
  this->SavePartialEvaluation(i);
}

/**
 * Applies the distribution multipliers.
 */
void
MOGOMEAOptimizer::applyDistributionMultipliers(int population_index)
{
  int i, j, k, m;

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    for (j = 0; j < linkage_model[population_index][i]->length; j++)
      for (k = 0; k < linkage_model[population_index][i]->set_length[j]; k++)
        for (m = 0; m < linkage_model[population_index][i]->set_length[j]; m++)
          decomposed_covariance_matrices[population_index][i][j][k][m] *=
            distribution_multipliers[population_index][i][j];
}


/**
 * Generates new solutions by sampling the mixture distribution.
 */
void
MOGOMEAOptimizer::generateAndEvaluateNewSolutionsToFillPopulationAndUpdateElitistArchive(int population_index)
{
  bool cluster_failure, all_multipliers_leq_one, *generational_improvement, any_improvement, *is_improved_by_AMS;
  int  i, j, k, m, oj, c, *order;

  if (m_PartialEvaluations && (number_of_generations[population_index] + 1) % 50 == 0)
    evaluateCompletePopulation(population_index);

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    computeParametersForSampling(population_index, i);

  generational_improvement = (bool *)Malloc(population_sizes[population_index] * sizeof(bool));

  for (i = 0; i < population_sizes[population_index]; i++)
    generational_improvement[i] = 0;

  for (k = 0; k < number_of_mixing_components[population_index]; k++)
  {
    order = randomPermutation(linkage_model[population_index][k]->length);
    for (m = 0; m < linkage_model[population_index][k]->length; m++)
    {
      samples_current_cluster = 0;
      oj = order[m];

      for (i = 0; i < population_sizes[population_index]; i++)
      {
        if (cluster_index_for_population[population_index][i] != k)
          continue;
        if (generateNewSolutionFromFOSElement(population_index, k, oj, i))
          generational_improvement[i] = 1;
        samples_current_cluster++;
      }

      adaptDistributionMultipliers(population_index, k, oj);
      for (i = 0; i < population_sizes[population_index]; i++)
        if (cluster_index_for_population[population_index][i] == k && generational_improvement[i])
          updateElitistArchive(populations[population_index][i]);
    }
    free(order);

    c = 0;
    if (number_of_generations[population_index] > 0)
    {
      is_improved_by_AMS = (bool *)Malloc(population_sizes[population_index] * sizeof(bool));
      for (i = 0; i < population_sizes[population_index]; i++)
        is_improved_by_AMS[i] = 0;
      for (i = 0; i < population_sizes[population_index]; i++)
      {
        if (cluster_index_for_population[population_index][i] != k)
          continue;
        is_improved_by_AMS[i] = applyAMS(population_index, i, k);
        generational_improvement[i] |= is_improved_by_AMS[i];

        c++;
        if (c >= 0.5 * tau * num_individuals_in_cluster[population_index][k])
          break;
      }
      c = 0;
      for (i = 0; i < population_sizes[population_index]; i++)
      {
        if (cluster_index_for_population[population_index][i] != k)
          continue;
        if (is_improved_by_AMS[i])
          updateElitistArchive(populations[population_index][i]);
        c++;
        if (c >= 0.5 * tau * num_individuals_in_cluster[population_index][k])
          break;
      }
      free(is_improved_by_AMS);
    }
  }

  for (i = 0; i < population_sizes[population_index]; i++)
  {
    if (generational_improvement[i])
      populations[population_index][i]->NIS = 0;
    else
      populations[population_index][i]->NIS++;
  }

  // Forced Improvements
  if (use_forced_improvement)
  {
    for (i = 0; i < population_sizes[population_index]; i++)
    {
      if (populations[population_index][i]->NIS > maximum_no_improvement_stretch)
        applyForcedImprovements(population_index, i, &(generational_improvement[i]));
    }
  }

  cluster_failure = 1;
  for (i = 0; i < number_of_mixing_components[population_index]; i++)
    for (j = 0; j < linkage_model[population_index][i]->length; j++)
      if (distribution_multipliers[population_index][i][j] > 1.0)
      {
        cluster_failure = 0;
        break;
      }

  if (cluster_failure)
    no_improvement_stretch[population_index]++;

  any_improvement = 0;
  for (i = 0; i < population_sizes[population_index]; i++)
  {
    if (generational_improvement[i])
    {
      any_improvement = 1;
      break;
    }
  }

  if (any_improvement)
    no_improvement_stretch[population_index] = 0;
  else
  {
    all_multipliers_leq_one = 1;
    for (i = 0; i < number_of_mixing_components[population_index]; i++)
      for (m = 0; m < linkage_model[population_index][i]->length; m++)
        if (distribution_multipliers[population_index][i][m] > 1.0)
        {
          all_multipliers_leq_one = 0;
          break;
        }

    if (all_multipliers_leq_one)
      no_improvement_stretch[population_index]++;
  }

  free(generational_improvement);
}

bool
MOGOMEAOptimizer::applyAMS(int population_index, int individual_index, int cluster_index)
{
  bool   improvement;
  double delta_AMS, *solution_backup;
  int    m;

  individual * ind_backup;
  ind_backup = initializeIndividual();

  delta_AMS = 2.0;
  improvement = 0;
  solution_backup = (double *)Malloc(number_of_parameters * sizeof(double));

  copyIndividual(populations[population_index][individual_index], ind_backup);
  for (m = 0; m < number_of_parameters; m++)
  {
    populations[population_index][individual_index]->parameters[m] +=
      delta_AMS *
      (mean_vectors[population_index][cluster_index][m] - mean_vectors_previous[population_index][cluster_index][m]);
  }

  evaluateIndividual(population_index, individual_index, -1);
  if (solutionWasImprovedByFOSElement(population_index, cluster_index, -1, individual_index) ||
      constraintParetoDominates(populations[population_index][individual_index]->objective_values,
                                populations[population_index][individual_index]->constraint_value,
                                ind_backup->objective_values,
                                ind_backup->constraint_value))
    improvement = 1;

  if (!improvement)
  {
    copyIndividual(ind_backup, populations[population_index][individual_index]);
  }
  else
  {
    this->SavePartialEvaluation(individual_index);
  }

  free(solution_backup);
  ezilaitiniIndividual(ind_backup);
  return (improvement);
}

void
MOGOMEAOptimizer::applyForcedImprovements(int population_index, int individual_index, bool * improved)
{
  int          i, j, k, m, cluster_index, donor_index, objective_index, *order, num_indices, *indices;
  double       distance, distance_smallest, *objective_values_scaled, alpha, *FI_backup;
  individual * ind_backup;

  i = individual_index;
  populations[population_index][i]->NIS = 0;
  cluster_index = cluster_index_for_population[population_index][i];
  donor_index = 0;
  ind_backup = initializeIndividual();

  objective_values_scaled = (double *)Malloc(number_of_objectives * sizeof(double));
  for (j = 0; j < number_of_objectives; j++)
    objective_values_scaled[j] =
      populations[population_index][i]->objective_values[j] / objective_ranges[population_index][j];
  distance_smallest = 1e308;
  for (j = 0; j < elitist_archive_size; j++)
  {
    if (elitist_archive_indices_inactive[j])
      continue;
    for (k = 0; k < number_of_objectives; k++)
      objective_values_scaled[k] = elitist_archive[j]->objective_values[k] / objective_ranges[population_index][k];
    distance = distanceEuclidean(
      objective_values_scaled, objective_means_scaled[population_index][cluster_index], number_of_objectives);
    if (distance < distance_smallest)
    {
      donor_index = j;
      distance_smallest = distance;
    }
  }

  alpha = 0.5;
  while (alpha >= 0.05)
  {
    order = randomPermutation(linkage_model[population_index][cluster_index]->length);
    for (m = 0; m < linkage_model[population_index][cluster_index]->length; m++)
    {
      num_indices = linkage_model[population_index][cluster_index]->set_length[order[m]];
      indices = linkage_model[population_index][cluster_index]->sets[order[m]];

      FI_backup = (double *)Malloc(num_indices * sizeof(double));

      copyIndividualWithoutParameters(populations[population_index][i], ind_backup);

      param_helper.MoveDataPointer(populations[population_index][i]->parameters);
      this->PreloadPartialEvaluation(param_helper, order[m]);

      for (j = 0; j < num_indices; j++)
      {
        FI_backup[j] = populations[population_index][i]->parameters[indices[j]];
        populations[population_index][i]->parameters[indices[j]] =
          alpha * populations[population_index][i]->parameters[indices[j]] +
          (1.0 - alpha) * elitist_archive[donor_index]->parameters[indices[j]];
      }

      evaluateIndividual(population_index, i, order[m]);

      if (single_objective_clusters[population_index][cluster_index] != -1)
      {
        objective_index = single_objective_clusters[population_index][cluster_index];
        if (populations[population_index][i]->objective_values[objective_index] <
            ind_backup->objective_values[objective_index])
          *improved = 1;
      }
      else if (constraintParetoDominates(populations[population_index][i]->objective_values,
                                         populations[population_index][i]->constraint_value,
                                         ind_backup->objective_values,
                                         ind_backup->constraint_value))
        *improved = 1;

      if (!(*improved))
      {
        for (j = 0; j < num_indices; j++)
          populations[population_index][i]->parameters[indices[j]] = FI_backup[j];
        copyIndividualWithoutParameters(ind_backup, populations[population_index][i]);
        free(FI_backup);
      }
      else
      {
        free(FI_backup);
        break;
      }
    }
    alpha *= 0.5;

    free(order);
    if (*improved)
      break;
  }
  if (!(*improved))
  {
    copyIndividual(elitist_archive[donor_index], populations[population_index][i]);
    evaluateIndividual(population_index, i, -1);
  }
  this->SavePartialEvaluation(i);
  updateElitistArchive(populations[population_index][i]);
  ezilaitiniIndividual(ind_backup);

  free(objective_values_scaled);
}

/**
 * Computes the Cholesky-factor matrices required for sampling
 * the multivariate normal distributions in the mixture distribution.
 */
void
MOGOMEAOptimizer::computeParametersForSampling(int population_index, int cluster_index)
{
  int i;

  if (!use_univariate_FOS)
  {
    decomposed_cholesky_factors_lower_triangle[population_index][cluster_index] =
      (double ***)Malloc(linkage_model[population_index][cluster_index]->length * sizeof(double **));
    for (i = 0; i < linkage_model[population_index][cluster_index]->length; i++)
      decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][i] =
        choleskyDecomposition(decomposed_covariance_matrices[population_index][cluster_index][i],
                              linkage_model[population_index][cluster_index]->set_length[i]);
  }
}

/**
 * Generates and returns a single new solution by drawing
 * a sample for the variables in the selected FOS elementmax_clus
 * and inserting this into the population.
 */
double *
MOGOMEAOptimizer::generateNewPartialSolutionFromFOSElement(int population_index, int cluster_index, int FOS_index)
{
  int     i, num_indices, *indices;
  double *result, *z;

  num_indices = linkage_model[population_index][cluster_index]->set_length[FOS_index];
  indices = linkage_model[population_index][cluster_index]->sets[FOS_index];

  z = (double *)Malloc(num_indices * sizeof(double));

  for (i = 0; i < num_indices; i++)
    z[i] = random1DNormalUnit();

  if (use_univariate_FOS)
  {
    result = (double *)Malloc(1 * sizeof(double));
    result[0] = z[0] * sqrt(decomposed_covariance_matrices[population_index][cluster_index][FOS_index][0][0]) +
                mean_vectors[population_index][cluster_index][indices[0]];
  }
  else
  {
    result =
      matrixVectorMultiplication(decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][FOS_index],
                                 z,
                                 num_indices,
                                 num_indices);

    for (i = 0; i < num_indices; i++)
      result[i] += mean_vectors[population_index][cluster_index][indices[i]];
  }

  free(z);

  return (result);
}

/**
 * Generates and returns a single new solution by drawing
 * a single sample from a specified model.
 */
bool
MOGOMEAOptimizer::generateNewSolutionFromFOSElement(int population_index,
                                                    int cluster_index,
                                                    int FOS_index,
                                                    int individual_index)
{
  int          j, m, *indices, num_indices, *touched_indices, num_touched_indices;
  double *     result, *solution_AMS, *individual_backup;
  bool         improvement;
  individual * ind_backup;
  ind_backup = initializeIndividual();

  num_indices = linkage_model[population_index][cluster_index]->set_length[FOS_index];
  indices = linkage_model[population_index][cluster_index]->sets[FOS_index];
  num_touched_indices = num_indices;
  touched_indices = indices;
  improvement = 0;

  solution_AMS = (double *)Malloc(num_indices * sizeof(double));
  individual_backup = (double *)Malloc(num_touched_indices * sizeof(double));

  param_helper.MoveDataPointer(populations[population_index][individual_index]->parameters);
  this->PreloadPartialEvaluation(param_helper, FOS_index);

  result = generateNewPartialSolutionFromFOSElement(population_index, cluster_index, FOS_index);

  for (j = 0; j < num_touched_indices; j++)
    individual_backup[j] = populations[population_index][individual_index]->parameters[touched_indices[j]];
  for (j = 0; j < num_indices; j++)
    populations[population_index][individual_index]->parameters[indices[j]] = result[j];

  copyIndividualWithoutParameters(populations[population_index][individual_index], ind_backup);

  if ((number_of_generations[population_index] > 0) &&
      (samples_current_cluster <= 0.5 * tau * num_individuals_in_cluster[population_index][cluster_index]))
  {
    for (m = 0; m < num_indices; m++)
    {
      j = indices[m];
      solution_AMS[m] = result[m] + delta_AMS * distribution_multipliers[population_index][cluster_index][FOS_index] *
                                      (mean_vectors[population_index][cluster_index][j] -
                                       mean_vectors_previous[population_index][cluster_index][j]);
    }

    for (j = 0; j < num_indices; j++)
      populations[population_index][individual_index]->parameters[indices[j]] = solution_AMS[j];
  }

  evaluateIndividual(population_index, individual_index, FOS_index);

  if (solutionWasImprovedByFOSElement(population_index, cluster_index, FOS_index, individual_index) ||
      constraintParetoDominates(populations[population_index][individual_index]->objective_values,
                                populations[population_index][individual_index]->constraint_value,
                                ind_backup->objective_values,
                                ind_backup->constraint_value))
  {
    improvement = 1;
  }

  if (!improvement)
  {
    for (j = 0; j < num_touched_indices; j++)
      populations[population_index][individual_index]->parameters[touched_indices[j]] = individual_backup[j];
    copyIndividualWithoutParameters(ind_backup, populations[population_index][individual_index]);
  }
  else
  {
    this->SavePartialEvaluation(individual_index);
  }

  free(solution_AMS);
  free(individual_backup);
  free(result);

  ezilaitiniIndividual(ind_backup);
  return (improvement);
}

/**
 * Adapts the distribution multipliers according to
 * the SDR-AVS mechanism.
 */
void
MOGOMEAOptimizer::adaptDistributionMultipliers(int population_index, int cluster_index, int FOS_index)
{
  bool   improvementForFOSElement;
  double st_dev_ratio;

  improvementForFOSElement =
    generationalImprovementForOneClusterForFOSElement(population_index, cluster_index, FOS_index, &st_dev_ratio);

  if (improvementForFOSElement)
  {
    no_improvement_stretch[population_index] = 0;

    if (distribution_multipliers[population_index][cluster_index][FOS_index] < 1.0)
      distribution_multipliers[population_index][cluster_index][FOS_index] = 1.0;

    if (st_dev_ratio > st_dev_ratio_threshold)
      distribution_multipliers[population_index][cluster_index][FOS_index] *= distribution_multiplier_increase;
  }
  else
  {
    if ((distribution_multipliers[population_index][cluster_index][FOS_index] > 1.0) ||
        (no_improvement_stretch[population_index] >= maximum_no_improvement_stretch))
      distribution_multipliers[population_index][cluster_index][FOS_index] *= distribution_multiplier_decrease;

    if (no_improvement_stretch[population_index] < maximum_no_improvement_stretch)
    {
      if (distribution_multipliers[population_index][cluster_index][FOS_index] < 1.0)
        distribution_multipliers[population_index][cluster_index][FOS_index] = 1.0;
    }
  }
}

/**
 * Determines whether an improvement is found for a specified
 * population. Returns 1 in case of an improvement, 0 otherwise.
 * The standard-deviation ratio required by the SDR-AVS
 * mechanism is computed and returned in the pointer variable.
 */
bool
MOGOMEAOptimizer::generationalImprovementForOneClusterForFOSElement(int      population_index,
                                                                    int      cluster_index,
                                                                    int      FOS_index,
                                                                    double * st_dev_ratio)
{
  int i, number_of_improvements;

  number_of_improvements = 0;

  /* Determine st.dev. ratio */
  *st_dev_ratio = 0.0;
  for (i = 0; i < population_sizes[population_index]; i++)
  {
    if (cluster_index_for_population[population_index][i] == cluster_index)
    {
      if (solutionWasImprovedByFOSElement(population_index, cluster_index, FOS_index, i))
      {
        number_of_improvements++;
        (*st_dev_ratio) += getStDevRatioForOneClusterForFOSElement(
          population_index, cluster_index, FOS_index, populations[population_index][i]->parameters);
      }
    }
  }

  if (number_of_improvements > 0)
    (*st_dev_ratio) = (*st_dev_ratio) / number_of_improvements;

  if (number_of_improvements > 0)
    return (1);

  return (0);
}

/**
 * Computes and returns the standard-deviation-ratio
 * of a given point for a given model.
 */
double
MOGOMEAOptimizer::getStDevRatioForOneClusterForFOSElement(int      population_index,
                                                          int      cluster_index,
                                                          int      FOS_index,
                                                          double * parameters)
{
  int      i, *indices, num_indices;
  double **inverse, result, *x_min_mu, *z;

  result = 0.0;
  indices = linkage_model[population_index][cluster_index]->sets[FOS_index];
  num_indices = linkage_model[population_index][cluster_index]->set_length[FOS_index];

  x_min_mu = (double *)Malloc(num_indices * sizeof(double));

  for (i = 0; i < num_indices; i++)
    x_min_mu[i] = parameters[indices[i]] - mean_vectors[population_index][cluster_index][indices[i]];

  if (use_univariate_FOS)
  {
    result = fabs(x_min_mu[0] / sqrt(decomposed_covariance_matrices[population_index][cluster_index][FOS_index][0][0]));
  }
  else
  {
    inverse = matrixLowerTriangularInverse(
      decomposed_cholesky_factors_lower_triangle[population_index][cluster_index][FOS_index], num_indices);
    z = matrixVectorMultiplication(inverse, x_min_mu, num_indices, num_indices);

    for (i = 0; i < num_indices; i++)
    {
      if (fabs(z[i]) > result)
        result = fabs(z[i]);
    }

    free(z);
    for (i = 0; i < num_indices; i++)
      free(inverse[i]);
    free(inverse);
  }

  free(x_min_mu);

  return (result);
}

/**
 * Returns whether a solution has the
 * hallmark of an improvement (1 for yes, 0 for no).
 */
bool
MOGOMEAOptimizer::solutionWasImprovedByFOSElement(int population_index,
                                                  int cluster_index,
                                                  int FOS_index,
                                                  int individual_index)
{
  bool result;
  int  i, j;

  result = 0;

  if (populations[population_index][individual_index]->constraint_value == 0)
  {
    for (j = 0; j < number_of_objectives; j++)
    {
      if (populations[population_index][individual_index]->objective_values[j] <
          best_objective_values_in_elitist_archive[j])
      {
        result = 1;
        break;
      }
    }
  }

  if (single_objective_clusters[population_index][cluster_index] != -1)
  {
    return (result);
  }

  if (result != 1)
  {
    result = 1;
    for (i = 0; i < elitist_archive_size; i++)
    {
      if (elitist_archive_indices_inactive[i])
        continue;
      if (constraintParetoDominates(elitist_archive[i]->objective_values,
                                    elitist_archive[i]->constraint_value,
                                    populations[population_index][individual_index]->objective_values,
                                    populations[population_index][individual_index]->constraint_value))
      {
        result = 0;
        break;
      }
      else if (!constraintParetoDominates(populations[population_index][individual_index]->objective_values,
                                          populations[population_index][individual_index]->constraint_value,
                                          elitist_archive[i]->objective_values,
                                          elitist_archive[i]->constraint_value))
      {
        if (sameObjectiveBox(elitist_archive[i]->objective_values,
                             populations[population_index][individual_index]->objective_values))
        {
          result = 0;
          break;
        }
      }
    }
  }


  return (result);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Ezilaitini -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Undoes initialization procedure by freeing up memory.
 */
void
MOGOMEAOptimizer::ezilaitini(void)
{
  int i;

  for (i = 0; i < number_of_populations; i++)
  {
    ezilaitiniDistributionMultipliers(i);

    ezilaitiniMemoryOnePopulation(i);
  }

  ezilaitiniMemory();
  is_initialized = false;
}

void
MOGOMEAOptimizer::ezilaitiniMemory(void)
{
  int       i, default_front_size;
  double ** default_front;

  for (i = 0; i < elitist_archive_capacity; i++)
    ezilaitiniIndividual(elitist_archive[i]);
  free(elitist_archive);

  free(full_covariance_matrix);
  free(population_sizes);
  free(selection_sizes);
  free(cluster_sizes);
  free(populations);
  free(ranks);
  free(sorted_ranks);
  free(objective_ranges);
  free(selection);
  free(objective_values_selection_previous);
  free(ranks_selection);
  free(number_of_mixing_components);
  free(decomposed_covariance_matrices);
  free(distribution_multipliers);
  free(decomposed_cholesky_factors_lower_triangle);
  free(mean_vectors);
  free(mean_vectors_previous);
  free(objective_means_scaled);
  free(selection_indices);
  free(selection_indices_of_cluster_members);
  free(selection_indices_of_cluster_members_previous);
  free(pop_indices_selected);
  free(single_objective_clusters);
  free(cluster_index_for_population);
  free(num_individuals_in_cluster);
  free(number_of_generations);
  free(populations_terminated);
  free(no_improvement_stretch);

  free(number_of_elitist_solutions_copied);
  free(best_objective_values_in_elitist_archive);
  free(worst_objective_values_in_elitist_archive);
  free(elitist_archive_indices_inactive);
  free(objective_discretization);

  free(linkage_model);
}

/**
 * Undoes initialization procedure by freeing up memory.
 */
void
MOGOMEAOptimizer::ezilaitiniMemoryOnePopulation(int population_index)
{
  int i, j;

  for (i = 0; i < population_sizes[population_index]; i++)
  {
    ezilaitiniIndividual(populations[population_index][i]);
  }
  free(populations[population_index]);

  for (i = 0; i < selection_sizes[population_index]; i++)
  {
    ezilaitiniIndividual(selection[population_index][i]);
    free(objective_values_selection_previous[population_index][i]);
  }
  free(selection[population_index]);
  free(objective_values_selection_previous[population_index]);

  if (!learn_linkage_tree)
  {
    ezilaitiniCovarianceMatrices(population_index);
  }

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    free(mean_vectors[population_index][i]);
    free(mean_vectors_previous[population_index][i]);
    free(objective_means_scaled[population_index][i]);

    if (selection_indices_of_cluster_members[population_index][i] != NULL)
      free(selection_indices_of_cluster_members[population_index][i]);
    if (selection_indices_of_cluster_members_previous[population_index][i] != NULL)
      free(selection_indices_of_cluster_members_previous[population_index][i]);

    if (linkage_model[population_index][i] != NULL)
    {
      for (j = 0; j < linkage_model[population_index][i]->length; j++)
        free(linkage_model[population_index][i]->sets[j]);
      free(linkage_model[population_index][i]->sets);
      free(linkage_model[population_index][i]->set_length);
      free(linkage_model[population_index][i]);
    }
  }

  if (learn_linkage_tree)
  {
    free(full_covariance_matrix[population_index]);
  }

  free(linkage_model[population_index]);
  free(ranks[population_index]);
  free(sorted_ranks[population_index]);
  free(objective_ranges[population_index]);
  free(ranks_selection[population_index]);
  free(mean_vectors[population_index]);
  free(mean_vectors_previous[population_index]);
  free(objective_means_scaled[population_index]);
  free(selection_indices[population_index]);
  free(selection_indices_of_cluster_members[population_index]);
  free(selection_indices_of_cluster_members_previous[population_index]);
  free(pop_indices_selected[population_index]);
  free(decomposed_cholesky_factors_lower_triangle[population_index]);
  free(single_objective_clusters[population_index]);
  free(cluster_index_for_population[population_index]);
  free(num_individuals_in_cluster[population_index]);
  free(number_of_elitist_solutions_copied[population_index]);
}

/**
 * Undoes initialization procedure by freeing up memory.
 */
void
MOGOMEAOptimizer::ezilaitiniDistributionMultipliers(int population_index)
{
  int i;
  if (distribution_multipliers[population_index] == NULL)
    return;

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    free(distribution_multipliers[population_index][i]);
  }
  free(distribution_multipliers[population_index]);
}

void
MOGOMEAOptimizer::ezilaitiniCovarianceMatrices(int population_index)
{
  int i, j, k;

  for (i = 0; i < number_of_mixing_components[population_index]; i++)
  {
    for (j = 0; j < linkage_model[population_index][i]->length; j++)
    {
      for (k = 0; k < linkage_model[population_index][i]->set_length[j]; k++)
        free(decomposed_covariance_matrices[population_index][i][j][k]);
      free(decomposed_covariance_matrices[population_index][i][j]);
    }
    free(decomposed_covariance_matrices[population_index][i]);
  }
  free(decomposed_covariance_matrices[population_index]);
}

/**
 * Frees memory of the Cholesky decompositions required for sampling.
 */
void
MOGOMEAOptimizer::ezilaitiniParametersForSampling(int population_index)
{
  int i, j, k;

  if (!use_univariate_FOS)
  {
    for (k = 0; k < number_of_mixing_components[population_index]; k++)
    {
      for (i = 0; i < linkage_model[population_index][k]->length; i++)
      {
        for (j = 0; j < linkage_model[population_index][k]->set_length[i]; j++)
          free(decomposed_cholesky_factors_lower_triangle[population_index][k][i][j]);
        free(decomposed_cholesky_factors_lower_triangle[population_index][k][i]);
      }
      free(decomposed_cholesky_factors_lower_triangle[population_index][k]);
    }
  }
  if (learn_linkage_tree)
  {
    ezilaitiniCovarianceMatrices(population_index);
  }
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Run -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void
MOGOMEAOptimizer::generationalStepAllPopulations()
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

  generationalStepAllPopulationsRecursiveFold(population_index_smallest, population_index_biggest);
}

void
MOGOMEAOptimizer::generationalStepAllPopulationsRecursiveFold(int population_index_smallest,
                                                              int population_index_biggest)
{
  int i, j, population_index;

  for (i = 0; i < number_of_subgenerations_per_population_factor - 1; i++)
  {
    for (population_index = population_index_smallest; population_index <= population_index_biggest; population_index++)
    {
      if (!populations_terminated[population_index])
      {
        makeSelection(population_index);

        makePopulation(population_index);

        (number_of_generations[population_index])++;

        this->UpdateMetricIterationOutput();
        this->InvokeEvent(IterationEvent());

        if (checkTerminationConditionOnePopulation(population_index))
        {
          for (j = 0; j < number_of_populations; j++)
            populations_terminated[j] = 1;
          return;
        }
      }
    }

    for (population_index = population_index_smallest; population_index < population_index_biggest; population_index++)
      generationalStepAllPopulationsRecursiveFold(population_index_smallest, population_index);
  }
}

void
MOGOMEAOptimizer::runAllPopulations(void)
{
  while (!checkTerminationConditionAllPopulations())
  {
    if (number_of_populations < maximum_number_of_populations)
    {
      initializeNewPopulation();
    }

    computeApproximationSet();

    if (write_generational_statistics)
      writeGenerationalStatisticsForOnePopulation(number_of_populations - 1);

    if (write_generational_solutions)
      writeGenerationalSolutions(0);

    freeApproximationSet();

    generationalStepAllPopulations();

    total_number_of_generations++;
  }
}

// TODO: Implement these methods
void
MOGOMEAOptimizer::StartOptimization()
{
  number_of_parameters = this->GetCostFunction()->GetNumberOfParameters();
  this->SetCurrentPosition(this->GetInitialPosition());
  m_StopCondition = Unknown;
  this->ResumeOptimization();
}

void
MOGOMEAOptimizer::ResumeOptimization()
{
  InvokeEvent(StartEvent());
  this->run();
}

void
MOGOMEAOptimizer::StopOptimization()
{
  // set positions for each mixing component for initialization of population in next resolution.
  for (int i = 0; i < number_of_mixing_components[0]; i++)
  {
    param_helper.MoveDataPointer(mean_vectors[0][i]);
    this->SetPositionForMixingComponent(i, param_helper);
  }

  InvokeEvent(EndEvent());
}

void
MOGOMEAOptimizer::PrintSelf(std::ostream & os, Indent indent) const
{
  os << indent << this->GetNameOfClass() << ":\n";
}

void
MOGOMEAOptimizer::PrintSettings() const
{
  std::ostringstream oss;
  oss << "### Settings ######################################\n";
  oss << "#\n";
  oss << "# Statistics writing every generation: " << (write_generational_statistics ? "enabled" : "disabled") << "\n";
  oss << "# Population file writing            : " << (write_generational_solutions ? "enabled" : "disabled") << "\n";
  oss << "#\n";
  oss << "###################################################\n";
  oss << "#\n";
  oss << "# Number of parameters     = " << number_of_parameters << "\n";
  oss << "# Tau                      = " << tau << "\n";
  oss << "# Population size          = " << base_population_size << "\n";
  oss << "# Number of populations    = " << maximum_number_of_populations << "\n";
  oss << "# FOS element size         = " << FOS_element_size << "\n";
  oss << "# Number of mix. com. (k)  = " << base_number_of_mixing_components << "\n";
  oss << "# Dis. mult. decreaser     = " << distribution_multiplier_decrease << "\n";
  oss << "# St. dev. rat. threshold  = " << st_dev_ratio_threshold << "\n";
  oss << "# Elitist ar. size target  = " << elitist_archive_size_target << "\n";
  oss << "# Maximum numb. of eval.   = " << maximum_number_of_evaluations << "\n";
  oss << "# Maximum numb. of gen.    = " << maximum_number_of_generations << "\n";
  oss << "# Time limit (s)           = " << maximum_number_of_seconds << "\n";
  oss << "# Random seed              = " << (long)random_seed << "\n";
  oss << "#\n";
  oss << "###################################################\n";

  elastix::log::info(oss.str());
}

MOGOMEAOptimizer::~MOGOMEAOptimizer()
{
  if (is_initialized)
  {
    ezilaitini();
  }
}

double
MOGOMEAOptimizer::ComputeAverageDistributionMultiplier() const
{
  double sum_multiplier = 0.0;
  uint   count = 0;
  for (int i = 0; i < number_of_populations; i++)
  {
    for (int j = 0; j < number_of_mixing_components[i]; j++)
    {
      for (int k = 0; k < linkage_model[i][j]->length; k++)
      {
        sum_multiplier += distribution_multipliers[i][j][k];
        count++;
      }
    }
  }
  return sum_multiplier / (double)count;
}

/**
 * Runs the MIDEA.
 */
void
MOGOMEAOptimizer::run(void)
{
  initializeRandomNumberGenerator();

  initialize();

  PrintSettings();

  runAllPopulations();

  computeApproximationSet();

  writeGenerationalStatisticsForOnePopulation(number_of_populations - 1);

  writeGenerationalSolutions(1);
  freeApproximationSet();

  ezilaitini();
}
} // namespace itk

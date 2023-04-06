/**
 *
 * MO-RV-GOMEA
 *
 * If you use this software for any purpose, please cite the most recent publication:
 * A. Bouter, N.H. Luong, C. Witteveen, T. Alderliesten, P.A.N. Bosman. 2017.
 * The Multi-Objective Real-Valued Gene-pool Optimal Mixing Evolutionary Algorithm.
 * In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2017).
 * DOI: 10.1145/3071178.3071274
 *
 * Copyright (c) 1998-2017 Peter A.N. Bosman
 *
 * The software in this file is the proprietary information of
 * Peter A.N. Bosman.
 *
 * IN NO EVENT WILL THE AUTHOR OF THIS SOFTWARE BE LIABLE TO YOU FOR ANY
 * DAMAGES, INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOST SAVINGS, OR OTHER
 * INCIDENTIAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR THE INABILITY
 * TO USE SUCH PROGRAM, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGES, OR FOR ANY CLAIM BY ANY OTHER PARTY. THE AUTHOR MAKES NO
 * REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE
 * AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY ANYONE AS A RESULT OF
 * USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 *
 * The software in this file is the result of (ongoing) scientific research.
 * The following people have been actively involved in this research over
 * the years:
 * - Peter A.N. Bosman
 * - Dirk Thierens
 * - Jörn Grahl
 * - Anton Bouter
 *
 */

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "./MO_optimization.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace MOGOMEA_UTIL
{
/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Ranking -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns 1 if x constraint-Pareto-dominates y, 0 otherwise.
 * x is not better than y unless:
 * - x and y are both infeasible and x has a smaller sum of constraint violations, or
 * - x is feasible and y is not, or
 * - x and y are both feasible and x Pareto dominates y
 */
short
constraintParetoDominates(double * objective_values_x,
                          double   constraint_value_x,
                          double * objective_values_y,
                          double   constraint_value_y)
{
  short result;

  result = 0;

  if (!use_constraints)
    result = paretoDominates(objective_values_x, objective_values_y);
  else
  {
    if (constraint_value_x > 0) /* x is infeasible */
    {
      if (constraint_value_y > 0) /* Both are infeasible */
      {
        if (constraint_value_x < constraint_value_y)
          result = 1;
      }
    }
    else /* x is feasible */
    {
      if (constraint_value_y > 0) /* x is feasible and y is not */
        result = 1;
      else /* Both are feasible */
        result = paretoDominates(objective_values_x, objective_values_y);
    }
  }

  return (result);
}

/**
 * Returns 1 if x Pareto-dominates y, 0 otherwise.
 */
short
paretoDominates(double * objective_values_x, double * objective_values_y)
{
  short strict;
  int   i, result;

  result = 1;
  strict = 0;
  for (i = 0; i < number_of_objectives; i++)
  {
    if (fabs(objective_values_x[i] - objective_values_y[i]) >= 0.00001)
    {
      if (objective_values_x[i] > objective_values_y[i])
      {
        result = 0;
        break;
      }
      if (objective_values_x[i] < objective_values_y[i])
        strict = 1;
    }
  }
  if (strict == 0 && result == 1)
    result = 0;

  return (result);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*=-=-=-=-=-=-=-=-=-=-=-= Section Elitist Archive -==-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Adapts the objective box discretization. If the numbre
 * of solutions in the elitist archive is too high or too low
 * compared to the population size, the objective box
 * discretization is adjusted accordingly. In doing so, the
 * entire elitist archive is first emptied and then refilled.
 */
void
adaptObjectiveDiscretization(void)
{
  int           i, j, k, na, nb, nc, elitist_archive_size_target_lower_bound, elitist_archive_size_target_upper_bound;
  double        low, high, *elitist_archive_objective_ranges, elitist_archive_copy_size;
  individual ** elitist_archive_copy;

  // printf("===========================================\n");
  // printf("Generation     : %d\n",number_of_generations);
  // printf("No improv. str.: %d\n",no_improvement_stretch);
  // printf("#Elitist target: %d\n",elitist_archive_size_target);
  // printf("#Elitist before: %d\n",elitist_archive_size);
  // printf("In effect      : %s\n",objective_discretization_in_effect?"true":"false");
  // printf("OBD before     : %e %e\n",objective_discretization[0],objective_discretization[1]);

  elitist_archive_size_target_lower_bound = (int)(0.75 * elitist_archive_size_target);
  elitist_archive_size_target_upper_bound = (int)(1.25 * elitist_archive_size_target);

  if (objective_discretization_in_effect && (elitist_archive_size < elitist_archive_size_target_lower_bound))
    objective_discretization_in_effect = 0;

  if (elitist_archive_size > elitist_archive_size_target_upper_bound)
  {
    objective_discretization_in_effect = 1;

    elitist_archive_objective_ranges = (double *)Malloc(number_of_objectives * sizeof(double));
    for (j = 0; j < number_of_objectives; j++)
    {
      low = elitist_archive[0]->objective_values[j];
      high = elitist_archive[0]->objective_values[j];

      for (i = 0; i < elitist_archive_size; i++)
      {
        if (elitist_archive[i]->objective_values[j] < low)
          low = elitist_archive[i]->objective_values[j];
        if (elitist_archive[i]->objective_values[j] > high)
          high = elitist_archive[i]->objective_values[j];
      }

      elitist_archive_objective_ranges[j] = high - low;
    }

    na = 1;
    nb = (int)pow(2.0, 25.0);
    for (k = 0; k < 25; k++)
    {
      elitist_archive_copy_size = elitist_archive_size;
      elitist_archive_copy = (individual **)Malloc(elitist_archive_copy_size * sizeof(individual *));
      for (i = 0; i < elitist_archive_copy_size; i++)
        elitist_archive_copy[i] = initializeIndividual();
      for (i = 0; i < elitist_archive_copy_size; i++)
      {
        copyIndividual(elitist_archive[i], elitist_archive_copy[i]);
      }

      nc = (na + nb) / 2;
      for (i = 0; i < number_of_objectives; i++)
        objective_discretization[i] = elitist_archive_objective_ranges[i] / ((double)nc);

      /* Restore the original elitist archive after the first cycle in this loop */
      if (k > 0)
      {
        elitist_archive_size = 0;
        for (i = 0; i < elitist_archive_copy_size; i++)
          addToElitistArchive(elitist_archive_copy[i], i);
      }

      /* Clear the elitist archive */
      elitist_archive_size = 0;

      /* Rebuild the elitist archive */
      for (i = 0; i < elitist_archive_copy_size; i++)
        updateElitistArchive(elitist_archive_copy[i]);

      if (elitist_archive_size <= elitist_archive_size_target_lower_bound)
        na = nc;
      else
        nb = nc;

      /* Copy the entire elitist archive */
      if (elitist_archive_copy != NULL)
      {
        for (i = 0; i < elitist_archive_copy_size; i++)
          ezilaitiniIndividual(elitist_archive_copy[i]);
        free(elitist_archive_copy);
      }
    }

    free(elitist_archive_objective_ranges);
  }
  // printf("In effect      : %s\n",objective_discretization_in_effect?"true":"false");
  // printf("OBD after      : %e %e\n",objective_discretization[0],objective_discretization[1]);
  // printf("#Elitist after : %d\n",elitist_archive_size);
  // printf("===========================================\n");
}

/**
 * Returns 1 if two solutions share the same objective box, 0 otherwise.
 */
short
sameObjectiveBox(double * objective_values_a, double * objective_values_b)
{
  int i;

  if (!objective_discretization_in_effect)
  {
    /* If the solutions are identical, they are still in the (infinitely small) same objective box. */
    for (i = 0; i < number_of_objectives; i++)
    {
      if (objective_values_a[i] != objective_values_b[i])
        return (0);
    }

    return (1);
  }

  for (i = 0; i < number_of_objectives; i++)
  {
    if (((int)(objective_values_a[i] / objective_discretization[i])) !=
        ((int)(objective_values_b[i] / objective_discretization[i])))
    {
      return (0);
    }
  }

  return (1);
}

/**
 * Updates the elitist archive by offering a new solution
 * to possibly be added to the archive. If there are no
 * solutions in the archive yet, the solution is added.
 * Otherwise, the number of times the solution is
 * dominated is computed. Solution A is always dominated
 * by solution B that is in the same domination-box if
 * B dominates A or A and B do not dominate each other.
 * If the number of times a solution is dominated, is 0,
 * the solution is added to the archive and all solutions
 * dominated by the new solution, are purged from the archive.
 */
void
updateElitistArchive(individual * ind)
{
  short is_dominated_itself, is_extreme_compared_to_archive, all_to_be_removed;
  int   i, j, *indices_dominated, number_of_solutions_dominated, insert_index;

  is_extreme_compared_to_archive = 0;
  all_to_be_removed = 1;
  insert_index = elitist_archive_size;
  if (ind->constraint_value * use_constraints == 0)
  {
    if (elitist_archive_size == 0)
    {
      is_extreme_compared_to_archive = 1;
    }
    else
    {
      for (j = 0; j < number_of_objectives; j++)
      {
        if (ind->objective_values[j] < best_objective_values_in_elitist_archive[j])
        {
          is_extreme_compared_to_archive = 1;
          break;
        }
      }
    }
  }

  if (elitist_archive_size == 0)
    addToElitistArchive(ind, insert_index);
  else
  {
    indices_dominated = (int *)Malloc(elitist_archive_size * sizeof(int));
    number_of_solutions_dominated = 0;
    is_dominated_itself = 0;
    double * bla = (double *)Malloc(number_of_objectives * sizeof(double));
    bla[0] = 0.5;
    bla[1] = 0.5;
    for (i = 0; i < elitist_archive_size; i++)
    {
      if (elitist_archive_indices_inactive[i])
      {
        if (i < insert_index)
          insert_index = i;
        continue;
      }
      all_to_be_removed = 0;
      if (constraintParetoDominates(elitist_archive[i]->objective_values,
                                    elitist_archive[i]->constraint_value,
                                    ind->objective_values,
                                    ind->constraint_value))
        is_dominated_itself = 1;
      else
      {

        if (!constraintParetoDominates(ind->objective_values,
                                       ind->constraint_value,
                                       elitist_archive[i]->objective_values,
                                       elitist_archive[i]->constraint_value))
        {
          if (sameObjectiveBox(elitist_archive[i]->objective_values, ind->objective_values) &&
              (!is_extreme_compared_to_archive))
            is_dominated_itself = 1;
        }
      }

      if (is_dominated_itself)
        break;
    }
    free(bla);

    if (all_to_be_removed)
      addToElitistArchive(ind, insert_index);
    else if (!is_dominated_itself)
    {
      for (i = 0; i < elitist_archive_size; i++)
      {
        if (elitist_archive_indices_inactive[i])
          continue;
        if (constraintParetoDominates(ind->objective_values,
                                      ind->constraint_value,
                                      elitist_archive[i]->objective_values,
                                      elitist_archive[i]->constraint_value) ||
            sameObjectiveBox(elitist_archive[i]->objective_values, ind->objective_values))
        {
          indices_dominated[number_of_solutions_dominated] = i;
          elitist_archive_indices_inactive[i] = 1;
          number_of_solutions_dominated++;
        }
      }

      if (number_of_solutions_dominated > 0)
      {
        if (ind->constraint_value == 0)
        {
          for (i = 0; i < number_of_solutions_dominated; i++)
          {
            for (j = 0; j < number_of_objectives; j++)
            {
              if (elitist_archive[indices_dominated[i]]->objective_values[j] ==
                  best_objective_values_in_elitist_archive[j])
              {
                best_objective_values_in_elitist_archive[j] = ind->objective_values[j];
              }
            }
          }
        }
        removeFromElitistArchive(indices_dominated, number_of_solutions_dominated);
      }

      addToElitistArchive(ind, insert_index);
    }

    free(indices_dominated);
  }
}

void
removeFromElitistArchive(int * indices, int number_of_indices)
{
  int i;

  for (i = 0; i < number_of_indices; i++)
    elitist_archive_indices_inactive[indices[i]] = 1;
}

/**
 * Adds a solution to the elitist archive.
 */
void
addToElitistArchive(individual * ind, int insert_index)
{
  int           i, j, elitist_archive_capacity_new, elitist_archive_size_new;
  short *       elitist_archive_indices_inactive_new;
  individual ** elitist_archive_new;

  if (insert_index >= elitist_archive_capacity)
  {
    elitist_archive_size_new = 0;
    elitist_archive_capacity_new = elitist_archive_capacity * 2 + 1;
    elitist_archive_new = (individual **)Malloc(elitist_archive_capacity_new * sizeof(individual *));
    elitist_archive_indices_inactive_new = (short *)Malloc(elitist_archive_capacity_new * sizeof(short));
    for (i = 0; i < elitist_archive_capacity_new; i++)
    {
      elitist_archive_new[i] = initializeIndividual();
      elitist_archive_indices_inactive_new[i] = 0;
    }

    for (i = 0; i < elitist_archive_size; i++)
    {
      copyIndividual(elitist_archive[i], elitist_archive_new[elitist_archive_size_new]);
      elitist_archive_size_new++;
    }

    for (i = 0; i < elitist_archive_capacity; i++)
      ezilaitiniIndividual(elitist_archive[i]);
    free(elitist_archive);
    free(elitist_archive_indices_inactive);

    elitist_archive_size = elitist_archive_size_new;
    elitist_archive_capacity = elitist_archive_capacity_new;
    elitist_archive = elitist_archive_new;
    elitist_archive_indices_inactive = elitist_archive_indices_inactive_new;
    insert_index = elitist_archive_size;
  }

  copyIndividual(ind, elitist_archive[insert_index]);

  if (insert_index == elitist_archive_size)
    elitist_archive_size++;
  elitist_archive_indices_inactive[insert_index] = 0;

  if (ind->constraint_value * use_constraints == 0)
    for (j = 0; j < number_of_objectives; j++)
    {
      if (ind->objective_values[j] < best_objective_values_in_elitist_archive[j])
        best_objective_values_in_elitist_archive[j] = ind->objective_values[j];
      if(ind->objective_values[j] > worst_objective_values_in_elitist_archive[j])
        worst_objective_values_in_elitist_archive[j] = ind->objective_values[j];
    }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Output =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/**
 * Writes (appends) statistics about the current generation to a
 * file named "statistics.dat".
 */
void
writeGenerationalStatisticsForOnePopulation(int population_index)
{
  writeGenerationalStatisticsForOnePopulationWithoutDPFSMetric(population_index);
}

/**
 * Writes (appends) statistics about the current generation
 * in case of multiple objectives when the D_{Pf->S} metric
 * cannot be computed for the selected objective functions.
 */
void
writeGenerationalStatisticsForOnePopulationWithoutDPFSMetric(int population_index)
{
  int    i;
  char   string[1000];
  short  enable_hyper_volume;
  FILE * file;

  enable_hyper_volume = 1;
  file = NULL;
  // if( total_number_of_generations == 0 && statistics_file_existed == 0 )
  if (statistics_file_existed == 0)
  {
    file = fopen((output_folder + "statistics.dat").c_str(), "w");

    sprintf(string, "# Generation  Evaluations  Time (s)");
    fputs(string, file);
    for (i = 0; i < number_of_objectives; i++)
    {
      sprintf(string, "  Best_obj[%d]", i);
      fputs(string, file);
    }
    if (enable_hyper_volume)
    {
      sprintf(string, "  Hypervolume(approx. set)");
      fputs(string, file);
    }
    sprintf(string, "  [ Pop.index  Subgen.  Pop.size  ]  #Real.Evals\n");
    fputs(string, file);
    statistics_file_existed = 1;
  }
  else
    file = fopen((output_folder + "statistics.dat").c_str(), "a");

  sprintf(string, "  %10d %11d %11.3f", total_number_of_generations, (int)number_of_evaluations, getTimer());
  fputs(string, file);

  for (i = 0; i < number_of_objectives; i++)
  {
    sprintf(string, " %11.3e", best_objective_values_in_elitist_archive[i]);
    fputs(string, file);
  }

  if (enable_hyper_volume)
  {
    last_hyper_volume = compute2DHyperVolume(approximation_set, approximation_set_size);
    sprintf(string, " %11.3e", last_hyper_volume);
    fputs(string, file);
  }

  sprintf(string,
          "  [ %4d %6d %10d ]  %11ld\n",
          population_index,
          number_of_generations[population_index],
          population_sizes[population_index],
          number_of_full_evaluations);
  fputs(string, file);

  fclose(file);
}

/**
 * Writes the solutions to various files. The filenames
 * contain the generation counter. If the flag final is
 * set (final != 0), the generation number in the filename
 * is replaced with the word "final".
 *
 * approximation_set_generation_xxxxx.dat: the approximation set (actual answer)
 * elitist_archive_generation_xxxxx.dat  : the elitist archive
 * population_generation_xxxxx.dat       : the population
 * selection_generation_xxxxx.dat        : the selected solutions
 * cluster_xxxxx_generation_xxxxx.dat    : the individual clusters
 */
void
writeGenerationalSolutions(short final)
{
  int    i, j, population_index;
  char   string[1000];
  FILE * file;

  // Approximation set
  if (final)
    sprintf(string, (output_folder + "approximation_set_generation_final.dat").c_str());
  else
    sprintf(string, (output_folder + "approximation_set_generation_%05d.dat").c_str(), total_number_of_generations);
  file = fopen(string, "w");

  for (i = 0; i < approximation_set_size; i++)
  {
    for (j = 0; j < number_of_parameters; j++)
    {
      sprintf(string, "%13e", approximation_set[i]->parameters[j]);
      fputs(string, file);
      if (j < number_of_parameters - 1)
      {
        sprintf(string, " ");
        fputs(string, file);
      }
    }
    sprintf(string, "     ");
    fputs(string, file);

    for (j = 0; j < number_of_objectives; j++)
    {
      sprintf(string, "%13e ", approximation_set[i]->objective_values[j]);
      fputs(string, file);
    }

    sprintf(string, "%13e ", approximation_set[i]->constraint_value);
    fputs(string, file);

    sprintf(string, "\n");
    fputs(string, file);
  }

  fclose(file);

  /*
// Elitist archive
if( final )
  sprintf( string, "elitist_archive_generation_final.dat" );
else
  sprintf( string, "elitist_archive_generation_%05d.dat", total_number_of_generations );
file = fopen( string, "w" );

for( i = 0; i < elitist_archive_size; i++ )
{
  for( j = 0; j < number_of_parameters; j++ )
  {
    sprintf( string, "%13e", elitist_archive[i][j] );
    fputs( string, file );
    if( j < number_of_parameters-1 )
    {
      sprintf( string, " " );
      fputs( string, file );
    }
  }
  sprintf( string, "     " );
  fputs( string, file );
  for( j = 0; j < number_of_objectives; j++ )
  {
    sprintf( string, "%13e ", elitist_archive_objective_values[i][j] );
    fputs( string, file );
  }

  sprintf( string, "%13e\n", elitist_archive_constraint_values[i] );
  fputs( string, file );
}

fclose( file );*/


  // Population
  for (population_index = 0; population_index < number_of_populations; population_index++)
  {
    if (final)
      sprintf(string, (output_folder + "population_%03d_generation_final.dat").c_str(), population_index);
    else
      sprintf(string,
              (output_folder + "population_%03d_generation_%05d.dat").c_str(),
              population_index,
              total_number_of_generations);
    file = fopen(string, "w");

    for (i = 0; i < population_sizes[population_index]; i++)
    {
      for (j = 0; j < number_of_parameters; j++)
      {
        sprintf(string, "%13e", populations[population_index][i]->parameters[j]);
        fputs(string, file);
        if (j < number_of_parameters - 1)
        {
          sprintf(string, " ");
          fputs(string, file);
        }
      }
      sprintf(string, "     ");
      fputs(string, file);
      for (j = 0; j < number_of_objectives; j++)
      {
        sprintf(string, "%13e ", populations[population_index][i]->objective_values[j]);
        fputs(string, file);
      }

      sprintf(string, "%13e\n", populations[population_index][i]->constraint_value);
      fputs(string, file);
    }

    fclose(file);
  }


  // Selection
  /*if( total_number_of_generations > 0 && !final )
{
  sprintf( string, "selection_generation_%05d.dat", total_number_of_generations );
  file = fopen( string, "w" );

  for( i = 0; i < selection_sizes[population_index]; i++ )
  {
    for( j = 0; j < number_of_parameters; j++ )
    {
      sprintf( string, "%13e", selection[i][j] );
      fputs( string, file );
      if( j < number_of_parameters-1 )
      {
        sprintf( string, " " );
        fputs( string, file );
      }
    }
    sprintf( string, "     " );
    fputs( string, file );
    for( j = 0; j < number_of_objectives; j++ )
    {
      sprintf( string, "%13e ", objective_values_selection[population_index][i][j] );
      fputs( string, file );
    }

    sprintf( string, "%13e\n", constraint_values_selection[population_index][i] );
    fputs( string, file );
  }

  fclose( file );
}

// Clusters
int k;
if( total_number_of_generations > 0 && !final )
{
  for( i = 0; i < number_of_mixing_components[0]; i++ )
  {
    //if( single_objective_clusters[population_index][i] != 0 ) continue;
    sprintf( string, "cluster_%05d_generation_%05d.dat", i, total_number_of_generations );
    file = fopen( string, "w" );

    for( j = 0; j < population_sizes[0]; j++ )
    {
      if( cluster_index_for_population[0][j] != i ) continue;
      for( k = 0; k < number_of_parameters; k++ )
      {
        sprintf( string, "%13e", populations[0][j][k] );
        fputs( string, file );
        if( k < number_of_parameters-1 )
        {
          sprintf( string, " " );
          fputs( string, file );
        }
      }
      sprintf( string, "     " );
      fputs( string, file );
      for( k = 0; k < number_of_objectives; k++ )
      {
        sprintf( string, "%13e ", objective_values[0][j][k] );
        fputs( string, file );
      }

      sprintf( string, "%13e\n", constraint_values_selection[0][j] );
      fputs( string, file );
    }

    fclose( file );
  }
}*/
}

/**
 * Computes the approximation set: the non-dominated solutions
 * in the population and the elitist archive combined.
 */
void
computeApproximationSet(void)
{
  short dominated, same_objectives;
  int   i, j, k, *indices_of_rank0, *population_indices_of_rank0, rank0_size, non_dominated_size,
    population_rank0_and_elitist_archive_size, *rank0_contribution, tot_rank0_size;
  double **population_rank0_and_elitist_archive, **population_rank0_and_elitist_archive_objective_values,
    *population_rank0_and_elitist_archive_constraint_values;

  /* First, join rank0 of the population with the elitist archive */
  indices_of_rank0 = (int *)Malloc(2 * population_sizes[number_of_populations - 1] * sizeof(int));
  population_indices_of_rank0 = (int *)Malloc(2 * population_sizes[number_of_populations - 1] * sizeof(int));
  rank0_size = 0;
  for (i = 0; i < number_of_populations; i++)
  {
    for (j = 0; j < population_sizes[i]; j++)
    {
      if (ranks[i][j] == 0)
      {
        indices_of_rank0[rank0_size] = j;
        population_indices_of_rank0[rank0_size] = i;
        rank0_size++;
      }
    }
  }

  population_rank0_and_elitist_archive_size = rank0_size + elitist_archive_size;
  population_rank0_and_elitist_archive =
    (double **)Malloc(population_rank0_and_elitist_archive_size * sizeof(double *));
  population_rank0_and_elitist_archive_objective_values =
    (double **)Malloc(population_rank0_and_elitist_archive_size * sizeof(double *));
  population_rank0_and_elitist_archive_constraint_values =
    (double *)Malloc(population_rank0_and_elitist_archive_size * sizeof(double));

  for (i = 0; i < population_rank0_and_elitist_archive_size; i++)
  {
    population_rank0_and_elitist_archive[i] = (double *)Malloc(number_of_parameters * sizeof(double));
    population_rank0_and_elitist_archive_objective_values[i] = (double *)Malloc(number_of_objectives * sizeof(double));
  }

  k = 0;
  for (i = 0; i < rank0_size; i++)
  {
    for (j = 0; j < number_of_parameters; j++)
      population_rank0_and_elitist_archive[k][j] =
        populations[population_indices_of_rank0[i]][indices_of_rank0[i]]->parameters[j];
    for (j = 0; j < number_of_objectives; j++)
      population_rank0_and_elitist_archive_objective_values[k][j] =
        populations[population_indices_of_rank0[i]][indices_of_rank0[i]]->objective_values[j];
    population_rank0_and_elitist_archive_constraint_values[k] =
      populations[population_indices_of_rank0[i]][indices_of_rank0[i]]->constraint_value;

    k++;
  }

  for (i = 0; i < elitist_archive_size; i++)
  {
    for (j = 0; j < number_of_parameters; j++)
      population_rank0_and_elitist_archive[k][j] = elitist_archive[i]->parameters[j];
    for (j = 0; j < number_of_objectives; j++)
      population_rank0_and_elitist_archive_objective_values[k][j] = elitist_archive[i]->objective_values[j];
    population_rank0_and_elitist_archive_constraint_values[k] = elitist_archive[i]->constraint_value;

    k++;
  }
  free(indices_of_rank0);

  /* Second, compute rank0 solutions amongst all solutions */
  indices_of_rank0 = (int *)Malloc(population_rank0_and_elitist_archive_size * sizeof(int));
  rank0_contribution = (int *)Malloc(number_of_populations * sizeof(int));
  for (i = 0; i < number_of_populations; i++)
    rank0_contribution[i] = 0;
  non_dominated_size = 0;
  for (i = 0; i < population_rank0_and_elitist_archive_size; i++)
  {
    dominated = 0;
    for (j = 0; j < population_rank0_and_elitist_archive_size; j++)
    {
      if (i != j)
      {
        if (constraintParetoDominates(population_rank0_and_elitist_archive_objective_values[j],
                                      population_rank0_and_elitist_archive_constraint_values[j],
                                      population_rank0_and_elitist_archive_objective_values[i],
                                      population_rank0_and_elitist_archive_constraint_values[i]))
        {
          dominated = 1;
          break;
        }
        same_objectives = 1;
        for (k = 0; k < number_of_objectives; k++)
        {
          if (population_rank0_and_elitist_archive_objective_values[i][k] !=
              population_rank0_and_elitist_archive_objective_values[j][k])
          {
            same_objectives = 0;
            break;
          }
        }
        if (same_objectives &&
            (population_rank0_and_elitist_archive_constraint_values[i] ==
             population_rank0_and_elitist_archive_constraint_values[j]) &&
            (i > j))
        {
          dominated = 1;
          if (i < rank0_size && j >= rank0_size)
            rank0_contribution[population_indices_of_rank0[i]]++;
          break;
        }
      }
    }

    if (!dominated)
    {
      if (i < rank0_size)
        rank0_contribution[population_indices_of_rank0[i]]++;
      indices_of_rank0[non_dominated_size] = i;
      non_dominated_size++;
    }
  }

  tot_rank0_size = 0;
  for (i = 0; i < number_of_populations; i++)
    tot_rank0_size += rank0_contribution[i];
  if (tot_rank0_size > 0)
  {
    for (i = 0; i < number_of_populations - 1; i++)
    {
      if (((double)rank0_contribution[i]) / (double)tot_rank0_size < 0.1)
        populations_terminated[i] = 1;
      else
        break;
    }
  }

  free(rank0_contribution);

  approximation_set_size = non_dominated_size;
  approximation_set = (individual **)Malloc(approximation_set_size * sizeof(individual *));
  for (i = 0; i < approximation_set_size; i++)
    approximation_set[i] = initializeIndividual();

  for (i = 0; i < non_dominated_size; i++)
  {
    for (j = 0; j < number_of_parameters; j++)
      approximation_set[i]->parameters[j] = population_rank0_and_elitist_archive[indices_of_rank0[i]][j];
    for (j = 0; j < number_of_objectives; j++)
      approximation_set[i]->objective_values[j] =
        population_rank0_and_elitist_archive_objective_values[indices_of_rank0[i]][j];
    approximation_set[i]->constraint_value =
      population_rank0_and_elitist_archive_constraint_values[indices_of_rank0[i]];
  }

  free(indices_of_rank0);
  free(population_indices_of_rank0);
  for (i = 0; i < population_rank0_and_elitist_archive_size; i++)
  {
    free(population_rank0_and_elitist_archive[i]);
    free(population_rank0_and_elitist_archive_objective_values[i]);
  }
  free(population_rank0_and_elitist_archive);
  free(population_rank0_and_elitist_archive_objective_values);
  free(population_rank0_and_elitist_archive_constraint_values);
}

/**
 * Frees the memory allocated for the approximation set.
 * The memory is only needed for reporting the current
 * answer (Pareto set), not for the internal workings
 * of the algorithm.
 */
void
freeApproximationSet(void)
{
  int i;

  for (i = 0; i < approximation_set_size; i++)
    ezilaitiniIndividual(approximation_set[i]);
  free(approximation_set);
}

double
compute2DHyperVolume(individual ** pareto_front, int population_size)
{
  int           i, n, *sorted;
  double        max_0, max_1, *obj_0, area;
  static double REFERENCE_MULTIPLIER = 1.1;

  n = population_size;
  max_0 = worst_objective_values_in_elitist_archive[0] * REFERENCE_MULTIPLIER;
  max_1 = worst_objective_values_in_elitist_archive[1] * REFERENCE_MULTIPLIER;
  obj_0 = (double *)Malloc(n * sizeof(double));
  for (i = 0; i < n; i++)
    obj_0[i] = pareto_front[i]->objective_values[0];
  sorted = mergeSort(obj_0, n);

  area = (max_0 - fmin(max_0, obj_0[sorted[n - 1]])) *
         (max_1 - fmin(max_1, pareto_front[sorted[n - 1]]->objective_values[1]));
  for (i = n - 2; i >= 0; i--)
    area += (fmin(max_0, obj_0[sorted[i + 1]]) - fmin(max_0, obj_0[sorted[i]])) *
            (max_1 - fmin(max_1, pareto_front[sorted[i]]->objective_values[1]));

  free(obj_0);
  free(sorted);

  return area;
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Individuals -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
individual *
initializeIndividual(void)
{
  individual * new_individual;

  new_individual = (individual *)Malloc(sizeof(individual));
  new_individual->parameters = (double *)Malloc(number_of_parameters * sizeof(double));
  new_individual->objective_values = (double *)Malloc(number_of_objectives * sizeof(double));
  new_individual->constraint_value = 0;
  new_individual->NIS = 0;
  new_individual->parameter_sum = 0;
  return (new_individual);
}

void
ezilaitiniIndividual(individual * ind)
{
  free(ind->objective_values);
  free(ind->parameters);
  free(ind);
}

void
copyIndividual(individual * source, individual * destination)
{
  int i;
  for (i = 0; i < number_of_parameters; i++)
    destination->parameters[i] = source->parameters[i];
  for (i = 0; i < number_of_objectives; i++)
    destination->objective_values[i] = source->objective_values[i];
  destination->constraint_value = source->constraint_value;
  destination->parameter_sum = source->parameter_sum;
}

void
copyIndividualWithoutParameters(individual * source, individual * destination)
{
  int i;
  for (i = 0; i < number_of_objectives; i++)
    destination->objective_values[i] = source->objective_values[i];
  destination->constraint_value = source->constraint_value;
  destination->parameter_sum = source->parameter_sum;
}
} // namespace MOGOMEA_UTIL

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

#pragma once

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "./FOS.h"
#include "./Tools.h"
#include "./Optimization.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace MOGOMEA_UTIL
{

typedef struct individual
{
  double * parameters;
  double * objective_values;
  double   constraint_value;
  int      NIS;

  double parameter_sum;
} individual;

/*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
short
constraintParetoDominates(double * objective_values_x,
                          double   constraint_value_x,
                          double * objective_values_y,
                          double   constraint_value_y);
short
paretoDominates(double * objective_values_x, double * objective_values_y);
void
updateElitistArchive(individual * ind);
void
removeFromElitistArchive(int * indices, int number_of_indices);
void
addToElitistArchive(individual * ind, int insert_index);
void
adaptObjectiveDiscretization(void);
short
sameObjectiveBox(double * objective_values_a, double * objective_values_b);
void
writeGenerationalStatisticsForOnePopulation(int population_index);
void
writeGenerationalStatisticsForOnePopulationWithoutDPFSMetric(int population_index);
void
writeGenerationalSolutions(short final);
void
computeApproximationSet(void);
void
freeApproximationSet(void);
double
compute2DHyperVolume(individual ** pareto_front, int population_size);
individual *
initializeIndividual(void);
void
ezilaitiniIndividual(individual * ind);
void
copyIndividual(individual * source, individual * destination);
void
copyIndividualWithoutParameters(individual * source, individual * destination);
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
inline int number_of_objectives, current_population_index,
  approximation_set_size; /* Number of solutions in the final answer (the approximation set). */
inline bool  use_constraints;
inline long  number_of_full_evaluations;
inline short statistics_file_existed;
inline short objective_discretization_in_effect, /* Whether the objective space is currently being discretized for the
                                             elitist archive. */
  *elitist_archive_indices_inactive;             /* Elitist archive solutions flagged for removal. */
inline int elitist_archive_size,                 /* Number of solutions in the elitist archive. */
  elitist_archive_size_target,                   /* The lower bound of the targeted size of the elitist archive. */
  elitist_archive_capacity;                      /* Current memory allocation to elitist archive. */
inline double *
  best_objective_values_in_elitist_archive, /* The best objective values in the archive in the individual objectives. */
  *objective_discretization, /* The length of the objective discretization in each dimension (for the elitist archive).
                              */
  **ranks;                   /* Ranks of all solutions in all populations. */
inline individual ***populations, /* The population containing the solutions. */
  ***selection,                   /* Selected solutions, one for each population. */
  **elitist_archive,              /* Archive of elitist solutions. */
  **approximation_set;            /* Set of non-dominated solutions from all populations and the elitist archive. */
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
} // namespace MOGOMEA_UTIL

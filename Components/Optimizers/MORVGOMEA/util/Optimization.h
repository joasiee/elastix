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
#include "./Tools.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace MOGOMEA_UTIL
{

/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
inline bool *populations_terminated, /* Which populations have been terminated. */
  evaluations_for_statistics_hit,     /* Can be used to write statistics after a certain number of evaluations. */
  write_generational_solutions;
inline unsigned int number_of_parameters; /* The number of parameters to be optimized. */
inline int number_of_populations, /* The number of parallel populations that initially partition the search space. */
  *number_of_generations, /* The current generation count of a subgeneration in the interleaved multi-start scheme. */
  total_number_of_generations,       /* The overarching generation count in the interleaved multi-start scheme. */
  *population_sizes;                 /* The size of the population. */
inline size_t number_of_evaluations; /* The current number of times a function evaluation was performed. */
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
} // namespace MOGOMEA_UTIL

/**
 *
 * RV-GOMEA
 *
 * If you use this software for any purpose, please cite the most recent publication:
 * A. Bouter, C. Witteveen, T. Alderliesten, P.A.N. Bosman. 2017.
 * Exploiting Linkage Information in Real-Valued Optimization with the Real-Valued
 * Gene-pool Optimal Mixing Evolutionary Algorithm. In Proceedings of the Genetic
 * and Evolutionary Computation Conference (GECCO 2017).
 * DOI: 10.1145/3071178.3071272
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
#include "Tools.h"
#include <eigen3/Eigen/Dense>
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace GOMEA
{
typedef struct FOS
{
  int    length;
  int ** sets;
  int *  set_length;
} FOS;

typedef enum
{
  None,
  EuclideanSimilarity
} BSplineStaticLinkageType;

extern int *mpm_number_of_indices, number_of_parameters, FOS_element_ub, use_univariate_FOS, learn_linkage_tree,
  static_linkage_tree, random_linkage_tree, bspline_marginal_cp, FOS_element_size;
extern BSplineStaticLinkageType static_linkage_type;
extern double ***               MI_matrices, **S_matrix, *S_vector;
extern std::vector<int>         grid_region_dimensions;

/*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
void
printFOS(FOS * fos);
FOS *
readFOSFromFile(FILE * file);
FOS *
copyFOS(FOS * f);
FOS *
learnLinkageTree(MatrixXd & covariance_matrix);
int
determineNearestNeighbour(int index, int mpm_length);
double
getSimilarity(int a, int b);
double **
computeMIMatrix(const MatrixXd & covariance_matrix, int n);
MatrixXd
computeDistanceMatrixBSplineGrid(int n);
MatrixXd
computeMIMatrixBSplineGrid(const MatrixXd & covariance_matrix, int n);
VectorXd
computeGridPosition(int index, const std::vector<int> & divisions);
MatrixXd
getStaticSimilarityMatrix(int n);
int *
matchFOSElements(FOS * new_FOS, FOS * prev_FOS);
int *
hungarianAlgorithm(int ** similarity_matrix, int dim);
void
hungarianAlgorithmAddToTree(int     x,
                            int     prevx,
                            short * S,
                            int *   prev,
                            int *   slack,
                            int *   slackx,
                            int *   lx,
                            int *   ly,
                            int **  similarity_matrix,
                            int     dim);
void
ezilaitiniFOS(FOS * lm);
void
filterFOS(FOS * input_FOS, int lb, int ub);

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
} // namespace GOMEA

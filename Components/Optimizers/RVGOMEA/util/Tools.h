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

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <eigen3/Eigen/Dense>
#include <cblas.h>

using Eigen::MatrixXd;

#define PI 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798
#define FALSE 0
#define TRUE 1

namespace GOMEA
{
void *
Malloc(long size);
double **
matrixNew(int n, int m);
int
linpackDCHDC(double a[], int lda, int p, double work[], int ipvt[]);
void
choleskyDecomposition(MatrixXd & result, MatrixXd & matrix, int n);
int
linpackDTRDI(double t[], int ldt, int n);
double **
matrixLowerTriangularInverse(double ** matrix, int n);
int *
mergeSort(double * array, int array_size);
void
mergeSortWithinBounds(double * array, int * sorted, int * tosort, int p, int q);
void
mergeSortWithinBoundsInt(int * array, int * sorted, int * tosort, int p, int q);

void
mergeSortMerge(double * array, int * sorted, int * tosort, int p, int r, int q);
int *
mergeSortInt(int * array, int array_size);
void
mergeSortMergeInt(int * array, int * sorted, int * tosort, int p, int r, int q);

int *
getRanks(double * array, int array_size);
int *
getRanksFromSorted(int * sorted, int array_size);

double
randomRealUniform01(void);
int
randomInt(int maximum);
double
random1DNormalUnit(void);
double
random1DNormalParameterized(double mean, double variance);
void
initializeRandomNumberGenerator(void);
int *
randomPermutation(int n);
int **
allPermutations(int length, int * numberOfPermutations);
int **
allPermutationsSubroutine(int from, int length, int * numberOfPermutations);

double
max(double x, double y);
double
min(double x, double y);
double
distanceEuclidean(double * solution_a, double * solution_b, int n);
double
distanceEuclidean2D(double x1, double y1, double x2, double y2);

double *
matrixVectorPartialMultiplication(double ** matrix,
                                  double *  vector,
                                  int       n0,
                                  int       number_of_elements,
                                  int *     element_indices);

extern int64_t random_seed, /* The seed used for the random-number generator. */
  random_seed_changing;     /* Internally used variable for randomly setting a random seed. */

extern double haveNextNextGaussian, /* Internally used variable for sampling the normal distribution. */
  nextNextGaussian;
}
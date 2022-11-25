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

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "./Tools.h"
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

namespace GOMEA
{
uint64_t     random_seed{ 0 };
std::mt19937 mersenne_generator;


/*-=-=-=-=-=-=-=-=-=-=-= Section Elementary Operations -=-=-=-=-=-=-=-=-=-=-*/
/**
 * Allocates memory and exits the program in case of a memory allocation failure.
 */
void *
Malloc(long size)
{
  PROFILE_FUNCTION();
  void * result;

  result = (void *)malloc(size);
  if (!result)
  {
    printf("\n");
    printf("Error while allocating memory in Malloc( %ld ), aborting program.", size);
    printf("\n");

    exit(0);
  }

  return (result);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Matrix -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void
shrunkCovariance(MatrixXd & emp_cov, double shrinkage)
{
  PROFILE_FUNCTION();
  const int    n = emp_cov.rows();
  const double mu = emp_cov.trace() / n;
  shrinkage = std::max(0.0, shrinkage);
  emp_cov = (1.0 - shrinkage) * emp_cov;
  emp_cov.diagonal().array() += shrinkage * mu;
}

void
shrunkCovarianceOAS(MatrixXd & emp_cov, int pop_size)
{
  PROFILE_FUNCTION();
  const int    n = emp_cov.rows();
  const double mu = emp_cov.trace() / n;
  const double mu2 = mu * mu;

  const double alpha = (emp_cov.array().pow(2)).mean();
  const double num = alpha + mu2;
  const double den = (1.0 + double(pop_size)) * (alpha - mu2 / n);

  const double shrinkage = den == 0.0 ? 1.0 : std::min(num / den, 1.0);
  shrunkCovariance(emp_cov, shrinkage);
}

void
getShrinkageLW(MatrixXd & X, MatrixXd & emp_cov)
{
  PROFILE_FUNCTION();
  const int    n_samples = X.rows();
  const int    n_features = X.cols();
  MatrixXd     X_2 = X.array().pow(2);
  VectorXd     emp_cov_trace = X.colwise().sum() / n_samples;
  const double mu = emp_cov_trace.sum() / n_features;
  double       delta_ = (X.transpose() * X).array().pow(2).sum() / pow(n_samples, 2);
  double       beta_ = (X_2.transpose() * X_2).sum();

  double       beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_);
  const double delta = (delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * pow(mu, 2)) / n_features;
  beta = std::min(beta, delta);
  const double shrinkage = beta == 0.0 ? 0.0 : beta / delta;

  shrunkCovariance(emp_cov, shrinkage);
}

/**
 * Computes the lower-triangle Cholesky Decomposition
 * of a square, symmetric and positive-definite matrix.
 * Subroutines from LINPACK and BLAS are used.
 */
float
choleskyDecomposition(MatrixXd & result, MatrixXd & matrix, int n)
{
  PROFILE_FUNCTION();
  const char uplo{ 'L' };
  int        info;

  result = matrix.triangularView<Eigen::Lower>();

  info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo, n, result.data(), n);
  if (info != 0) /* Matrix is not positive definite */
  {
    result.fill(0.0);
    result.diagonal() = matrix.diagonal().array().sqrt();
    return static_cast<float>(info) / static_cast<float>(n) * 100.0f;
  }
  return 100.0f;
}

void
lowerTriangularMatrixInverse(MatrixXd & A, int n)
{
  PROFILE_FUNCTION();
  const char uplo{ 'L' };
  const char diag{ 'N' };
  LAPACKE_dtrtri(LAPACK_COL_MAJOR, uplo, diag, n, A.data(), n);
}


/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Merge Sort -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Sorts an array of doubles and returns the sort-order (small to large).
 */
int *
mergeSort(double * array, int array_size)
{
  PROFILE_FUNCTION();
  int i, *sorted, *tosort;

  sorted = (int *)Malloc(array_size * sizeof(int));
  tosort = (int *)Malloc(array_size * sizeof(int));
  for (i = 0; i < array_size; i++)
    tosort[i] = i;

  if (array_size == 1)
    sorted[0] = 0;
  else
    mergeSortWithinBounds(array, sorted, tosort, 0, array_size - 1);

  free(tosort);

  return (sorted);
}

/**
 * Subroutine of merge sort, sorts the part of the array between p and q.
 */
void
mergeSortWithinBounds(double * array, int * sorted, int * tosort, int p, int q)
{
  PROFILE_FUNCTION();
  int r;

  if (p < q)
  {
    r = (p + q) / 2;
    mergeSortWithinBounds(array, sorted, tosort, p, r);
    mergeSortWithinBounds(array, sorted, tosort, r + 1, q);
    mergeSortMerge(array, sorted, tosort, p, r + 1, q);
  }
}
void
mergeSortWithinBoundsInt(int * array, int * sorted, int * tosort, int p, int q)
{
  PROFILE_FUNCTION();
  int r;

  if (p < q)
  {
    r = (p + q) / 2;
    mergeSortWithinBoundsInt(array, sorted, tosort, p, r);
    mergeSortWithinBoundsInt(array, sorted, tosort, r + 1, q);
    mergeSortMergeInt(array, sorted, tosort, p, r + 1, q);
  }
}
/**
 * Subroutine of merge sort, merges the results of two sorted parts.
 */
void
mergeSortMerge(double * array, int * sorted, int * tosort, int p, int r, int q)
{
  PROFILE_FUNCTION();
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
        if (array[tosort[i]] < array[tosort[j]])
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

int *
mergeSortInt(int * array, int array_size)
{
  PROFILE_FUNCTION();
  int i, *sorted, *tosort;

  sorted = (int *)Malloc(array_size * sizeof(int));
  tosort = (int *)Malloc(array_size * sizeof(int));
  for (i = 0; i < array_size; i++)
    tosort[i] = i;

  if (array_size == 1)
    sorted[0] = 0;
  else
    mergeSortWithinBoundsInt(array, sorted, tosort, 0, array_size - 1);

  free(tosort);

  return (sorted);
}

void
mergeSortMergeInt(int * array, int * sorted, int * tosort, int p, int r, int q)
{
  PROFILE_FUNCTION();
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
        if (array[tosort[i]] < array[tosort[j]])
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

int *
getRanks(double * array, int array_size)
{
  PROFILE_FUNCTION();
  int i, *sorted, *ranks;

  sorted = mergeSort(array, array_size);
  ranks = (int *)Malloc(array_size * sizeof(int));
  for (i = 0; i < array_size; i++)
    ranks[sorted[i]] = i;

  free(sorted);
  return (ranks);
}

int *
getRanksFromSorted(int * sorted, int array_size)
{
  PROFILE_FUNCTION();
  int i, *ranks;

  ranks = (int *)Malloc(array_size * sizeof(int));
  for (i = 0; i < array_size; i++)
    ranks[sorted[i]] = i;

  return (ranks);
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Random Numbers -=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns a random double, distributed uniformly between 0 and 1.
 */
double
randomRealUniform01(void)
{
  static std::uniform_real_distribution<double> distribution(0.0, 1.0);
  return distribution(mersenne_generator);
}

/**
 * Returns a random integer, distributed uniformly between 0 and maximum.
 */
int
randomInt(int maximum)
{
  int result;
  result = (int)(((double)maximum) * randomRealUniform01());
  return (result);
}

/**
 * Returns a random double, distributed normally with mean 0 and variance 1.
 */
double
random1DNormalUnit(void)
{
  static std::normal_distribution<double> distribution(0.0, 1.0);
  return distribution(mersenne_generator);
}

/**
 * Returns a random double, distributed normally with given mean and variance.
 */
double
random1DNormalParameterized(double mean, double variance)
{
  std::normal_distribution<double> distribution(mean, sqrt(variance));
  return distribution(mersenne_generator);
}

/**
 * Initializes the random number generator.
 */
void
initializeRandomNumberGenerator(void)
{
  if (random_seed == 0)
  {
    mersenne_generator = std::mt19937(std::random_device()());
    return;
  }
  mersenne_generator = std::mt19937{ random_seed };
}

/**
 * Returns a random compact (using integers 0,1,...,n-1) permutation
 * of length n using the Fisher-Yates shuffle.
 */
int *
randomPermutation(int n)
{
  PROFILE_FUNCTION();
  int i, j, dummy, *result;

  result = (int *)Malloc(n * sizeof(int));
  for (i = 0; i < n; i++)
    result[i] = i;

  for (i = n - 1; i > 0; i--)
  {
    j = randomInt(i + 1);
    dummy = result[j];
    result[j] = result[i];
    result[i] = dummy;
  }

  return (result);
}

/*
 * Returns all compact integer permutations of
 * a specified length, sorted in ascending
 * radix-sort order.
 */
int **
allPermutations(int length, int * numberOfPermutations)
{
  PROFILE_FUNCTION();
  int ** result;

  result = allPermutationsSubroutine(0, length, numberOfPermutations);

  return (result);
}

/*
 * Subroutine of allPermutations.
 */
int **
allPermutationsSubroutine(int from, int length, int * numberOfPermutations)
{
  PROFILE_FUNCTION();
  int i, j, k, q, **result, **smallerResult, smallerNumberOfPermutations;

  (*numberOfPermutations) = 1;
  for (i = 2; i <= length; i++)
    (*numberOfPermutations) *= i;

  result = (int **)Malloc((*numberOfPermutations) * sizeof(int *));
  for (i = 0; i < *numberOfPermutations; i++)
    result[i] = (int *)Malloc(length * sizeof(int));

  if (length == 1)
  {
    result[0][0] = from;
  }
  else
  {
    smallerResult = allPermutationsSubroutine(from + 1, length - 1, &smallerNumberOfPermutations);

    k = 0;
    for (i = from; i < from + length; i++)
    {
      for (j = 0; j < smallerNumberOfPermutations; j++)
      {
        result[k][0] = i;
        for (q = 1; q < length; q++)
          result[k][q] = smallerResult[j][q - 1] <= i ? smallerResult[j][q - 1] - 1 : smallerResult[j][q - 1];
        k++;
      }
    }

    for (i = 0; i < smallerNumberOfPermutations; i++)
      free(smallerResult[i]);
    free(smallerResult);
  }

  return (result);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/**
 * Computes the distance between two solutions a and b as
 * the Euclidean distance in parameter space.
 */
double
distanceEuclidean(double * x, double * y, int number_of_dimensions)
{
  PROFILE_FUNCTION();
  int    i;
  double value, result;

  result = 0.0;
  for (i = 0; i < number_of_dimensions; i++)
  {
    value = y[i] - x[i];
    result += value * value;
  }
  result = sqrt(result);

  return (result);
}

void
writeMatrixToFile(MatrixXd & matrix, const char * filename)
{
  std::ofstream file(filename);
  if (file.is_open())
  {
    file << matrix << "\n";
    file.close();
  }
  else
  {
    std::cout << "Error opening file " << filename << " for writing matrix." << std::endl;
  }
}

/**
 * Computes the Euclidean distance between two points.
 */
double
distanceEuclidean2D(double x1, double y1, double x2, double y2)
{
  PROFILE_FUNCTION();
  double result;

  result = (y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2);
  result = sqrt(result);

  return (result);
}

/**
 * Returns 1 if x is better than y, 0 otherwise.
 * x is not better than y unless:
 * - x and y are both infeasible and x has a smaller sum of constraint violations, or
 * - x is feasible and y is not, or
 * - x and y are both feasible and x has a smaller objective value than y
 */
short
betterFitness(double objective_value_x,
              double constraint_value_x,
              double objective_value_y,
              double constraint_value_y,
              bool   use_constraints)
{
  short result{ 0 };

  if (!use_constraints)
  {
    if (objective_value_x < objective_value_y)
      result = 1;
  }
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
      {
        if (objective_value_x < objective_value_y)
          result = 1;
      }
    }
  }

  return (result);
}
} // namespace GOMEA

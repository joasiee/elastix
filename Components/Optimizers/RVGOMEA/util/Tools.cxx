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
int64_t random_seed, random_seed_changing;
double  haveNextNextGaussian, nextNextGaussian;

/*-=-=-=-=-=-=-=-=-=-=-= Section Elementary Operations -=-=-=-=-=-=-=-=-=-=-*/
/**
 * Allocates memory and exits the program in case of a memory allocation failure.
 */
void *
Malloc(long size)
{
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
/**
 * Creates a new matrix with dimensions n x m.
 */
double **
matrixNew(int n, int m)
{
  int       i;
  double ** result;

  result = (double **)malloc(n * (sizeof(double *)));
  for (i = 0; i < n; i++)
    result[i] = (double *)malloc(m * (sizeof(double)));

  return (result);
}

/**
 * LINPACK subroutine.
 */
int
linpackDCHDC(double a[], int lda, int p, double work[], int ipvt[])
{
  int    info, j, jp, k, l, maxl, pl, pu;
  double maxdia, temp;

  pl = 1;
  pu = 0;
  info = p;
  for (k = 1; k <= p; k++)
  {
    maxdia = a[k - 1 + (k - 1) * lda];
    maxl = k;
    if (pl <= k && k < pu)
    {
      for (l = k + 1; l <= pu; l++)
      {
        if (maxdia < a[l - 1 + (l - 1) * lda])
        {
          maxdia = a[l - 1 + (l - 1) * lda];
          maxl = l;
        }
      }
    }

    if (maxdia <= 0.0)
    {
      info = k - 1;

      return (info);
    }

    if (k != maxl)
    {
      cblas_dswap(k - 1, a + 0 + (k - 1) * lda, 1, a + 0 + (maxl - 1) * lda, 1);

      a[maxl - 1 + (maxl - 1) * lda] = a[k - 1 + (k - 1) * lda];
      a[k - 1 + (k - 1) * lda] = maxdia;
      jp = ipvt[maxl - 1];
      ipvt[maxl - 1] = ipvt[k - 1];
      ipvt[k - 1] = jp;
    }
    work[k - 1] = sqrt(a[k - 1 + (k - 1) * lda]);
    a[k - 1 + (k - 1) * lda] = work[k - 1];

    for (j = k + 1; j <= p; j++)
    {
      if (k != maxl)
      {
        if (j < maxl)
        {
          temp = a[k - 1 + (j - 1) * lda];
          a[k - 1 + (j - 1) * lda] = a[j - 1 + (maxl - 1) * lda];
          a[j - 1 + (maxl - 1) * lda] = temp;
        }
        else if (maxl < j)
        {
          temp = a[k - 1 + (j - 1) * lda];
          a[k - 1 + (j - 1) * lda] = a[maxl - 1 + (j - 1) * lda];
          a[maxl - 1 + (j - 1) * lda] = temp;
        }
      }
      a[k - 1 + (j - 1) * lda] = a[k - 1 + (j - 1) * lda] / work[k - 1];
      work[j - 1] = a[k - 1 + (j - 1) * lda];
      temp = -a[k - 1 + (j - 1) * lda];

      cblas_daxpy(j - k, temp, work + k, 1, a + k + (j - 1) * lda, 1);
    }
  }

  return (info);
}


/**
 * Computes the lower-triangle Cholesky Decomposition
 * of a square, symmetric and positive-definite matrix.
 * Subroutines from LINPACK and BLAS are used.
 */
void
choleskyDecomposition(MatrixXd & result, MatrixXd & matrix, int n)
{
  int     i, j, k, info, *ipvt;
  double *work;

  work = (double *)Malloc(n * sizeof(double));
  ipvt = (int *)Malloc(n * sizeof(int));
  result.triangularView<Eigen::Upper>() = matrix.triangularView<Eigen::Upper>();

  info = linpackDCHDC(result.data(), n, n, work, ipvt);

  if (info != n) /* Matrix is not positive definite */
  {
    result.diagonal() = matrix.diagonal().array().sqrt();
  }

  free(ipvt);
  free(work);

}

/**
 * LINPACK subroutine.
 */
int
linpackDTRDI(double t[], int ldt, int n)
{
  int    j, k, info;
  double temp;

  info = 0;
  for (k = n; 1 <= k; k--)
  {
    if (t[k - 1 + (k - 1) * ldt] == 0.0)
    {
      info = k;
      break;
    }

    t[k - 1 + (k - 1) * ldt] = 1.0 / t[k - 1 + (k - 1) * ldt];
    temp = -t[k - 1 + (k - 1) * ldt];

    if (k != n)
    {
      cblas_dscal(n - k, temp, t + k + (k - 1) * ldt, 1);
    }

    for (j = 1; j <= k - 1; j++)
    {
      temp = t[k - 1 + (j - 1) * ldt];
      t[k - 1 + (j - 1) * ldt] = 0.0;
      cblas_daxpy(n - k + 1, temp, t + k - 1 + (k - 1) * ldt, 1, t + k - 1 + (j - 1) * ldt, 1);
    }
  }

  return (info);
}

/**
 * Computes the inverse of a matrix that is of
 * lower triangular form.
 */
double **
matrixLowerTriangularInverse(double ** matrix, int n)
{
  int     i, j, k;
  double *t, **result;

  t = (double *)Malloc(n * n * sizeof(double));

  k = 0;
  for (i = 0; i < n; i++)
  {
    for (j = 0; j < n; j++)
    {
      t[k] = matrix[j][i];
      k++;
    }
  }

  result = matrixNew(n, n);
  k = 0;
  for (i = 0; i < n; i++)
  {
    for (j = 0; j < n; j++)
    {
      result[j][i] = i > j ? 0.0 : t[k];
      k++;
    }
  }

  free(t);

  return (result);
}


/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Merge Sort -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Sorts an array of doubles and returns the sort-order (small to large).
 */
int *
mergeSort(double * array, int array_size)
{
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
  int64_t n26, n27;
  double  result;

  random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
  n26 = (int64_t)(random_seed_changing >> (48 - 26));
  random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
  n27 = (int64_t)(random_seed_changing >> (48 - 27));
  result = (((int64_t)n26 << 27) + n27) / ((double)(1LLU << 53));

  return (result);
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
  double v1, v2, s, multiplier, value;

  if (haveNextNextGaussian)
  {
    haveNextNextGaussian = 0;

    return (nextNextGaussian);
  }
  else
  {
    do
    {
      v1 = 2 * (randomRealUniform01()) - 1;
      v2 = 2 * (randomRealUniform01()) - 1;
      s = v1 * v1 + v2 * v2;
    } while (s >= 1);

    value = -2 * log(s) / s;
    multiplier = value <= 0.0 ? 0.0 : sqrt(value);
    nextNextGaussian = v2 * multiplier;
    haveNextNextGaussian = 1;

    return (v1 * multiplier);
  }
}

/**
 * Returns a random double, distributed normally with given mean and variance.
 */
double
random1DNormalParameterized(double mean, double variance)
{
  double result;

  result = mean + sqrt(variance) * random1DNormalUnit();

  return (result);
}

/**
 * Initializes the random number generator.
 */
void
initializeRandomNumberGenerator(void)
{
  struct timeval tv;

  while (random_seed_changing == 0)
  {
    gettimeofday(&tv, NULL);
    random_seed_changing = (int64_t)tv.tv_usec;
    random_seed_changing = (random_seed_changing / ((int)(9.99 * randomRealUniform01()) + 1)) *
                           (((int)(randomRealUniform01() * 1000000.0)) % 10);
  }

  random_seed = random_seed_changing;
}

/**
 * Returns a random compact (using integers 0,1,...,n-1) permutation
 * of length n using the Fisher-Yates shuffle.
 */
int *
randomPermutation(int n)
{
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

double
min(double x, double y)
{
  if (x <= y)
    return x;
  return y;
}

double
max(double x, double y)
{
  if (x >= y)
    return x;
  return y;
}

/**
 * Computes the distance between two solutions a and b as
 * the Euclidean distance in parameter space.
 */
double
distanceEuclidean(double * x, double * y, int number_of_dimensions)
{
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

/**
 * Computes the Euclidean distance between two points.
 */
double
distanceEuclidean2D(double x1, double y1, double x2, double y2)
{
  double result;

  result = (y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2);
  result = sqrt(result);

  return (result);
}
} // namespace GOMEA

/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef _itkAdvancedImageToImageMetric_hxx
#define _itkAdvancedImageToImageMetric_hxx

#include "itkAdvancedImageToImageMetric.h"

#include "itkAdvancedRayCastInterpolateImageFunction.h"
#include "itkComputeImageExtremaFilter.h"
#include "itkImageFullSampler.h"

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template <class TFixedImage, class TMovingImage>
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::AdvancedImageToImageMetric()
{
  /** Don't use the default gradient image as implemented by ITK.
   * It uses a Gaussian derivative, which introduces extra smoothing,
   * which may not always be desired. Also, when the derivatives are
   * computed using Gaussian filtering, the gray-values should also be
   * blurred, to have a consistent 'image model'.
   */
  this->SetComputeGradient(false);

  /** OpenMP related. Switch to on when available */
#ifdef ELASTIX_USE_OPENMP
  this->m_UseOpenMP = true;
#else
  this->m_UseOpenMP = false;
#endif

  /** Initialize the m_ThreaderMetricParameters. */
  this->m_ThreaderMetricParameters.st_Metric = this;

} // end Constructor


/**
 * ********************* SetNumberOfWorkUnits ****************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::SetNumberOfWorkUnits(ThreadIdType numberOfThreads)
{
  // Note: This is a workaround for ITK5, which renamed NumberOfThreads
  // to NumberOfWorkUnits
  Superclass::SetNumberOfWorkUnits(numberOfThreads);

#ifdef ELASTIX_USE_OPENMP
  const int nthreads = static_cast<int>(Self::GetNumberOfWorkUnits());
  omp_set_num_threads(nthreads);
#endif
} // end SetNumberOfWorkUnits()


/**
 * ********************* Initialize ****************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  /** Setup the parameters for the gray value limiters. */
  this->InitializeLimiters();

  /** Connect the image sampler */
  this->InitializeImageSampler();

  /** Check if the interpolator is a B-spline interpolator. */
  this->CheckForBSplineInterpolator();

  /** Check if the transform is an advanced transform. */
  this->CheckForAdvancedTransform();

  /** Check if the transform is a B-spline transform. */
  this->CheckForBSplineTransform();

  /** Initialize some threading related parameters. */
  if (this->m_UseMultiThread)
  {
    this->InitializeThreadingParameters();

    const auto setNumberOfWorkUnitsIfNotNull = [this](const auto bsplineInterpolator) {
      if (!bsplineInterpolator.IsNull())
      {
        bsplineInterpolator->SetNumberOfWorkUnits(this->Superclass::GetNumberOfWorkUnits());
      }
    };
    setNumberOfWorkUnitsIfNotNull(m_BSplineInterpolator);
    setNumberOfWorkUnitsIfNotNull(m_BSplineInterpolatorFloat);
  }

  this->SetNumberOfFixedImageVoxels(this->GetFixedImageRegion().GetNumberOfPixels());
} // end Initialize()


/**
 * ********************* InitializeThreadingParameters ****************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::InitializeThreadingParameters() const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   *
   * This function is only to be called at the start of each resolution.
   * Re-initialization of the potentially large vectors is performed after
   * each iteration, in the accumulate functions, in a multi-threaded fashion.
   * This has performance benefits for larger vector sizes.
   */

  /** Only resize the array of structs when needed. */
  if (this->m_GetValuePerThreadVariablesSize != numberOfThreads)
  {
    this->m_GetValuePerThreadVariables.reset(new AlignedGetValuePerThreadStruct[numberOfThreads]);
    this->m_GetValuePerThreadVariablesSize = numberOfThreads;
  }

  /** Only resize the array of structs when needed. */
  if (this->m_GetValueAndDerivativePerThreadVariablesSize != numberOfThreads)
  {
    this->m_GetValueAndDerivativePerThreadVariables.reset(
      new AlignedGetValueAndDerivativePerThreadStruct[numberOfThreads]);
    this->m_GetValueAndDerivativePerThreadVariablesSize = numberOfThreads;
  }

  /** Some initialization. */
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    this->m_GetValuePerThreadVariables[i].st_NumberOfPixelsCounted = NumericTraits<SizeValueType>::Zero;
    this->m_GetValuePerThreadVariables[i].st_Value = NumericTraits<MeasureType>::Zero;

    this->m_GetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted = NumericTraits<SizeValueType>::Zero;
    this->m_GetValueAndDerivativePerThreadVariables[i].st_Value = NumericTraits<MeasureType>::Zero;
    this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative.SetSize(this->GetNumberOfParameters());
    this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative.Fill(
      NumericTraits<DerivativeValueType>::ZeroValue());
  }

} // end InitializeThreadingParameters()


/**
 * ****************** InitializeLimiters *****************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::InitializeLimiters()
{
  /** Set up fixed limiter. */
  if (this->GetUseFixedImageLimiter())
  {
    if (this->GetFixedImageLimiter() == nullptr)
    {
      itkExceptionMacro(<< "No fixed image limiter has been set!");
    }

    using ComputeFixedImageExtremaFilterType = typename itk::ComputeImageExtremaFilter<FixedImageType>;
    typename ComputeFixedImageExtremaFilterType::Pointer computeFixedImageExtrema =
      ComputeFixedImageExtremaFilterType::New();
    computeFixedImageExtrema->SetInput(this->GetFixedImage());
    computeFixedImageExtrema->SetImageRegion(this->GetFixedImageRegion());
    if (this->m_FixedImageMask.IsNotNull())
    {
      computeFixedImageExtrema->SetUseMask(true);

      const FixedImageMaskSpatialObject2Type * fMask =
        dynamic_cast<const FixedImageMaskSpatialObject2Type *>(this->m_FixedImageMask.GetPointer());
      if (fMask)
      {
        computeFixedImageExtrema->SetImageSpatialMask(fMask);
      }
      else
      {
        computeFixedImageExtrema->SetImageMask(this->GetFixedImageMask());
      }
    }

    computeFixedImageExtrema->Update();

    this->m_FixedImageTrueMax = computeFixedImageExtrema->GetMaximum();
    this->m_FixedImageTrueMin = computeFixedImageExtrema->GetMinimum();

    this->m_FixedImageMinLimit = static_cast<FixedImageLimiterOutputType>(
      this->m_FixedImageTrueMin -
      this->m_FixedLimitRangeRatio * (this->m_FixedImageTrueMax - this->m_FixedImageTrueMin));
    this->m_FixedImageMaxLimit = static_cast<FixedImageLimiterOutputType>(
      this->m_FixedImageTrueMax +
      this->m_FixedLimitRangeRatio * (this->m_FixedImageTrueMax - this->m_FixedImageTrueMin));

    this->m_FixedImageLimiter->SetLowerThreshold(static_cast<RealType>(this->m_FixedImageTrueMin));
    this->m_FixedImageLimiter->SetUpperThreshold(static_cast<RealType>(this->m_FixedImageTrueMax));
    this->m_FixedImageLimiter->SetLowerBound(this->m_FixedImageMinLimit);
    this->m_FixedImageLimiter->SetUpperBound(this->m_FixedImageMaxLimit);

    this->m_FixedImageLimiter->Initialize();
  }

  /** Set up moving limiter. */
  if (this->GetUseMovingImageLimiter())
  {
    if (this->GetMovingImageLimiter() == nullptr)
    {
      itkExceptionMacro(<< "No moving image limiter has been set!");
    }

    using ComputeMovingImageExtremaFilterType = typename itk::ComputeImageExtremaFilter<MovingImageType>;
    typename ComputeMovingImageExtremaFilterType::Pointer computeMovingImageExtrema =
      ComputeMovingImageExtremaFilterType::New();
    computeMovingImageExtrema->SetInput(this->GetMovingImage());
    computeMovingImageExtrema->SetImageRegion(this->GetMovingImage()->GetBufferedRegion());
    if (this->m_MovingImageMask.IsNotNull())
    {
      computeMovingImageExtrema->SetUseMask(true);
      const MovingImageMaskSpatialObject2Type * mMask =
        dynamic_cast<const MovingImageMaskSpatialObject2Type *>(this->m_MovingImageMask.GetPointer());
      if (mMask)
      {
        computeMovingImageExtrema->SetImageSpatialMask(mMask);
      }
      else
      {
        computeMovingImageExtrema->SetImageMask(this->GetMovingImageMask());
      }
    }
    computeMovingImageExtrema->Update();

    this->m_MovingImageTrueMax = computeMovingImageExtrema->GetMaximum();
    this->m_MovingImageTrueMin = computeMovingImageExtrema->GetMinimum();

    this->m_MovingImageMinLimit = static_cast<MovingImageLimiterOutputType>(
      this->m_MovingImageTrueMin -
      this->m_MovingLimitRangeRatio * (this->m_MovingImageTrueMax - this->m_MovingImageTrueMin));
    this->m_MovingImageMaxLimit = static_cast<MovingImageLimiterOutputType>(
      this->m_MovingImageTrueMax +
      this->m_MovingLimitRangeRatio * (this->m_MovingImageTrueMax - this->m_MovingImageTrueMin));

    this->m_MovingImageLimiter->SetLowerThreshold(static_cast<RealType>(this->m_MovingImageTrueMin));
    this->m_MovingImageLimiter->SetUpperThreshold(static_cast<RealType>(this->m_MovingImageTrueMax));
    this->m_MovingImageLimiter->SetLowerBound(this->m_MovingImageMinLimit);
    this->m_MovingImageLimiter->SetUpperBound(this->m_MovingImageMaxLimit);

    this->m_MovingImageLimiter->Initialize();
  }

} // end InitializeLimiters()


/**
 * ********************* InitializeImageSampler ****************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::InitializeImageSampler()
{
  if (this->GetUseImageSampler())
  {
    /** Check if the ImageSampler is set. */
    if (!this->m_ImageSampler)
    {
      itkExceptionMacro(<< "ImageSampler is not present");
    }

    /** Initialize the Image Sampler. */
    this->m_ImageSampler->SetInput(this->m_FixedImage);
    this->m_ImageSampler->SetMask(this->m_FixedImageMask);
    this->m_ImageSampler->SetInputImageRegion(this->GetFixedImageRegion());
    // if (!m_PartialEvaluations)
    this->m_ImageSampler->Update();
    this->SetNumberOfFixedImageSamples(this->m_ImageSampler->GetOutput()->Size());
  }

} // end InitializeImageSampler()


/**
 * ****************** CheckForBSplineInterpolator **********************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::CheckForBSplineInterpolator()
{
  /** Check if the interpolator is of type BSplineInterpolateImageFunction,
   * or of type AdvancedLinearInterpolateImageFunction.
   * If so, we can make use of their EvaluateDerivatives methods.
   * Otherwise, we precompute the gradients using a central difference scheme,
   * and do evaluate the gradient using nearest neighbor interpolation.
   */
  this->m_InterpolatorIsBSpline = false;
  BSplineInterpolatorType * testPtr = dynamic_cast<BSplineInterpolatorType *>(this->m_Interpolator.GetPointer());
  if (testPtr)
  {
    this->m_InterpolatorIsBSpline = true;
    this->m_BSplineInterpolator = testPtr;
    itkDebugMacro("Interpolator is B-spline");
  }
  else
  {
    this->m_BSplineInterpolator = nullptr;
    itkDebugMacro("Interpolator is not B-spline");
  }

  this->m_InterpolatorIsBSplineFloat = false;
  BSplineInterpolatorFloatType * testPtr2 =
    dynamic_cast<BSplineInterpolatorFloatType *>(this->m_Interpolator.GetPointer());
  if (testPtr2)
  {
    this->m_InterpolatorIsBSplineFloat = true;
    this->m_BSplineInterpolatorFloat = testPtr2;
    itkDebugMacro("Interpolator is BSplineFloat");
  }
  else
  {
    this->m_BSplineInterpolatorFloat = nullptr;
    itkDebugMacro("Interpolator is not BSplineFloat");
  }

  this->m_InterpolatorIsReducedBSpline = false;
  ReducedBSplineInterpolatorType * testPtr3 =
    dynamic_cast<ReducedBSplineInterpolatorType *>(this->m_Interpolator.GetPointer());
  if (testPtr3)
  {
    this->m_InterpolatorIsReducedBSpline = true;
    this->m_ReducedBSplineInterpolator = testPtr3;
    itkDebugMacro("Interpolator is ReducedBSpline");
  }
  else
  {
    this->m_ReducedBSplineInterpolator = nullptr;
    itkDebugMacro("Interpolator is not ReducedBSpline");
  }

  this->m_InterpolatorIsLinear = false;
  LinearInterpolatorType * testPtr4 = dynamic_cast<LinearInterpolatorType *>(this->m_Interpolator.GetPointer());
  if (testPtr4)
  {
    this->m_InterpolatorIsLinear = true;
    this->m_LinearInterpolator = testPtr4;
  }
  else
  {
    this->m_LinearInterpolator = nullptr;
  }

  /** Don't overwrite the gradient image if GetComputeGradient() == true.
   * Otherwise we can use a forward difference derivative, or the derivative
   * provided by the B-spline interpolator.
   */
  if (!this->GetComputeGradient())
  {
    /** In addition, don't compute the moving image gradient for 2D/3D registration,
     * i.e. whenever the interpolator is a ray cast interpolator.
     * This is a bit of a hack that does not respect the setting of the boolean
     * m_ComputeGradient. By doing this, there is no way to ask no gradient
     * computation at all (to save memory).
     * The best solution would be to remove everything below this point, and to
     * override the ComputeGradient() function of ITK by computing a central
     * difference derivative. This way SetComputeGradient will enable or disable
     * the gradient computation and let derived classes choose if it needs the
     * precomputation of the gradient.
     *
     * For more details see the post about "2D/3D registration memory issue" in
     * elastix's mailing list (2 July 2012).
     */
    using RayCastInterpolatorType =
      itk::AdvancedRayCastInterpolateImageFunction<MovingImageType, CoordinateRepresentationType>;
    const bool interpolatorIsRayCast =
      dynamic_cast<RayCastInterpolatorType *>(this->m_Interpolator.GetPointer()) != nullptr;

    if (!this->m_InterpolatorIsBSpline && !this->m_InterpolatorIsBSplineFloat &&
        !this->m_InterpolatorIsReducedBSpline && !this->m_InterpolatorIsLinear && !interpolatorIsRayCast)
    {
      this->m_CentralDifferenceGradientFilter = CentralDifferenceGradientFilterType::New();
      this->m_CentralDifferenceGradientFilter->SetUseImageSpacing(true);
      this->m_CentralDifferenceGradientFilter->SetInput(this->m_MovingImage);
      this->m_CentralDifferenceGradientFilter->Update();
      this->m_GradientImage = this->m_CentralDifferenceGradientFilter->GetOutput();
    }
    else
    {
      this->m_CentralDifferenceGradientFilter = nullptr;
      this->m_GradientImage = nullptr;
    }
  }

} // end CheckForBSplineInterpolator()


/**
 * ****************** CheckForAdvancedTransform **********************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::CheckForAdvancedTransform()
{
  /** Check if the transform is of type AdvancedTransform. */
  this->m_TransformIsAdvanced = false;
  AdvancedTransformType * testPtr = dynamic_cast<AdvancedTransformType *>(this->m_Transform.GetPointer());
  if (!testPtr)
  {
    this->m_AdvancedTransform = nullptr;
    itkDebugMacro("Transform is not Advanced");
    itkExceptionMacro(<< "The AdvancedImageToImageMetric requires an AdvancedTransform");
  }
  else
  {
    this->m_TransformIsAdvanced = true;
    this->m_AdvancedTransform = testPtr;
    itkDebugMacro("Transform is Advanced");
  }

} // end CheckForAdvancedTransform()


/**
 * ****************** CheckForBSplineTransform **********************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::CheckForBSplineTransform() const
{
  /** Check if this transform is a combo transform. */
  CombinationTransformType * testPtr_combo =
    dynamic_cast<CombinationTransformType *>(this->m_AdvancedTransform.GetPointer());

  /** Check if this transform is a B-spline transform. */
  BSplineOrder1TransformType * testPtr_1 =
    dynamic_cast<BSplineOrder1TransformType *>(this->m_AdvancedTransform.GetPointer());
  BSplineOrder2TransformType * testPtr_2 =
    dynamic_cast<BSplineOrder2TransformType *>(this->m_AdvancedTransform.GetPointer());
  BSplineOrder3TransformType * testPtr_3 =
    dynamic_cast<BSplineOrder3TransformType *>(this->m_AdvancedTransform.GetPointer());

  bool transformIsBSpline = false;
  if (testPtr_1 || testPtr_2 || testPtr_3)
  {
    transformIsBSpline = true;
  }
  else if (testPtr_combo)
  {
    /** Check if the current transform is a B-spline transform. */
    const BSplineOrder1TransformType * testPtr_1b =
      dynamic_cast<const BSplineOrder1TransformType *>(testPtr_combo->GetCurrentTransform());
    const BSplineOrder2TransformType * testPtr_2b =
      dynamic_cast<const BSplineOrder2TransformType *>(testPtr_combo->GetCurrentTransform());
    const BSplineOrder3TransformType * testPtr_3b =
      dynamic_cast<const BSplineOrder3TransformType *>(testPtr_combo->GetCurrentTransform());
    if (testPtr_1b || testPtr_2b || testPtr_3b)
    {
      transformIsBSpline = true;
    }
  }

  /** Store the result. */
  this->m_TransformIsBSpline = transformIsBSpline;

} // end CheckForBSplineTransform()


/**
 * ******************* EvaluateMovingImageValueAndDerivativeWithOptionalThreadId ******************
 */

template <class TFixedImage, class TMovingImage>
template <typename... TOptionalThreadId>
bool
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::EvaluateMovingImageValueAndDerivativeWithOptionalThreadId(
  const MovingImagePointType & mappedPoint,
  RealType &                   movingImageValue,
  MovingImageDerivativeType *  gradient,
  const TOptionalThreadId... optionalThreadId) const
{
  /** Check if mapped point inside image buffer. */
  MovingImageContinuousIndexType cindex;
  this->m_Interpolator->ConvertPointToContinuousIndex(mappedPoint, cindex);
  bool sampleOk = this->m_Interpolator->IsInsideBuffer(cindex);
  if (sampleOk)
  {
    /** Compute value and possibly derivative. */
    if (gradient)
    {
      if (this->m_InterpolatorIsBSpline && !this->GetComputeGradient())
      {
        /** Compute moving image value and gradient using the B-spline kernel. */
        this->m_BSplineInterpolator->EvaluateValueAndDerivativeAtContinuousIndex(
          cindex, movingImageValue, *gradient, optionalThreadId...);
      }
      else if (this->m_InterpolatorIsBSplineFloat && !this->GetComputeGradient())
      {
        /** Compute moving image value and gradient using the B-spline kernel. */
        this->m_BSplineInterpolatorFloat->EvaluateValueAndDerivativeAtContinuousIndex(
          cindex, movingImageValue, *gradient, optionalThreadId...);
      }
      else if (this->m_InterpolatorIsReducedBSpline && !this->GetComputeGradient())
      {
        /** Compute moving image value and gradient using the B-spline kernel. */
        movingImageValue = this->m_Interpolator->EvaluateAtContinuousIndex(cindex);
        (*gradient) = this->m_ReducedBSplineInterpolator->EvaluateDerivativeAtContinuousIndex(cindex);
        // this->m_ReducedBSplineInterpolator->EvaluateValueAndDerivativeAtContinuousIndex(
        //  cindex, movingImageValue, *gradient );
      }
      else if (this->m_InterpolatorIsLinear && !this->GetComputeGradient())
      {
        /** Compute moving image value and gradient using the linear interpolator. */
        this->m_LinearInterpolator->EvaluateValueAndDerivativeAtContinuousIndex(cindex, movingImageValue, *gradient);
      }
      else
      {
        /** Get the gradient by NearestNeighboorInterpolation of the gradient image.
         * It is assumed that the gradient image is computed.
         */
        movingImageValue = this->m_Interpolator->EvaluateAtContinuousIndex(cindex);
        MovingImageIndexType index;
        for (unsigned int j = 0; j < MovingImageDimension; ++j)
        {
          index[j] = static_cast<long>(Math::Round<double>(cindex[j]));
        }
        (*gradient) = this->m_GradientImage->GetPixel(index);
      }

      /** The moving image gradient is multiplied with its scales, when requested. */
      if (this->m_UseMovingImageDerivativeScales)
      {
        if (!this->m_ScaleGradientWithRespectToMovingImageOrientation)
        {
          for (unsigned int i = 0; i < MovingImageDimension; ++i)
          {
            (*gradient)[i] *= this->m_MovingImageDerivativeScales[i];
          }
        }
        else
        {
          /** Optionally, the scales are applied with respect to the moving image orientation.
           * The above default option implicitly applies the scales with respect to the
           * orientation of the transformation axis. In some cases you may want to restrict
           * moving image motion with respect to its own axes. This is achieved below by pre
           * and post rotation by the direction cosines of the moving image.
           * First the gradient is rotated backwards to a standardized axis.
           */
          using InternalMatrixType = typename MovingImageType::DirectionType::InternalMatrixType;
          const InternalMatrixType M = this->GetMovingImage()->GetDirection().GetVnlMatrix();
          vnl_vector<double>       rotated_gradient_vnl = M.transpose() * gradient->GetVnlVector();

          /** Then scales are applied. */
          for (unsigned int i = 0; i < MovingImageDimension; ++i)
          {
            rotated_gradient_vnl[i] *= this->m_MovingImageDerivativeScales[i];
          }

          /** The scaled gradient is then rotated forwards again. */
          rotated_gradient_vnl = M * rotated_gradient_vnl;

          /** Copy the vnl version back to the original. */
          for (unsigned int i = 0; i < MovingImageDimension; ++i)
          {
            (*gradient)[i] = rotated_gradient_vnl[i];
          }
        }
      } // end if m_UseMovingImageDerivativeScales
    }   // end if gradient
    else
    {
#ifdef ELASTIX_USE_OPENMP
      if (this->m_InterpolatorIsBSpline)
      {
        const unsigned int threadid = static_cast<unsigned int>(omp_get_thread_num());
        movingImageValue = this->m_BSplineInterpolator->EvaluateAtContinuousIndex(cindex, threadid);
      }
      else
        movingImageValue = this->m_Interpolator->EvaluateAtContinuousIndex(cindex);
#else
      movingImageValue = this->m_Interpolator->EvaluateAtContinuousIndex(cindex);
#endif
    }
  } // end if sampleOk

  return sampleOk;

} // end EvaluateMovingImageValueAndDerivativeWithOptionalThreadId()


/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::EvaluateTransformJacobianInnerProduct(
  const TransformJacobianType &     jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType &                  imageJacobian) const
{
  using JacobianIteratorType = typename TransformJacobianType::const_iterator;

  /** Multiple the 1-by-dim vector movingImageDerivative with the
   * dim-by-length matrix jacobian, to get a 1-by-length vector imageJacobian.
   * An optimized route can be taken for B-spline transforms.
   */
  if (this->m_TransformIsBSpline)
  {
    // For the B-spline we know that the Jacobian is mostly empty.
    //       [ j ... j 0 ... 0 0 ... 0 ]
    // jac = [ 0 ... 0 j ... j 0 ... 0 ]
    //       [ 0 ... 0 0 ... 0 j ... j ]
    const unsigned int sizeImageJacobian = imageJacobian.GetSize();
    const unsigned int numberOfParametersPerDimension = sizeImageJacobian / FixedImageDimension;
    unsigned int       counter = 0;
    for (unsigned int dim = 0; dim < FixedImageDimension; ++dim)
    {
      const double imDeriv = movingImageDerivative[dim];
      for (unsigned int mu = 0; mu < numberOfParametersPerDimension; ++mu)
      {
        imageJacobian(counter) = jacobian(dim, counter) * imDeriv;
        ++counter;
      }
    }
  }
  else
  {
    /** Otherwise perform a full multiplication. */
    JacobianIteratorType jac = jacobian.begin();
    imageJacobian.Fill(0.0);

    for (unsigned int dim = 0; dim < FixedImageDimension; ++dim)
    {
      const double imDeriv = movingImageDerivative[dim];

      for (auto & imageJacobianElement : imageJacobian)
      {
        imageJacobianElement += (*jac) * imDeriv;
        ++jac;
      }
    }
  }

} // end EvaluateTransformJacobianInnerProduct()


/**
 * ********************** TransformPoint ************************
 */

template <class TFixedImage, class TMovingImage>
auto
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::TransformPoint(const FixedImagePointType & fixedImagePoint) const
  -> MovingImagePointType
{
  return Superclass::m_Transform->TransformPoint(fixedImagePoint);

} // end TransformPoint()


/**
 * *************** EvaluateTransformJacobian ****************
 */

template <class TFixedImage, class TMovingImage>
bool
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::EvaluateTransformJacobian(
  const FixedImagePointType &  fixedImagePoint,
  TransformJacobianType &      jacobian,
  NonZeroJacobianIndicesType & nzji) const
{
  /** Advanced transform: generic sparse Jacobian support */
  this->m_AdvancedTransform->GetJacobian(fixedImagePoint, jacobian, nzji);

  /** For future use: return whether the sample is valid */
  const bool valid = true;
  return valid;

} // end EvaluateTransformJacobian()


/**
 * ************************** IsInsideMovingMask *************************
 */

template <class TFixedImage, class TMovingImage>
bool
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::IsInsideMovingMask(const MovingImagePointType & point) const
{
  /** If a mask has been set: */
  if (this->m_MovingImageMask.IsNotNull())
  {
    return this->m_MovingImageMask->IsInsideInWorldSpace(point);
  }

  /** If no mask has been set, just return true. */
  return true;

} // end IsInsideMovingMask()

/**
 * ************************** IsInsideFixedMask *************************
 */

template <class TFixedImage, class TMovingImage>
bool
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::IsInsideFixedMask(const FixedImagePointType & point) const
{
  /** If a mask has been set: */
  if (this->m_FixedImageMask.IsNotNull())
  {
    return this->m_FixedImageMask->IsInsideInWorldSpace(point);
  }

  /** If no mask has been set, just return true. */
  return true;

} // end IsInsideFixedMask()

/**
 * ************************** IsInsideFixedMask *************************
 */

template <class TFixedImage, class TMovingImage>
double
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::PctInsideFixedMask(const FixedImageRegionType & region) const
{
  /** If a mask has been set: */
  if (this->m_FixedImageMask.IsNotNull())
  {
    FixedImageConstPointer                                  fixedImage = this->GetFixedImage();
    ImageRegionConstIteratorWithIndex<const FixedImageType> regionIterator(fixedImage, region);

    unsigned long pixelsInMask = 0L;
    while (!regionIterator.IsAtEnd())
    {
      FixedImagePointType point;
      fixedImage->TransformIndexToPhysicalPoint(regionIterator.GetIndex(), point);
      if (this->IsInsideFixedMask(point))
        ++pixelsInMask;

      ++regionIterator;
    }
    return static_cast<double>(pixelsInMask) / static_cast<double>(region.GetNumberOfPixels());
  }
  return 100.0;

} // end IsInsideFixedMask()

/**
 * ************************** TransformImageToMaskRegion *************************
 */

template <class TFixedImage, class TMovingImage>
auto
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::TransformImageToMaskRegion(
  const FixedImageRegionType & region) const -> FixedImageRegionType
{
  FixedImagePointType            lower, upper;
  FixedImageConstPointer         fixedImage = this->GetFixedImage();
  const ImageSpatialObjectType * fixedImageMask =
    dynamic_cast<const ImageSpatialObjectType *>(this->GetFixedImageMask());

  MovingImageContinuousIndexType lowerIndex = region.GetIndex();
  MovingImageContinuousIndexType upperIndex = region.GetUpperIndex();

  fixedImage->TransformContinuousIndexToPhysicalPoint(lowerIndex, lower);
  fixedImage->TransformContinuousIndexToPhysicalPoint(upperIndex, upper);
  fixedImageMask->GetImage()->TransformPhysicalPointToContinuousIndex(lower, lowerIndex);
  fixedImageMask->GetImage()->TransformPhysicalPointToContinuousIndex(upper, upperIndex);

  FixedImageRegionType maskRegion;
  FixedImageSizeType   maskSize;
  FixedImageIndexType  maskIndex;
  for (unsigned int i = 0; i < Self::FixedImageDimension; ++i)
  {
    maskSize[i] = std::ceil(upperIndex[i]) - std::floor(lowerIndex[i]) + 1;
    maskIndex[i] = static_cast<FixedImageIndexValueType>(std::floor(lowerIndex[i]));
  }
  maskRegion.SetIndex(maskIndex);
  maskRegion.SetSize(maskSize);

  return maskRegion;
}


/**
 * *********************** GetSelfHessian ***********************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::GetSelfHessian(
  const TransformParametersType & itkNotUsed(parameters),
  HessianType &                   H) const
{
  itkDebugMacro("GetSelfHessian()");

  const auto numberOfParameters = this->GetNumberOfParameters();

  /** Set identity matrix as default implementation. */
  H.set_size(numberOfParameters, numberOfParameters);
  // H.Fill(0.0);
  // H.fill_diagonal(1.0);
  for (unsigned int i = 0; i < numberOfParameters; ++i)
  {
    H(i, i) = 1.0;
  }

} // end GetSelfHessian()


/**
 * *********************** BeforeThreadedGetValueAndDerivative ***********************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::BeforeThreadedGetValueAndDerivative(
  const TransformParametersType & parameters) const
{
  /** In this function do all stuff that cannot be multi-threaded. */
  if (this->m_UseMetricSingleThreaded)
  {
    this->SetTransformParameters(parameters);
    this->GetImageSampler()->Update();

    for (auto & subfunctionSampler : m_SubfunctionSamplers)
    {
      if (subfunctionSampler)
        subfunctionSampler->Update();
    }
  }

} // end BeforeThreadedGetValueAndDerivative()


/**
 * **************** GetValueThreaderCallback *******
 */

template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::GetValueThreaderCallback(void * arg)
{
  ThreadInfoType * infoStruct = static_cast<ThreadInfoType *>(arg);
  ThreadIdType     threadID = infoStruct->WorkUnitID;

  MultiThreaderParameterType * temp = static_cast<MultiThreaderParameterType *>(infoStruct->UserData);

  temp->st_Metric->ThreadedGetValue(threadID);

  return itk::ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end GetValueThreaderCallback()


/**
 * *********************** LaunchGetValueThreaderCallback***************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::LaunchGetValueThreaderCallback() const
{
  /** Setup threader. */
  this->m_Threader->SetSingleMethod(this->GetValueThreaderCallback,
                                    const_cast<void *>(static_cast<const void *>(&this->m_ThreaderMetricParameters)));

  /** Launch. */
  this->m_Threader->SingleMethodExecute();

} // end LaunchGetValueThreaderCallback()


/**
 * **************** GetValueAndDerivativeThreaderCallback *******
 */

template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivativeThreaderCallback(void * arg)
{
  ThreadInfoType * infoStruct = static_cast<ThreadInfoType *>(arg);
  ThreadIdType     threadID = infoStruct->WorkUnitID;

  MultiThreaderParameterType * temp = static_cast<MultiThreaderParameterType *>(infoStruct->UserData);

  temp->st_Metric->ThreadedGetValueAndDerivative(threadID);

  return itk::ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end GetValueAndDerivativeThreaderCallback()


/**
 * *********************** LaunchGetValueAndDerivativeThreaderCallback***************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::LaunchGetValueAndDerivativeThreaderCallback() const
{
  /** Setup threader. */
  this->m_Threader->SetSingleMethod(this->GetValueAndDerivativeThreaderCallback,
                                    const_cast<void *>(static_cast<const void *>(&this->m_ThreaderMetricParameters)));

  /** Launch. */
  this->m_Threader->SingleMethodExecute();

} // end LaunchGetValueAndDerivativeThreaderCallback()


/**
 *********** AccumulateDerivativesThreaderCallback *************
 */

template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::AccumulateDerivativesThreaderCallback(void * arg)
{
  ThreadInfoType * infoStruct = static_cast<ThreadInfoType *>(arg);
  ThreadIdType     threadID = infoStruct->WorkUnitID;
  ThreadIdType     nrOfThreads = infoStruct->NumberOfWorkUnits;

  MultiThreaderParameterType * temp = static_cast<MultiThreaderParameterType *>(infoStruct->UserData);

  const unsigned int numPar = temp->st_Metric->GetNumberOfParameters();
  const unsigned int subSize =
    static_cast<unsigned int>(std::ceil(static_cast<double>(numPar) / static_cast<double>(nrOfThreads)));
  const unsigned int jmin = threadID * subSize;
  unsigned int       jmax = (threadID + 1) * subSize;
  jmax = (jmax > numPar) ? numPar : jmax;

  /** This thread accumulates all sub-derivatives into a single one, for the
   * range [ jmin, jmax [. Additionally, the sub-derivatives are reset.
   */
  const DerivativeValueType zero = NumericTraits<DerivativeValueType>::Zero;
  const DerivativeValueType normalization = 1.0 / temp->st_NormalizationFactor;
  for (unsigned int j = jmin; j < jmax; ++j)
  {
    DerivativeValueType tmp = zero;
    for (ThreadIdType i = 0; i < nrOfThreads; ++i)
    {
      tmp += temp->st_Metric->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative[j];

      /** Reset this variable for the next iteration. */
      temp->st_Metric->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative[j] = zero;
    }
    temp->st_DerivativePointer[j] = tmp * normalization;
  }

  return itk::ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end AccumulateDerivativesThreaderCallback()


/**
 * *********************** CheckNumberOfSamples ***********************
 */

template <class TFixedImage, class TMovingImage>
bool
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::CheckNumberOfSamples(unsigned long wanted,
                                                                            unsigned long found) const
{
  this->m_NumberOfPixelsCounted = found;
  if (found < wanted * this->GetRequiredRatioOfValidSamples())
  {
    itkExceptionMacro("Too many samples map outside moving image buffer: "
                      << found << " / " << wanted * this->GetRequiredRatioOfValidSamples() << std::endl);
  }
  return true;
} // end CheckNumberOfSamples()

template <class TFixedImage, class TMovingImage>
typename AdvancedImageToImageMetric<TFixedImage, TMovingImage>::MeasureType
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::GetValue(const TransformParametersType & parameters,
                                                                MeasureType &                   constraintValue) const
{
  MeasureType                        measure = this->GetValue(parameters);
  const BSplineOrder3TransformType * bsplinePtr = this->GetTransformAsBsplinePtr();

  Constraints constraints;
  constraints.missedPixelPct = m_PctMissedPixels;
  constraints.bsplineFolds = bsplinePtr->ComputeNumberOfFolds();

  constraintValue = this->GetConstraintValue(constraints);
  return measure;
}

template <class TFixedImage, class TMovingImage>
typename AdvancedImageToImageMetric<TFixedImage, TMovingImage>::MeasureType
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::GetValue(const TransformParametersType & parameters,
                                                                int                             fosIndex,
                                                                int                             individualIndex,
                                                                MeasureType &                   constraintValue) const
{
  MeasureType measure = NumericTraits<MeasureType>::Zero;

  IntermediateResults result = fosIndex == -1 ? this->GetValuePartial(parameters, fosIndex)
                                              : m_SolutionEvaluations[individualIndex] - m_PartialEvaluationHelper +
                                                  this->GetValuePartial(parameters, fosIndex);

  const BSplineOrder3TransformType * bsplinePtr = this->GetTransformAsBsplinePtr();
  if (bsplinePtr && constraintValue >= 0)
    result.constraints.bsplineFolds = bsplinePtr->ComputeNumberOfFolds();

  measure = this->GetValue(result);
  constraintValue = this->GetConstraintValue(result.constraints);

  m_PartialEvaluationHelper = std::move(result);

  return measure;
} // end GetValue()


template <class TFixedImage, class TMovingImage>
typename AdvancedImageToImageMetric<TFixedImage, TMovingImage>::MeasureType
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::GetConstraintValue(Constraints constraints) const
{
  MeasureType constraintValue = NumericTraits<MeasureType>::Zero;

  MeasureType missedPixelPct =
    (constraints.missedPixelPct >= m_MissedPixelConstraintThreshold) * constraints.missedPixelPct;

  return std::max(missedPixelPct, (double)constraints.bsplineFolds);
}

template <class TFixedImage, class TMovingImage>
IntermediateResults
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::GetValuePartial(const ParametersType & parameters,
                                                                       int                    fosIndex) const
{
  IntermediateResults result{ 1 };
  (void)fosIndex;
  itkWarningMacro(<< Self::GetNameOfClass() << ": Missing partial evaluations implementation.");
  result[0] = this->GetValue(parameters);
  return result;
}

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::PreloadPartialEvaluation(
  const TransformParametersType & parameters,
  int                             fosIndex) const
{
  if (m_PartialEvaluations)
    m_PartialEvaluationHelper = this->GetValuePartial(parameters, fosIndex);
}

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::SavePartialEvaluation(int individualIndex)
{
  if (m_PartialEvaluations)
    m_SolutionEvaluations[individualIndex] = std::move(m_PartialEvaluationHelper);
}

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::CopyPartialEvaluation(int toCopy, int toChange)
{
  m_SolutionEvaluations[toChange] = m_SolutionEvaluations[toCopy];
}

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::InitSubfunctionSamplers(int pop_size)
{
  int i;
  using ImageFullSamplerType = ImageFullSampler<FixedImageType>;

  this->GetRegionsForFOS();

  int           n_regions = this->m_BSplineFOSRegions.size();
  unsigned long totalSamples = 0L;

  this->m_SubfunctionSamplers.clear();
  this->m_SubfunctionSamplers.reserve(n_regions);
  const bool useMask = this->GetImageSampler()->GetUseMask();

  for (i = 0; i < n_regions; ++i)
  {
    ImageSamplerInputImageRegionType & region = this->m_BSplineFOSRegions[i];
    ImageSamplerPointer subfunctionSampler = !this->m_ImageSampler || this->PctInsideFixedMask(region) < 0.05
                                               ? ImageFullSamplerType::New()
                                               : this->m_ImageSampler->Clone();

    subfunctionSampler->SetUseMultiThread(false);
    subfunctionSampler->SetInput(this->GetFixedImage());
    subfunctionSampler->SetUseMask(useMask);
    subfunctionSampler->SetInputImageRegion(region);
    subfunctionSampler->SetNumberOfSamples(static_cast<int>(region.GetNumberOfPixels() * this->m_SamplingPercentage));

    const ImageSpatialObjectType * fixedImageMask =
      dynamic_cast<const ImageSpatialObjectType *>(this->GetFixedImageMask());
    if (fixedImageMask)
    {
      ImageSpatialObjectPointer fixedImageMaskRegion = fixedImageMask->Clone();
      fixedImageMaskRegion->SetImage(fixedImageMask->GetImage());
      fixedImageMaskRegion->Update();
      subfunctionSampler->SetMask(fixedImageMaskRegion);
    }

    try
    {
      subfunctionSampler->Update();
    }
    catch (const InputRegionOutsideOfMaskError & e)
    {}

    if (subfunctionSampler->GetOutput()->Size() == 0UL)
    {
      this->m_SubfunctionSamplers.push_back(nullptr);
      continue;
    }

    totalSamples += subfunctionSampler->GetOutput()->Size();
    this->m_SubfunctionSamplers.push_back(subfunctionSampler);
  }

  this->SetNumberOfFixedImageSamples(totalSamples);

  this->m_BSplinePointsRegions.clear();
  this->m_BSplinePointsRegionsNoMask.clear();

  this->m_SolutionEvaluations.resize(pop_size);
  this->m_BSplinePointsRegions.resize(1);
  this->m_BSplinePointsRegionsNoMask.resize(1);

  // add regions (that are within mask if set) to 0th list containing all regions of image combined.
  for (i = 0; i < this->m_BSplineFOSRegions.size(); ++i)
  {
    this->m_BSplinePointsRegionsNoMask[0].push_back(i);
    if (this->m_SubfunctionSamplers[i])
      this->m_BSplinePointsRegions[0].push_back(i);
  }
}

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::SelectNewSamplesSubfunctionSamplers()
{
  if (this->m_PartialEvaluations)
  {
    for (ImageSamplerPointer & sampler : this->m_SubfunctionSamplers)
    {
      if (sampler)
        sampler->Modified();
    }
  }
}

template <class TFixedImage, class TMovingImage>
inline void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::RepairFoldsInBsplineTransform(
  TransformParametersType & parameters)
{
  this->SetTransformParameters(parameters);
  this->GetTransformAsBsplinePtr()->RepairFolds();
}

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::GetRegionsForFOS()
{
  unsigned int i, j, d;

  const BSplineOrder3TransformType * bsplinePtr = this->GetTransformAsBsplinePtr();

  ImagePointer           wrappedImage = bsplinePtr->GetWrappedImages()[0];
  const FixedImageType * fixedImage = this->GetFixedImage();
  const int              num_points = bsplinePtr->GetNumberOfParameters() / FixedImageDimension;
  RegionType             coeffRegionCropped = wrappedImage->GetLargestPossibleRegion();
  IndexType              coeffRegionCroppedIndex;

  this->m_BSplinePointOffsetMap.clear();
  // this->m_BSplineRegionsToFosSets.clear();

  // this->m_BSplineRegionsToFosSets.resize(coeffRegionCropped.GetNumberOfPixels());
  this->m_BSplinePointOffsetMap.resize(coeffRegionCropped.GetNumberOfPixels());

  // crop control points grid to only contain those at lower left corners of fixed image pixel areas.
  for (d = 0; d < FixedImageDimension; ++d)
  {
    coeffRegionCropped.SetSize(d, coeffRegionCropped.GetSize(d) - 3);
    coeffRegionCroppedIndex[d] = coeffRegionCropped.GetIndex()[d] + 1;
  }
  coeffRegionCropped.SetIndex(coeffRegionCroppedIndex);

  this->m_BSplineFOSRegions.clear();
  this->m_BSplineFOSRegions.resize(coeffRegionCropped.GetNumberOfPixels());

  // iterate over these control points and calculate fixed image pixel regions
  ImageRegionConstIteratorWithIndex<ImageType> coeffImageIterator(wrappedImage, coeffRegionCropped);
  i = 0;
  while (!coeffImageIterator.IsAtEnd())
  {
    auto transformedPoint =
      wrappedImage->template TransformIndexToPhysicalPoint<PixelType>(coeffImageIterator.GetIndex());
    auto transformedPointUpper = transformedPoint + wrappedImage->GetDirection() * wrappedImage->GetSpacing();
    auto imageLowerIndex = fixedImage->TransformPhysicalPointToIndex(transformedPoint);
    auto imageUpperIndex = fixedImage->TransformPhysicalPointToIndex(transformedPointUpper);

    for (d = 0; d < FixedImageDimension; ++d)
    {
      imageLowerIndex[d] = std::max((int)imageLowerIndex[d], 0);
      imageUpperIndex[d] = std::min((int)imageUpperIndex[d], (int)fixedImage->GetLargestPossibleRegion().GetSize()[d]);
      this->m_BSplineFOSRegions[i].SetSize(d, imageUpperIndex[d] - imageLowerIndex[d]);
    }
    this->m_BSplineFOSRegions[i].SetIndex(imageLowerIndex);
    unsigned int offset = wrappedImage->ComputeOffset(coeffImageIterator.GetIndex());

    this->m_BSplinePointOffsetMap[offset] = i;
    ++coeffImageIterator;
    ++i;
  }
}

// assign fixed image regions to control points which affect them and vice versa.
template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::InitFOSMapping(int ** sets, int * set_length, int length)
{
  unsigned int i, j, d;

  this->m_FOS.length = length;
  this->m_FOS.sets = sets;
  this->m_FOS.set_length = set_length;

  this->m_BSplinePointsRegions.resize(length + 1);
  this->m_BSplinePointsRegionsNoMask.resize(length + 1);

  CombinationTransformType * comboPtr =
    dynamic_cast<CombinationTransformType *>(this->m_AdvancedTransform.GetPointer());
  const BSplineOrder3TransformType * bsplinePtr =
    dynamic_cast<const BSplineOrder3TransformType *>(comboPtr->GetCurrentTransform());
  ImagePointer wrappedImage = bsplinePtr->GetWrappedImages()[0];

  std::vector<bool> pointAdded(this->m_BSplinePointOffsetMap.size(), false);
  std::vector<bool> offsetAdded(this->m_BSplineFOSRegions.size(), false);
  const int         num_points = bsplinePtr->GetNumberOfParameters() / FixedImageDimension;

  // now compute mappings between control points and fos sets.
  // for each fos set j:
  for (j = 0; j < (unsigned)this->m_FOS.length; ++j)
  {
    std::fill(pointAdded.begin(), pointAdded.end(), false);
    std::fill(offsetAdded.begin(), offsetAdded.end(), false);

    // for each index i in fos set:
    for (i = 0; i < (unsigned)this->m_FOS.set_length[j]; ++i)
    {
      // calc control point number and its corresponding index
      int            cpoint = (this->m_FOS.sets[j][i] % num_points);
      ImageIndexType p = wrappedImage->ComputeIndex(cpoint);

      if (pointAdded[cpoint])
        continue;

      pointAdded[cpoint] = true;

      // calculate the region of influence for this control point
      ImageIndexType lower, upper;
      RegionType     hypercube;
      for (d = 0; d < FixedImageDimension; ++d)
      {
        lower[d] = std::max(static_cast<int>(p[d] - 2), 1);
        upper[d] = std::min(static_cast<int>(p[d] + 2),
                            static_cast<int>(wrappedImage->GetLargestPossibleRegion().GetSize(d) - 2));
        hypercube.SetSize(d, upper[d] - lower[d]);
      }
      hypercube.SetIndex(lower);
      ImageRegionConstIteratorWithIndex<ImageType> imageIterator(wrappedImage, hypercube);

      // precompute per control point the regions it influences, and per region the points it is influenced by.
      while (!imageIterator.IsAtEnd())
      {
        // offset needs to be adjusted to cropped coefficient grid.
        unsigned int offset = wrappedImage->ComputeOffset(imageIterator.GetIndex());
        offset = this->m_BSplinePointOffsetMap[offset];

        // add region to mapping from fos sets to regions if not added yet and within mask.
        if (!offsetAdded[offset])
        {
          offsetAdded[offset] = true;

          this->m_BSplinePointsRegionsNoMask[j + 1].push_back(offset);
          if (this->m_SubfunctionSamplers[offset])
          {
            this->m_BSplinePointsRegions[j + 1].push_back(offset);
            // this->m_BSplineRegionsToFosSets[offset].push_back(j);
          }
        }

        ++imageIterator;
      }
    }
  }
  if (this->GetImageSampler()->GetUseMask())
  {
    m_ParametersOutsideOfMask.resize(bsplinePtr->GetNumberOfParameters());
    this->ComputeParametersOutsideOfMask();
  }
}

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::ComputeParametersOutsideOfMask()
{
  for (int i = 1; i <= m_BSplinePointsRegions.size(); ++i)
  {
    if (m_BSplinePointsRegions[i].size() == 0)
    {
      for (int j = 0; j < m_FOS.set_length[i - 1]; ++j)
      {
        m_ParametersOutsideOfMask[m_FOS.sets[i - 1][j]] = true;
      }
    }
  }
}

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::WriteSamplesOfIteration(std::ofstream & outFile) const
{
  if (!this->GetPartialEvaluations())
  {
    this->GetImageSampler()->WriteSamplesToFile(outFile);
  }
  else
  {
    for (const int samplerIndex : this->m_BSplinePointsRegions[0])
    {
      ImageSamplerPointer sampler = this->m_SubfunctionSamplers[samplerIndex];
      sampler->WriteSamplesToFile(outFile);
    }
  }
}


/**
 * ********************* PrintSelf ****************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  /** Variables related to the Sampler. */
  os << indent << "Variables related to the Sampler: " << std::endl;
  os << indent.GetNextIndent() << "ImageSampler: " << this->m_ImageSampler.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "UseImageSampler: " << this->m_UseImageSampler << std::endl;

  /** Variables for the Limiters. */
  os << indent << "Variables related to the Limiters: " << std::endl;
  os << indent.GetNextIndent() << "FixedLimitRangeRatio: " << this->m_FixedLimitRangeRatio << std::endl;
  os << indent.GetNextIndent() << "MovingLimitRangeRatio: " << this->m_MovingLimitRangeRatio << std::endl;
  os << indent.GetNextIndent() << "UseFixedImageLimiter: " << this->m_UseFixedImageLimiter << std::endl;
  os << indent.GetNextIndent() << "UseMovingImageLimiter: " << this->m_UseMovingImageLimiter << std::endl;
  os << indent.GetNextIndent() << "FixedImageLimiter: " << this->m_FixedImageLimiter.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "MovingImageLimiter: " << this->m_MovingImageLimiter.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "FixedImageTrueMin: " << this->m_FixedImageTrueMin << std::endl;
  os << indent.GetNextIndent() << "MovingImageTrueMin: " << this->m_MovingImageTrueMin << std::endl;
  os << indent.GetNextIndent() << "FixedImageTrueMax: " << this->m_FixedImageTrueMax << std::endl;
  os << indent.GetNextIndent() << "MovingImageTrueMax: " << this->m_MovingImageTrueMax << std::endl;
  os << indent.GetNextIndent() << "FixedImageMinLimit: " << this->m_FixedImageMinLimit << std::endl;
  os << indent.GetNextIndent() << "MovingImageMinLimit: " << this->m_MovingImageMinLimit << std::endl;
  os << indent.GetNextIndent() << "FixedImageMaxLimit: " << this->m_FixedImageMaxLimit << std::endl;
  os << indent.GetNextIndent() << "MovingImageMaxLimit: " << this->m_MovingImageMaxLimit << std::endl;

  /** Variables related to image derivative computation. */
  os << indent << "Variables related to image derivative computation: " << std::endl;
  os << indent.GetNextIndent() << "InterpolatorIsBSpline: " << this->m_InterpolatorIsBSpline << std::endl;
  os << indent.GetNextIndent() << "BSplineInterpolator: " << this->m_BSplineInterpolator.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "InterpolatorIsBSplineFloat: " << this->m_InterpolatorIsBSplineFloat << std::endl;
  os << indent.GetNextIndent() << "BSplineInterpolatorFloat: " << this->m_BSplineInterpolatorFloat.GetPointer()
     << std::endl;
  os << indent.GetNextIndent()
     << "CentralDifferenceGradientFilter: " << this->m_CentralDifferenceGradientFilter.GetPointer() << std::endl;

  /** Variables used when the transform is a B-spline transform. */
  os << indent << "Variables store the transform as an AdvancedTransform: " << std::endl;
  os << indent.GetNextIndent() << "TransformIsAdvanced: " << this->m_TransformIsAdvanced << std::endl;
  os << indent.GetNextIndent() << "AdvancedTransform: " << this->m_AdvancedTransform.GetPointer() << std::endl;

  /** Other variables. */
  os << indent << "Other variables of the AdvancedImageToImageMetric: " << std::endl;
  os << indent.GetNextIndent() << "RequiredRatioOfValidSamples: " << this->m_RequiredRatioOfValidSamples << std::endl;
  os << indent.GetNextIndent() << "UseMovingImageDerivativeScales: " << this->m_UseMovingImageDerivativeScales
     << std::endl;
  os << indent.GetNextIndent() << "MovingImageDerivativeScales: " << this->m_MovingImageDerivativeScales << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef _itkAdvancedImageToImageMetric_hxx

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
#ifndef itkTransformBendingEnergyPenaltyTermAnalytic_hxx
#define itkTransformBendingEnergyPenaltyTermAnalytic_hxx

#include "itkTransformBendingEnergyPenaltyTermAnalytic.h"
#include "plastimatch/plm_image_header.h"

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif

namespace itk
{

template <class TFixedImage, class TScalarType>
TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::TransformBendingEnergyPenaltyTermAnalytic()
{
  this->m_RegularizationParameters.implementation = 'c';
  this->m_RegularizationParameters.curvature_penalty = 100.0f;
  this->SetUseImageSampler(true); // needed to copy over values
}

template <class TFixedImage, class TScalarType>
void
TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::Initialize()
{
  if (Self::FixedImageDimension != 3)
  {
    itkExceptionMacro(<< "Analytic bending energy can only be used when registering 3D volumes.");
    return;
  }

  this->Superclass::Initialize();

  const CombinationTransformType *   comboPtr = dynamic_cast<const CombinationTransformType *>(this->GetTransform());
  const BSplineOrder3TransformType * bsplinePtr =
    dynamic_cast<const BSplineOrder3TransformType *>(comboPtr->GetCurrentTransform());

  ImagePointer           wrappedImage = bsplinePtr->GetWrappedImages()[0];
  FixedImageConstPointer fixedImage = this->GetFixedImage();

  const Plm_image_header plmImageHeader{ fixedImage->GetLargestPossibleRegion(),
                                         fixedImage->GetOrigin(),
                                         fixedImage->GetSpacing(),
                                         fixedImage->GetDirection() };

  float gridSpacing[3];
  for (int d = 0; d < 3; ++d)
  {
    gridSpacing[d] = wrappedImage->GetSpacing()[d];
  }

  this->m_BsplineXform.initialize_unaligned(&plmImageHeader, gridSpacing);
  this->m_BsplineScore.total_grad = 0;
  this->m_BsplineScore.set_num_coeff(this->m_BsplineXform.num_coeff);

  this->m_BSplineRegularize.initialize(&this->m_RegularizationParameters, &this->m_BsplineXform);
}

template <class TFixedImage, class TScalarType>
typename TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::MeasureType
TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::GetValue(const ParametersType & parameters) const
{
  this->m_BsplineXform.coeff_ = parameters.data_block();
  this->m_BsplineScore.reset_score();
  this->m_BSplineRegularize.compute_score(
    &this->m_BsplineScore, &this->m_RegularizationParameters, &this->m_BsplineXform);

  return static_cast<MeasureType>(this->m_BsplineScore.rmetric) /
         static_cast<MeasureType>(this->GetNumberOfFixedImageVoxels());
}

template <class TFixedImage, class TScalarType>
typename TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::MeasureType
TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::GetValue(
  IntermediateResults & evaluation) const
{
  return evaluation[0] / static_cast<MeasureType>(this->GetNumberOfFixedImageVoxels());
}

/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TScalarType>
IntermediateResults
TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::GetValuePartial(const ParametersType & parameters,
                                                                                     int fosIndex) const
{
  IntermediateResults result{ 1 };

  m_BsplineXform.coeff_ = parameters.data_block();
  result[0] = m_BSplineRegularize.compute_score_analytic_omp_regions(this->m_BSplinePointsRegionsNoMask[fosIndex + 1],
                                                                     &m_RegularizationParameters,
                                                                     &m_BSplineRegularize,
                                                                     &m_BsplineXform);

  return result;
} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedImage, class TScalarType>
void
TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::GetDerivative(const ParametersType & parameters,
                                                                                   DerivativeType & derivative) const
{
  /** Slower, but works. */
  MeasureType dummyvalue = NumericTraits<MeasureType>::Zero;
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()

/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TScalarType>
void
TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  this->m_BsplineXform.coeff_ = parameters.data_block();
  this->m_BsplineScore.total_grad = derivative.data_block();
  this->m_BsplineScore.reset_score();
  this->m_BSplineRegularize.compute_score(
    &this->m_BsplineScore, &this->m_RegularizationParameters, &this->m_BsplineXform);

  value = static_cast<MeasureType>(this->m_BsplineScore.rmetric) /
          static_cast<MeasureType>(this->GetNumberOfFixedImageVoxels());
  for (unsigned int i = 0; i < derivative.Size(); ++i)
  {
    derivative[i] /= static_cast<MeasureType>(this->GetNumberOfFixedImageVoxels());
  }
} // end GetValueAndDerivative()


} // end namespace itk

#endif // #ifndef itkTransformBendingEnergyPenaltyTerm_hxx

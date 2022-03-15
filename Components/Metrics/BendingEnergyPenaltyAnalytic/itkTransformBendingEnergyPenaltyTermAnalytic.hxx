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

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif

namespace itk
{

template <class TFixedImage, class TScalarType>
typename TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::MeasureType
TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::GetValue(const ParametersType & parameters) const
{
  MeasureType measure = NumericTraits<MeasureType>::Zero;

  return measure;
}

/**
 * ******************* GetValuePartial *******************
 */

template <class TFixedImage, class TScalarType>
typename TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::MeasureType
TransformBendingEnergyPenaltyTermAnalytic<TFixedImage, TScalarType>::GetValue(const ParametersType & parameters,
                                                                              const int              fosIndex) const
{
  MeasureType measure = NumericTraits<MeasureType>::Zero;

  return measure;
} // end GetValuePartial()


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
{} // end GetValueAndDerivative()


} // end namespace itk

#endif // #ifndef itkTransformBendingEnergyPenaltyTerm_hxx

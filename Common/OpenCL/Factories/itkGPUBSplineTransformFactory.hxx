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
#ifndef itkGPUBSplineTransformFactory_hxx
#define itkGPUBSplineTransformFactory_hxx

#include "itkGPUBSplineTransformFactory.h"

namespace itk
{
template <typename NDimensions>
void
GPUBSplineTransformFactory2<NDimensions>::RegisterOneFactory()
{
  typedef GPUBSplineTransformFactory2<NDimensions> GPUTransformFactoryType;
  auto                                             factory = GPUTransformFactoryType::New();
  ObjectFactoryBase::RegisterFactory(factory);
}


//------------------------------------------------------------------------------
template <typename NDimensions>
GPUBSplineTransformFactory2<NDimensions>::GPUBSplineTransformFactory2()
{
  this->RegisterAll();
}


//------------------------------------------------------------------------------
template <typename NDimensions>
void
GPUBSplineTransformFactory2<NDimensions>::Register1D()
{
  // Define visitor and perform factory registration
  typelist::VisitDimension<RealTypeList, 1> visitor;
  visitor(*this);
}


//------------------------------------------------------------------------------
template <typename NDimensions>
void
GPUBSplineTransformFactory2<NDimensions>::Register2D()
{
  // Define visitor and perform factory registration
  typelist::VisitDimension<RealTypeList, 2> visitor;
  visitor(*this);
}


//------------------------------------------------------------------------------
template <typename NDimensions>
void
GPUBSplineTransformFactory2<NDimensions>::Register3D()
{
  // Define visitor and perform factory registration
  typelist::VisitDimension<RealTypeList, 3> visitor;
  visitor(*this);
}


} // namespace itk

#endif // end #ifndef itkGPUBSplineTransformFactory_hxx

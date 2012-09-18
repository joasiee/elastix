/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
#include "itkOpenCLSize.h"
//#include "itkOpenCLDevice.h"

namespace itk
{
static std::size_t opencl_gcd_of_size(std::size_t x, std::size_t y)
{
  std::size_t remainder;

  while ( ( remainder = x % y ) != 0 )
    {
    x = y;
    y = remainder;
    }
  return y;
}

const OpenCLSize::Null OpenCLSize::null = {};

//------------------------------------------------------------------------------
OpenCLSize OpenCLSize::toLocalWorkSize(const OpenCLSize & maxWorkItemSize,
                                       const std::size_t maxItemsPerGroup) const
{
  // Adjust for the maximum work item size in each dimension.
  std::size_t width = m_Dim >= 1 ? opencl_gcd_of_size( m_Sizes[0], maxWorkItemSize.GetWidth() ) : 1;
  std::size_t height = m_Dim >= 2 ? opencl_gcd_of_size( m_Sizes[1], maxWorkItemSize.GetHeight() ) : 1;
  std::size_t depth = m_Dim >= 3 ? opencl_gcd_of_size( m_Sizes[2], maxWorkItemSize.GetDepth() ) : 1;

  // Reduce in size by a factor of 2 until underneath the maximum group size.
  while ( maxItemsPerGroup && ( width * height * depth ) > maxItemsPerGroup )
    {
    width = ( width > 1 ) ? ( width / 2 ) : 1;
    height = ( height > 1 ) ? ( height / 2 ) : 1;
    depth = ( depth > 1 ) ? ( depth / 2 ) : 1;
    }

  // Return the final result.
  if ( m_Dim >= 3 )
    {
    return OpenCLSize(width, height, depth);
    }
  else if ( m_Dim >= 2 )
    {
    return OpenCLSize(width, height);
    }
  else
    {
    return OpenCLSize(width);
    }
}

//------------------------------------------------------------------------------
//OpenCLSize OpenCLSize::toLocalWorkSize(const OpenCLDevice & device) const
//{
//  return toLocalWorkSize( device.maximumWorkItemSize(),
//                          device.maximumWorkItemsPerGroup() );
//}

//------------------------------------------------------------------------------
static inline std::size_t opencl_cl_round_to(const std::size_t value, const std::size_t multiple)
{
  if ( multiple <= 1 )
    {
    return value;
    }
  std::size_t remainder = value % multiple;
  if ( !remainder )
    {
    return value;
    }
  else
    {
    return value + multiple - remainder;
    }
}

//------------------------------------------------------------------------------
OpenCLSize OpenCLSize::roundTo(const OpenCLSize & size) const
{
  if ( m_Dim == 1 )
    {
    return OpenCLSize( opencl_cl_round_to(m_Sizes[0], size.m_Sizes[0]) );
    }
  else if ( m_Dim == 2 )
    {
    return OpenCLSize( opencl_cl_round_to(m_Sizes[0], size.m_Sizes[0]),
                       opencl_cl_round_to(m_Sizes[1], size.m_Sizes[1]) );
    }
  else
    {
    return OpenCLSize( opencl_cl_round_to(m_Sizes[0], size.m_Sizes[0]),
                       opencl_cl_round_to(m_Sizes[1], size.m_Sizes[1]),
                       opencl_cl_round_to(m_Sizes[2], size.m_Sizes[2]) );
    }
}

//------------------------------------------------------------------------------
//! Operator ==
bool operator==(const OpenCLSize & lhs, const OpenCLSize & rhs)
{
  if ( &rhs == &lhs )
    {
    return true;
    }

  return lhs.GetDimension() == rhs.GetDimension()
         && lhs.GetWidth() == rhs.GetWidth()
         && lhs.GetHeight() == rhs.GetHeight()
         && lhs.GetDepth() == rhs.GetDepth();
}

//------------------------------------------------------------------------------
//! Operator !=
bool operator!=(const OpenCLSize & lhs, const OpenCLSize & rhs)
{
  return !( lhs == rhs );
}
} // namespace itk

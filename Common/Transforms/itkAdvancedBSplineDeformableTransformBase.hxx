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
#ifndef itkAdvancedBSplineDeformableTransformBase_hxx
#define itkAdvancedBSplineDeformableTransformBase_hxx

#include "itkAdvancedBSplineDeformableTransformBase.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkContinuousIndex.h"
#include "itkIdentityTransform.h"
#include <vnl/vnl_math.h>

namespace itk
{

// Constructor with default arguments
template <class TScalarType, unsigned int NDimensions>
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::AdvancedBSplineDeformableTransformBase(
  const unsigned splineOrder)
  : Superclass(SpaceDimension)
  , m_SplineOrder(splineOrder)
{
  this->m_InternalParametersBuffer = ParametersType(0);
  // Make sure the parameters pointer is not NULL after construction.
  this->m_InputParametersPointer = &(this->m_InternalParametersBuffer);

  // Initialize coeffient images
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    this->m_WrappedImage[j] = ImageType::New();
    this->m_WrappedImage[j]->SetRegions(this->m_GridRegion);
    this->m_WrappedImage[j]->SetOrigin(this->m_GridOrigin.GetDataPointer());
    this->m_WrappedImage[j]->SetSpacing(this->m_GridSpacing.GetDataPointer());
    this->m_WrappedImage[j]->SetDirection(this->m_GridDirection);
    this->m_CoefficientImages[j] = nullptr;
  }

  this->m_ValidRegion = this->m_GridRegion;

  // Initialize Jacobian images
  //   for ( unsigned int j = 0; j < SpaceDimension; j++ )
  //     {
  //     this->m_JacobianImage[j] = ImageType::New();
  //     this->m_JacobianImage[j]->SetRegions( this->m_GridRegion );
  //     this->m_JacobianImage[j]->SetOrigin( this->m_GridOrigin.GetDataPointer() );
  //     this->m_JacobianImage[j]->SetSpacing( this->m_GridSpacing.GetDataPointer() );
  //     this->m_JacobianImage[j]->SetDirection( this->m_GridDirection );
  //     }

  /** Fixed Parameters store the following information:
   *     Grid Size
   *     Grid Origin
   *     Grid Spacing
   *     Grid Direction
   *  The size of these is equal to the  NInputDimensions
   */
  this->m_FixedParameters.SetSize(NumberOfFixedParameters);
  this->m_FixedParameters.Fill(0.0);
  for (unsigned int i = 0; i < NDimensions; ++i)
  {
    this->m_FixedParameters[2 * NDimensions + i] = this->m_GridSpacing[i];
  }
  for (unsigned int di = 0; di < NDimensions; ++di)
  {
    for (unsigned int dj = 0; dj < NDimensions; ++dj)
    {
      this->m_FixedParameters[3 * NDimensions + (di * NDimensions + dj)] = this->m_GridDirection[di][dj];
    }
  }

  this->UpdatePointIndexConversions();

  this->m_LastJacobianIndex = this->m_ValidRegion.GetIndex();
}


// Get the number of parameters
template <class TScalarType, unsigned int NDimensions>
auto
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::GetNumberOfParameters() const
  -> NumberOfParametersType
{

  // The number of parameters equal SpaceDimension * number of
  // of pixels in the grid region.
  return (static_cast<NumberOfParametersType>(SpaceDimension) *
          static_cast<NumberOfParametersType>(this->m_GridRegion.GetNumberOfPixels()));
}


// Get the number of parameters per dimension
template <class TScalarType, unsigned int NDimensions>
auto
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::GetNumberOfParametersPerDimension() const
  -> NumberOfParametersType
{
  // The number of parameters per dimension equal number of
  // of pixels in the grid region.
  return (static_cast<NumberOfParametersType>(this->m_GridRegion.GetNumberOfPixels()));
}


// Set the grid spacing
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::UpdatePointIndexConversions()
{
  DirectionType scale;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    scale[i][i] = this->m_GridSpacing[i];
  }

  this->m_IndexToPoint = this->m_GridDirection * scale;
  this->m_PointToIndexMatrix = this->m_IndexToPoint.GetInverse();
  this->m_PointToIndexMatrixTransposed = this->m_PointToIndexMatrix.GetTranspose();
  this->m_PointToIndexMatrixIsDiagonal = true;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_PointToIndexMatrix2[i][j] = static_cast<ScalarType>(this->m_PointToIndexMatrix[i][j]);
      this->m_PointToIndexMatrixTransposed2[i][j] = static_cast<ScalarType>(this->m_PointToIndexMatrixTransposed[i][j]);
      if (i != j && this->m_PointToIndexMatrix[i][j] != 0.0)
      {
        this->m_PointToIndexMatrixIsDiagonal = false;
      }
    }
  }

  this->m_PointToIndexMatrixDiagonal = this->m_PointToIndexMatrixTransposed2.GetVnlMatrix().get_diagonal().data_block();
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_PointToIndexMatrixDiagonalProducts[i + SpaceDimension * j] =
        this->m_PointToIndexMatrixDiagonal[i] * m_PointToIndexMatrixDiagonal[j];
    }
  }

} // end UpdatePointIndexConversions()


//
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::UpdateGridOffsetTable()
{
  SizeType gridSize = this->m_GridRegion.GetSize();
  this->m_GridOffsetTable.Fill(1);
  for (unsigned int j = 1; j < SpaceDimension; ++j)
  {
    this->m_GridOffsetTable[j] = this->m_GridOffsetTable[j - 1] * gridSize[j - 1];
  }

} // end UpdateGridOffsetTable()


// Set the grid spacing
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::SetGridSpacing(const SpacingType & spacing)
{
  if (this->m_GridSpacing != spacing)
  {
    this->m_GridSpacing = spacing;

    // set spacing for each coefficient and Jacobian image
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_WrappedImage[j]->SetSpacing(this->m_GridSpacing.GetDataPointer());
    }

    this->UpdatePointIndexConversions();

    this->Modified();
  }
}


// Set the grid direction
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::SetGridDirection(const DirectionType & direction)
{
  if (this->m_GridDirection != direction)
  {
    this->m_GridDirection = direction;

    // set direction for each coefficient and Jacobian image
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_WrappedImage[j]->SetDirection(this->m_GridDirection);
    }

    this->UpdatePointIndexConversions();

    this->Modified();
  }
}


// Set the grid origin
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::SetGridOrigin(const OriginType & origin)
{
  if (this->m_GridOrigin != origin)
  {
    this->m_GridOrigin = origin;

    // set spacing for each coefficient and jacobianimage
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_WrappedImage[j]->SetOrigin(this->m_GridOrigin.GetDataPointer());
    }

    this->Modified();
  }
}


// Set the parameters
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::SetIdentity()
{
  if (this->m_InputParametersPointer)
  {
    ParametersType * parameters = const_cast<ParametersType *>(this->m_InputParametersPointer);
    parameters->Fill(0.0);
    this->Modified();
  }
  else
  {
    itkExceptionMacro(<< "Input parameters for the spline haven't been set ! Set them using the SetParameters or "
                         "SetCoefficientImage method first.");
  }
}


// Set the parameters
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::SetParameters(const ParametersType & parameters)
{

  // check if the number of parameters match the
  // expected number of parameters
  if (parameters.Size() != this->GetNumberOfParameters())
  {
    itkExceptionMacro(<< "Mismatched between parameters size " << parameters.size() << " and region size "
                      << this->m_GridRegion.GetNumberOfPixels());
  }

  // Clean up buffered parameters
  this->m_InternalParametersBuffer = ParametersType(0);

  // Keep a reference to the input parameters
  this->m_InputParametersPointer = &parameters;

  // Wrap flat array as images of coefficients
  this->WrapAsImages();

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();
}


// Set the Fixed Parameters
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::SetFixedParameters(
  const FixedParametersType & passedParameters)
{
  FixedParametersType parameters(NumberOfFixedParameters);

  // check if the number of parameters match the
  // expected number of parameters
  if (passedParameters.Size() == NDimensions * 3)
  {
    parameters.Fill(0.0);
    for (unsigned int i = 0; i < 3 * NDimensions; ++i)
    {
      parameters[i] = passedParameters[i];
    }
    for (unsigned int di = 0; di < NDimensions; ++di)
    {
      parameters[3 * NDimensions + (di * NDimensions + di)] = 1;
    }
  }
  else if (passedParameters.Size() != NumberOfFixedParameters)
  {
    itkExceptionMacro(<< "Mismatched between parameters size " << passedParameters.size()
                      << " and number of fixed parameters " << NumberOfFixedParameters);
  }
  else
  {
    for (unsigned int i = 0; i < NumberOfFixedParameters; ++i)
    {
      parameters[i] = passedParameters[i];
    }
  }

  /*********************************************************
    Fixed Parameters store the following information:
        Grid Size
        Grid Origin
        Grid Spacing
        Grid Direction
     The size of these is equal to the  NInputDimensions
  *********************************************************/

  /** Set the Grid Parameters */
  SizeType gridSize;
  for (unsigned int i = 0; i < NDimensions; ++i)
  {
    gridSize[i] = static_cast<int>(parameters[i]);
  }
  RegionType bsplineRegion;
  bsplineRegion.SetSize(gridSize);

  /** Set the Origin Parameters */
  OriginType origin;
  for (unsigned int i = 0; i < NDimensions; ++i)
  {
    origin[i] = parameters[NDimensions + i];
  }

  /** Set the Spacing Parameters */
  SpacingType spacing;
  for (unsigned int i = 0; i < NDimensions; ++i)
  {
    spacing[i] = parameters[2 * NDimensions + i];
  }

  /** Set the Direction Parameters */
  DirectionType direction;
  for (unsigned int di = 0; di < NDimensions; ++di)
  {
    for (unsigned int dj = 0; dj < NDimensions; ++dj)
    {
      direction[di][dj] = parameters[3 * NDimensions + (di * NDimensions + dj)];
    }
  }

  this->SetGridSpacing(spacing);
  this->SetGridDirection(direction);
  this->SetGridOrigin(origin);
  this->SetGridRegion(bsplineRegion);
  this->UpdateGridOffsetTable();

  this->Modified();
}


// Wrap flat parameters as images
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::WrapAsImages()
{
  /**
   * Wrap flat parameters array into SpaceDimension number of ITK images
   * NOTE: For efficiency, parameters are not copied locally. The parameters
   * are assumed to be maintained by the caller.
   */
  PixelType *  dataPointer = const_cast<PixelType *>((this->m_InputParametersPointer->data_block()));
  unsigned int numberOfPixels = this->m_GridRegion.GetNumberOfPixels();

  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    this->m_WrappedImage[j]->GetPixelContainer()->SetImportPointer(dataPointer, numberOfPixels);
    dataPointer += numberOfPixels;
    this->m_CoefficientImages[j] = this->m_WrappedImage[j];
  }
}


// Set the parameters by value
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::SetParametersByValue(
  const ParametersType & parameters)
{
  // check if the number of parameters match the
  // expected number of parameters
  if (parameters.Size() != this->GetNumberOfParameters())
  {
    itkExceptionMacro(<< "Mismatched between parameters size " << parameters.size() << " and region size "
                      << this->m_GridRegion.GetNumberOfPixels());
  }

  // copy it
  this->m_InternalParametersBuffer = parameters;
  this->m_InputParametersPointer = &(this->m_InternalParametersBuffer);

  // wrap flat array as images of coefficients
  this->WrapAsImages();

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();
}


// Get the parameters
template <class TScalarType, unsigned int NDimensions>
auto
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::GetParameters() const -> const ParametersType &
{
  /** NOTE: For efficiency, this class does not keep a copy of the parameters -
   * it just keeps pointer to input parameters.
   */
  if (nullptr == this->m_InputParametersPointer)
  {
    itkExceptionMacro(<< "Cannot GetParameters() because m_InputParametersPointer is NULL. Perhaps "
                         "SetCoefficientImages() has been called causing the NULL pointer.");
  }

  return (*this->m_InputParametersPointer);
}

// Get the parameters
template <class TScalarType, unsigned int NDimensions>
auto
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::GetFixedParameters() const
  -> const FixedParametersType &
{
  RegionType resRegion = this->GetGridRegion();

  for (unsigned int i = 0; i < NDimensions; ++i)
  {
    this->m_FixedParameters[i] = (resRegion.GetSize())[i];
  }
  for (unsigned int i = 0; i < NDimensions; ++i)
  {
    this->m_FixedParameters[NDimensions + i] = (this->GetGridOrigin())[i];
  }
  for (unsigned int i = 0; i < NDimensions; ++i)
  {
    this->m_FixedParameters[2 * NDimensions + i] = (this->GetGridSpacing())[i];
  }
  for (unsigned int di = 0; di < NDimensions; ++di)
  {
    for (unsigned int dj = 0; dj < NDimensions; ++dj)
    {
      this->m_FixedParameters[3 * NDimensions + (di * NDimensions + dj)] = (this->GetGridDirection())[di][dj];
    }
  }

  return (this->m_FixedParameters);
}

// Set the B-Spline coefficients using input images
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::SetCoefficientImages(ImagePointer images[])
{
  if (images[0])
  {
    this->SetGridRegion(images[0]->GetBufferedRegion());
    this->SetGridSpacing(images[0]->GetSpacing());
    this->SetGridDirection(images[0]->GetDirection());
    this->SetGridOrigin(images[0]->GetOrigin());
    this->UpdateGridOffsetTable();

    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_CoefficientImages[j] = images[j];
    }

    // Clean up buffered parameters
    this->m_InternalParametersBuffer = ParametersType(0);
    this->m_InputParametersPointer = nullptr;
  }
}

template <class TScalarType, unsigned int NDimensions>
unsigned int
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::ComputeNumberOfFolds() const
{
  unsigned int totalNumberOfFolds = 0;

  if (m_ComputeControlPointFolds)
  {
    for (unsigned int i = 0; i < this->m_GridRegion.GetNumberOfPixels(); ++i)
      totalNumberOfFolds += this->IsControlPointFolded(i);
  }

  return totalNumberOfFolds;
}

template <class TScalarType, unsigned int NDimensions>
inline void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::RepairFolds()
{
  if (m_ComputeControlPointFolds)
  {
    bool foldsInGrid = true;
    int  maxNrPasses = 10;
    int  nrPasses = 0;
    elastix::log::info(std::ostringstream{} << "\nRepairing folds in grid...");
    elastix::log::info(std::ostringstream{} << "  Number of folds in grid: " << this->ComputeNumberOfFolds());
    while (foldsInGrid && nrPasses < maxNrPasses)
    {
      foldsInGrid = false;
      for (unsigned int i = 0; i < this->m_GridRegion.GetNumberOfPixels(); ++i)
      {
        foldsInGrid |= this->RepairFold(i);
      }
      nrPasses += foldsInGrid;
    }
    elastix::log::info(std::ostringstream{} << "  Number of passes to repair folds: " << nrPasses);
    if (nrPasses > 0)
      elastix::log::info(std::ostringstream{} << "  Number of folds after " << nrPasses << "passes: " << this->ComputeNumberOfFolds());
    elastix::log::info(std::ostringstream{}); // empty line
  }
}

template <class TScalarType, unsigned int NDimensions>
bool
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::IsControlPointFolded(int offset) const
{
  IndexType                                    index = m_WrappedImage[0]->ComputeIndex(offset);
  RegionType                                   region = this->GetRegionAroundControlPoint(index);
  ImageRegionConstIteratorWithIndex<ImageType> wrappedImageIterator(m_WrappedImage[0], region);
  DirectionType                                direction = this->GetGridDirection();
  OutputPointType                              centerPoint = this->GetPositionOfControlPoint(index);

  while (!wrappedImageIterator.IsAtEnd())
  {
    if (m_WrappedImage[0]->ComputeOffset(wrappedImageIterator.GetIndex()) != offset)
    {

      OutputPointType point = this->GetPositionOfControlPoint(wrappedImageIterator.GetIndex());
      OffsetType      indexDiff = wrappedImageIterator.GetIndex() - index;
      OutputPointType diff = centerPoint - point;
      bool            folded = true;

      for (unsigned int dim = 0; dim < NDimensions; ++dim)
      {
        diff[dim] = diff[dim] * indexDiff[dim] * direction[dim][dim];
        folded = folded && (diff[dim] >= 0);
      }

      if (folded)
        return folded;
    }

    ++wrappedImageIterator;
  }

  return false;
}

template <class TScalarType, unsigned int NDimensions>
bool
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::RepairFold(int offset)
{
  IndexType                                    index = m_WrappedImage[0]->ComputeIndex(offset);
  RegionType                                   region = this->GetRegionAroundControlPoint(index);
  ImageRegionConstIteratorWithIndex<ImageType> wrappedImageIterator(m_WrappedImage[0], region);
  DirectionType                                direction = this->GetGridDirection();
  OutputPointType                              centerPoint = this->GetPositionOfControlPoint(index);
  bool                                         folded = false;

  while (!wrappedImageIterator.IsAtEnd())
  {
    if (m_WrappedImage[0]->ComputeOffset(wrappedImageIterator.GetIndex()) != offset)
    {

      OutputPointType point = this->GetPositionOfControlPoint(wrappedImageIterator.GetIndex());
      OffsetType      indexDiff = wrappedImageIterator.GetIndex() - index;
      OutputPointType diff = centerPoint - point;
      double          margin = 1e-3;

      for (unsigned int dim = 0; dim < NDimensions; ++dim)
      {
        diff[dim] = diff[dim] * indexDiff[dim] * direction[dim][dim];
        if (diff[dim] >= 0 && indexDiff[dim] != 0)
        {
          centerPoint[dim] = point[dim] - direction[dim][dim] * indexDiff[dim] * margin;
          folded = true;
        }
      }
    }
    ++wrappedImageIterator;
  }
  this->SetPositionOfControlPoint(index, centerPoint);
  return folded;
}

template <class TScalarType, unsigned int NDimensions>
auto
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::GetRegionAroundControlPoint(
  const IndexType & index) const -> RegionType
{
  OffsetType unitOffset;
  unitOffset.Fill(1);

  IndexType regionStart = index - unitOffset;
  IndexType regionEnd = index + unitOffset;

  for (unsigned int dim = 0; dim < NDimensions; ++dim)
  {
    regionStart[dim] = std::max(regionStart[dim], m_GridRegion.GetIndex()[dim]);
    regionEnd[dim] = std::min(regionEnd[dim], m_GridRegion.GetUpperIndex()[dim]);
  }

  OffsetType regionSizeAsOffset = regionEnd - regionStart + unitOffset;
  SizeType   regionSize;
  for (unsigned int dim = 0; dim < NDimensions; ++dim)
  {
    regionSize[dim] = regionSizeAsOffset[dim];
  }

  RegionType region(regionStart, regionSize);

  return region;
}

// Print self
template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::PrintSelf(std::ostream & os, Indent indent) const
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "GridRegion: " << this->m_GridRegion << std::endl;
  os << indent << "GridOrigin: " << this->m_GridOrigin << std::endl;
  os << indent << "GridSpacing: " << this->m_GridSpacing << std::endl;
  os << indent << "GridDirection:\n" << this->m_GridDirection << std::endl;
  os << indent << "GridOffsetTable: " << this->m_GridOffsetTable << std::endl;
  os << indent << "IndexToPoint:\n" << this->m_IndexToPoint << std::endl;
  os << indent << "PointToIndex:\n" << this->m_PointToIndexMatrix << std::endl;
  os << indent << "PointToIndex2:\n" << this->m_PointToIndexMatrix2 << std::endl;
  os << indent << "PointToIndexTransposed:\n" << this->m_PointToIndexMatrixTransposed << std::endl;
  os << indent << "PointToIndexTransposed2:\n" << this->m_PointToIndexMatrixTransposed2 << std::endl;

  os << indent << "CoefficientImage: [ " << this->m_CoefficientImages[0].GetPointer();
  for (unsigned int j = 1; j < SpaceDimension; ++j)
  {
    os << ", " << this->m_CoefficientImages[j].GetPointer();
  }
  os << " ]" << std::endl;

  os << indent << "WrappedImage: [ " << this->m_WrappedImage[0].GetPointer();
  for (unsigned int j = 1; j < SpaceDimension; ++j)
  {
    os << ", " << this->m_WrappedImage[j].GetPointer();
  }
  os << " ]" << std::endl;

  os << indent << "InputParametersPointer: " << this->m_InputParametersPointer << std::endl;
  os << indent << "ValidRegion: " << this->m_ValidRegion << std::endl;
  os << indent << "LastJacobianIndex: " << this->m_LastJacobianIndex << std::endl;
}


// Transform a point
template <class TScalarType, unsigned int NDimensions>
bool
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::InsideValidRegion(
  const ContinuousIndexType & index) const
{
  bool inside = true;

  /** Check if index can be evaluated given the current grid. */
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    if (index[j] < this->m_ValidRegionBegin[j] || index[j] >= this->m_ValidRegionEnd[j])
    {
      inside = false;
      break;
    }
  }

  return inside;
}

template <class TScalarType, unsigned int NDimensions>
auto
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::TransformPointToContinuousGridIndex(
  const InputPointType & point) const -> ContinuousIndexType
{
  Vector<double, SpaceDimension> tvector;

  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    tvector[j] = point[j] - this->m_GridOrigin[j];
  }

  Vector<double, SpaceDimension> cvector = this->m_PointToIndexMatrix * tvector;
  ContinuousIndexType            cindex;

  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    cindex[j] = static_cast<typename ContinuousIndexType::CoordRepType>(cvector[j]);
  }
  return cindex;
}

template <class TScalarType, unsigned int NDimensions>
auto
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::TransformContinuousGridIndexToPoint(
  const ContinuousIndexType & index) const -> InputPointType
{
  Vector<double, SpaceDimension> cvector;
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    cvector[j] = index[j];
  }
  Vector<double, SpaceDimension> tvector = this->m_IndexToPoint * cvector;
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    tvector[j] = tvector[j] + this->m_GridOrigin[j];
  }

  InputPointType point;
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    point[j] = static_cast<typename InputPointType::CoordRepType>(tvector[j]);
  }
  return point;
}

template <class TScalarType, unsigned int NDimensions>
auto
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::GetPositionOfControlPoint(
  const IndexType & index) const -> OutputPointType
{
  OutputPointType point = this->TransformContinuousGridIndexToPoint(index);

  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    point[i] = point[i] + m_CoefficientImages[i]->GetPixel(index);
  }

  return point;
}

template <class TScalarType, unsigned int NDimensions>
inline void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::SetPositionOfControlPoint(
  const IndexType &       index,
  const OutputPointType & position)
{
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    m_CoefficientImages[i]->SetPixel(index, position[i] - this->TransformContinuousGridIndexToPoint(index)[i]);
  }
}

template <class TScalarType, unsigned int NDimensions>
void
AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>::WriteParametersAsPoints(std::ofstream & outfile) const
{
  const ParametersType &                       params = this->GetParameters();
  const size_t                                 nPoints = this->GetNumberOfParametersPerDimension();
  ImageRegionConstIteratorWithIndex<ImageType> coeffImageIterator(m_CoefficientImages[0],
                                                                  m_CoefficientImages[0]->GetLargestPossibleRegion());
  while (!coeffImageIterator.IsAtEnd())
  {
    IndexType index = coeffImageIterator.GetIndex();
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      outfile << index[i] << " ";
    }
    OutputPointType point = this->GetPositionOfControlPoint(index);

    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      outfile << point[i] << " ";
    }
    outfile << "\n";

    ++coeffImageIterator;
  }
  outfile.flush();
}

} // namespace itk

#endif

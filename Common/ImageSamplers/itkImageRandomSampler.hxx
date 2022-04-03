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
#ifndef itkImageRandomSampler_hxx
#define itkImageRandomSampler_hxx

#include "itkImageRandomSampler.h"

#include "itkImageRandomConstIteratorWithIndex.h"

namespace itk
{
/**
 * ******************* GenerateData *******************
 */

template <class TInputImage>
void
ImageRandomSampler<TInputImage>::GenerateData()
{
  /** Get a handle to the mask. If there was no mask supplied we exercise a multi-threaded version. */
  typename MaskType::ConstPointer mask = this->GetMask();
  if (mask.IsNull() && this->m_UseMultiThread)
  {
    /** Calls ThreadedGenerateData(). */
    return Superclass::GenerateData();
  }

  /** Get handles to the input image, output sample container. */
  InputImageConstPointer                     inputImage = this->GetInput();
  typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();

  /** Reserve memory for the output. */
  sampleContainer->Reserve(this->GetNumberOfSamples());

  /** Setup an iterator over the output, which is of ImageSampleContainerType. */
  typename ImageSampleContainerType::Iterator      iter;
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  if (mask.IsNull() || !this->m_UseMask)
  {
    for (iter = sampleContainer->Begin(); iter != end; ++iter)
    {
      InputImageIndexType positionIndex;
      this->GeneratePoint(positionIndex, (*iter).Value().m_ImageCoordinates);

      /** Get the value and put it in the sample. */
      (*iter).Value().m_ImageValue = static_cast<ImageSampleValueType>(inputImage->GetPixel(positionIndex));
    }
  } // end if no mask
  else
  {
    /** Update the mask. */
    if (mask->GetSource())
    {
      mask->GetSource()->Update();
    }

    /** Make sure we are not eternally trying to find samples: */
    const unsigned long maxSamples = 10 * this->GetNumberOfSamples();
    unsigned long       numSamples = 0L;

    /** Loop over the sample container. */
    InputImagePointType inputPoint;
    InputImageIndexType positionIndex;
    bool                insideMask = false;
    for (iter = sampleContainer->Begin(); iter != end; ++iter)
    {
      /** Loop until a valid sample is found. */
      do
      {
        /** Jump to a random position. */
        ++numSamples;
        /** Check if we are not trying eternally to find a valid point. */
        if (numSamples > maxSamples)
        {
          /** Squeeze the sample container to the size that is still valid. */
          typename ImageSampleContainerType::iterator stlnow = sampleContainer->begin();
          typename ImageSampleContainerType::iterator stlend = sampleContainer->end();
          stlnow += iter.Index();
          sampleContainer->erase(stlnow, stlend);
          itkExceptionMacro(
            << "Could not find enough image samples within reasonable time. Probably the mask is too small");
        }
        /** Get the index, and transform it to the physical coordinates. */
        this->GeneratePoint(positionIndex, inputPoint);
        /** Check if it's inside the mask. */
        insideMask = mask->IsInsideInWorldSpace(inputPoint);
      } while (!insideMask);

      /** Put the coordinates and the value in the sample. */
      (*iter).Value().m_ImageCoordinates = inputPoint;
      (*iter).Value().m_ImageValue = static_cast<ImageSampleValueType>(inputImage->GetPixel(positionIndex));

    } // end for loop
  }

} // end GenerateData()


/**
 * ******************* ThreadedGenerateData *******************
 */

template <class TInputImage>
void
ImageRandomSampler<TInputImage>::ThreadedGenerateData(const InputImageRegionType &, ThreadIdType threadId)
{
  /** Sanity check. */
  typename MaskType::ConstPointer mask = this->GetMask();
  if (mask.IsNotNull())
  {
    itkExceptionMacro(<< "ERROR: do not call this function when a mask is supplied.");
  }

  /** Get handle to the input image. */
  InputImageConstPointer inputImage = this->GetInput();

  /** Figure out which samples to process. */
  unsigned long chunkSize = this->GetNumberOfSamples() / this->GetNumberOfWorkUnits();
  unsigned long sampleStart = threadId * chunkSize;
  if (threadId == this->GetNumberOfWorkUnits() - 1)
  {
    chunkSize = this->GetNumberOfSamples() - ((this->GetNumberOfWorkUnits() - 1) * chunkSize);
  }

  /** Get a reference to the output and reserve memory for it. */
  ImageSampleContainerPointer & sampleContainerThisThread = this->m_ThreaderSampleContainer[threadId];
  sampleContainerThisThread->Reserve(chunkSize);

  /** Setup an iterator over the sampleContainerThisThread. */
  typename ImageSampleContainerType::Iterator      iter;
  typename ImageSampleContainerType::ConstIterator end = sampleContainerThisThread->End();

  /** Fill the local sample container. */
  unsigned long       sampleId = sampleStart;
  InputImageSizeType  regionSize = this->GetCroppedInputImageRegion().GetSize();
  InputImageIndexType regionIndex = this->GetCroppedInputImageRegion().GetIndex();
  for (iter = sampleContainerThisThread->Begin(); iter != end; ++iter, sampleId++)
  {
    InputImageIndexType positionIndex;
    this->GeneratePoint(positionIndex, (*iter).Value().m_ImageCoordinates);

    /** Get the value and put it in the sample. */
    (*iter).Value().m_ImageValue = static_cast<ImageSampleValueType>(inputImage->GetPixel(positionIndex));

  } // end for loop

} // end ThreadedGenerateData()

/**
 * ******************* GeneratePoint *******************
 */

template <class TInputImage>
void
ImageRandomSampler<TInputImage>::GeneratePoint(InputImageIndexType & index, InputImagePointType & point)
{
  InputImageConstPointer inputImage = this->GetInput();
  InputImageSizeType     unitSize;
  unitSize.Fill(1);
  InputImageIndexType smallestIndex = this->GetCroppedInputImageRegion().GetIndex();
  InputImageIndexType largestIndex = smallestIndex + this->GetCroppedInputImageRegion().GetSize() - unitSize;

  for (unsigned int i = 0; i < InputImageDimension; ++i)
  {
    std::uniform_int_distribution<unsigned long> dist(smallestIndex[i], largestIndex[i]);
    index[i] = dist(this->m_RandomGenerator);
  }

  /** Transform index to the physical coordinates and put it in the sample. */
  inputImage->TransformIndexToPhysicalPoint(index, point);
}
} // end namespace itk

#endif // end #ifndef itkImageRandomSampler_hxx

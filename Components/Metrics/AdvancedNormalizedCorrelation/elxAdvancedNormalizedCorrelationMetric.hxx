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
#ifndef elxAdvancedNormalizedCorrelationMetric_hxx
#define elxAdvancedNormalizedCorrelationMetric_hxx

#include "elxAdvancedNormalizedCorrelationMetric.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
AdvancedNormalizedCorrelationMetric<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Get and set SubtractMean. Default true. */
  bool subtractMean = true;
  this->GetConfiguration()->ReadParameter(subtractMean, "SubtractMean", this->GetComponentLabel(), level, 0);
  this->SetSubtractMean(subtractMean);

} // end BeforeEachResolution()

template <class TElastix>
void
AdvancedNormalizedCorrelationMetric<TElastix>::BeforeRegistration(void)
{
  this->AddTargetCellToIterationInfo("3:MissedPixels");
  this->GetIterationInfoAt("3:MissedPixels") << std::showpoint << std::fixed;
}

template <class TElastix>
void
AdvancedNormalizedCorrelationMetric<TElastix>::AfterEachIteration(void)
{
  this->GetIterationInfoAt("3:MissedPixels") << boost::accumulators::mean(this->m_MissedPixelsMean);
  this->m_MissedPixelsMean = {};
}


/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
AdvancedNormalizedCorrelationMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of AdvancedNormalizedCorrelation metric took: "
                                 << static_cast<long>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


} // end namespace elastix

#endif // end #ifndef elxAdvancedNormalizedCorrelationMetric_hxx

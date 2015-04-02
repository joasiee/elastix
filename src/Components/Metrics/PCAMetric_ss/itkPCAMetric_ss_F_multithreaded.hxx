/*======================================================================

  This file is part of the elastix software.

  Copyright (c) Erasmus MC, Rotterdam. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef PCAMETRIC_ss_F_MULTITHREADED_HXX
#define PCAMETRIC_ss_F_MULTITHREADED_HXX
#include "itkPCAMetric_ss_F_multithreaded.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "vnl/algo/vnl_matrix_update.h"
#include "itkImage.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_trace.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include <numeric>
#include <fstream>

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
PCAMetric_ss<TFixedImage,TMovingImage>
::PCAMetric_ss():
    m_TransformIsStackTransform( false ),
    m_NumEigenValues( 6 ),
    m_NumSingleSubjects( 1 )
{
    this->SetUseImageSampler( true );
    this->SetUseFixedImageLimiter( false );
    this->SetUseMovingImageLimiter( false );


    // Multi-threading structs
    this->m_PCAMetricssGetSamplesPerThreadVariables     = NULL;
    this->m_PCAMetricssGetSamplesPerThreadVariablesSize = 0;

    /** Initialize the m_ParzenWindowHistogramThreaderParameters. */
    this->m_PCAMetricssThreaderParameters.m_Metric = this;


} // end constructor

/**
 * ******************* Destructor *******************
 */

template< class TFixedImage, class TMovingImage >
PCAMetric_ss< TFixedImage, TMovingImage >
::~PCAMetric_ss()
{
    delete[] this->m_PCAMetricssGetSamplesPerThreadVariables;
} // end Destructor

/**
 * ******************* Initialize *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric_ss<TFixedImage,TMovingImage>
::Initialize(void) throw ( ExceptionObject )
{

    /** Initialize transform, interpolator, etc. */
    Superclass::Initialize();

    /** Retrieve slowest varying dimension and its size. */
    this->m_LastDimIndex = this->GetFixedImage()->GetImageDimension() - 1;
    this->m_G = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( m_LastDimIndex );

    if( this->m_NumEigenValues > this->m_G )
    {
        std::cerr << "ERROR: Number of eigenvalues is larger than number of images. Maximum number of eigenvalues equals: "
                  << this->m_G << std::endl;
    }
} // end Initializes


/**
 * ******************* PrintSelf *******************
 */

template < class TFixedImage, class TMovingImage>
void
PCAMetric_ss<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
    Superclass::PrintSelf( os, indent );

} // end PrintSelf

/**
 * ********************* InitializeThreadingParameters ****************************
 */

template< class TFixedImage, class TMovingImage >
void
PCAMetric_ss< TFixedImage, TMovingImage >
::InitializeThreadingParameters( void ) const
{
    /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   * Filling the potentially large vectors is performed later, in each thread,
   * which has performance benefits for larger vector sizes.
   */

    /** Only resize the array of structs when needed. */
    if( this->m_PCAMetricssGetSamplesPerThreadVariablesSize != this->m_NumberOfThreads )
    {
        delete[] this->m_PCAMetricssGetSamplesPerThreadVariables;
        this->m_PCAMetricssGetSamplesPerThreadVariables
                = new AlignedPCAMetricssGetSamplesPerThreadStruct[ this->m_NumberOfThreads ];
        this->m_PCAMetricssGetSamplesPerThreadVariablesSize = this->m_NumberOfThreads;
    }

    /** Some initialization. */
    for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
    {
        this->m_PCAMetricssGetSamplesPerThreadVariables[ i ].st_NumberOfPixelsCounted = NumericTraits< SizeValueType >::Zero;
        this->m_PCAMetricssGetSamplesPerThreadVariables[ i ].st_Derivative.SetSize( this->GetNumberOfParameters() );
    }

    this->m_PixelStartIndex.resize( this->m_NumberOfThreads );

} // end InitializeThreadingParameters()


/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template < class TFixedImage, class TMovingImage >
void
PCAMetric_ss<TFixedImage,TMovingImage>
::EvaluateTransformJacobianInnerProduct(
        const TransformJacobianType & jacobian,
        const MovingImageDerivativeType & movingImageDerivative,
        DerivativeType & imageJacobian ) const
{
    typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
    typedef typename DerivativeType::iterator              DerivativeIteratorType;
    JacobianIteratorType jac = jacobian.begin();
    imageJacobian.Fill( 0.0 );
    const unsigned int sizeImageJacobian = imageJacobian.GetSize();
    for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
    {
        const double imDeriv = movingImageDerivative[ dim ];
        DerivativeIteratorType imjac = imageJacobian.begin();

        for ( unsigned int mu = 0; mu < sizeImageJacobian; mu++ )
        {
            (*imjac) += (*jac) * imDeriv;
            ++imjac;
            ++jac;
        }
    }
} // end EvaluateTransformJacobianInnerProduct

/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
typename PCAMetric_ss<TFixedImage,TMovingImage>::MeasureType
PCAMetric_ss<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
    itkDebugMacro( "GetValue( " << parameters << " ) " );

    /** Call non-thread-safe stuff, such as:
     *   this->SetTransformParameters( parameters );
     *   this->GetImageSampler()->Update();
     * Because of these calls GetValueAndDerivative itself is not thread-safe,
     * so cannot be called multiple times simultaneously.
     * This is however needed in the CombinationImageToImageMetric.
     * In that case, you need to:
     * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
     * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
     *   calling GetValueAndDerivative
     * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
     * - Now you can call GetValueAndDerivative multi-threaded.
     */
    this->BeforeThreadedGetValueAndDerivative( parameters );

    /** Initialize some variables */
    this->m_NumberOfPixelsCounted = 0;
    MeasureType measure = NumericTraits< MeasureType >::Zero;

    /** Update the imageSampler and get a handle to the sample container. */
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
    unsigned int NumberOfSamples = sampleContainer->Size();
    MatrixType datablock( NumberOfSamples, this->m_G );

    /** Initialize dummy loop variable */
    unsigned int pixelIndex = 0;

    /** Initialize image sample matrix . */
    datablock.fill( itk::NumericTraits< RealType>::Zero );

    for ( fiter = fbegin; fiter != fend; ++fiter )
    {
        /** Read fixed coordinates. */
        FixedImagePointType fixedPoint = (*fiter).Value().m_ImageCoordinates;

        /** Transform sampled point to voxel coordinates. */
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

        unsigned int numSamplesOk = 0;

        /** Loop over t */
        for ( unsigned int d = 0; d < this->m_G; ++d )
        {
            /** Initialize some variables. */
            RealType movingImageValue;
            MovingImagePointType mappedPoint;

            /** Set fixed point's last dimension to lastDimPosition. */
            voxelCoord[ this->m_LastDimIndex ] = d;

            /** Transform sampled point back to world coordinates. */
            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );

            bool sampleOk = true;
            this->EvaluateMovingImageValueAndDerivative( fixedPoint, movingImageValue, 0 );

            /** Transform point and check if it is inside the B-spline support region. */
            if( d > (this->m_G-this->m_NumSingleSubjects -1) )
            {
                sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

                /** Check if point is inside mask. */
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }

                if( sampleOk )

                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(
                                mappedPoint, movingImageValue, 0 );
                }


                if( sampleOk )
                {
                    numSamplesOk++;
                    datablock( pixelIndex, d ) = movingImageValue;
                }
            }

        } /** end loop over t */

        if( numSamplesOk == this->m_G )
        {
            pixelIndex++;
            this->m_NumberOfPixelsCounted++;
        }

    }/** end first loop over image sample container */

    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples(NumberOfSamples, this->m_NumberOfPixelsCounted );
    MatrixType A( datablock.extract( this->m_NumberOfPixelsCounted, this->m_G ) );

    /** Calculate mean of from columns */
    vnl_vector< RealType > mean( this->m_G );
    mean.fill( NumericTraits< RealType >::Zero );
    for( int i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        for( int j = 0; j < this->m_G; j++)
        {
            mean(j) += A(i,j);
        }
    }
    mean /= RealType(this->m_NumberOfPixelsCounted);

    MatrixType Amm( this->m_NumberOfPixelsCounted, this->m_G );
    Amm.fill( NumericTraits< RealType >::Zero );

    for (int i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        for(int j = 0; j < this->m_G; j++)
        {
            Amm(i,j) = A(i,j)-mean(j);
        }
    }

    /** Compute covariancematrix C */
    MatrixType C( Amm.transpose()*Amm );
    C /=  static_cast< RealType > ( RealType(this->m_NumberOfPixelsCounted) - 1.0 );

    vnl_diag_matrix< RealType > S( this->m_G );
    S.fill( NumericTraits< RealType >::Zero );
    for( int j = 0; j < this->m_G; j++)
    {
        S(j,j) = 1.0/sqrt(C(j,j));
    }

    /** Compute correlation matrix K */
    MatrixType K(S*C*S);

    /** Compute first eigenvalue and eigenvector of K */
    vnl_symmetric_eigensystem< RealType > eig( K );

    RealType sumEigenValuesUsed = itk::NumericTraits< RealType >::Zero;
    for(unsigned int i = 1; i < this->m_NumEigenValues+1; i++)
    {
        sumEigenValuesUsed += eig.get_eigenvalue(this->m_G - i);
    }

    measure = this->m_G - sumEigenValuesUsed;

    /** Return the measure value. */
    return measure;

} // end GetValue

/**
 * ******************* GetDerivative *******************
 */

template < class TFixedImage, class TMovingImage>
void
PCAMetric_ss<TFixedImage,TMovingImage>
::GetDerivative( const TransformParametersType & parameters,
                 DerivativeType & derivative ) const
{
    /** When the derivative is calculated, all information for calculating
     * the metric value is available. It does not cost anything to calculate
     * the metric value now. Therefore, we have chosen to only implement the
     * GetValueAndDerivative(), supplying it with a dummy value variable. */
    MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;

    this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative

/**
     * ******************* GetValueAndDerivativeSingleThreaded *******************
     */

template <class TFixedImage, class TMovingImage>
void PCAMetric_ss<TFixedImage, TMovingImage>::GetValueAndDerivativeSingleThreaded( const TransformParametersType & parameters,
                                                                                   MeasureType& value, DerivativeType& derivative ) const
{
    itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

    /** Initialize some variables */
    this->m_NumberOfPixelsCounted = 0;
    MeasureType measure = NumericTraits< MeasureType >::Zero;
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

    /** Call non-thread-safe stuff, such as:
     *   this->SetTransformParameters( parameters );
     *   this->GetImageSampler()->Update();
     * Because of these calls GetValueAndDerivative itself is not thread-safe,
     * so cannot be called multiple times simultaneously.
     * This is however needed in the CombinationImageToImageMetric.
     * In that case, you need to:
     * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
     * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
     *   calling GetValueAndDerivative
     * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
     * - Now you can call GetValueAndDerivative multi-threaded.
     */
    this->BeforeThreadedGetValueAndDerivative( parameters );

    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    std::vector< FixedImagePointType > SamplesOK;

    /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
    unsigned int NumberOfSamples = sampleContainer->Size();
    MatrixType datablock( NumberOfSamples, this->m_G );

    /** Initialize dummy loop variables */
    unsigned int pixelIndex = 0;

    /** Initialize image sample matrix . */
    datablock.fill( itk::NumericTraits< RealType >::Zero );

    for ( fiter = fbegin; fiter != fend; ++fiter )
    {
        /** Read fixed coordinates. */
        FixedImagePointType fixedPoint = (*fiter).Value().m_ImageCoordinates;

        /** Transform sampled point to voxel coordinates. */
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

        unsigned int numSamplesOk = 0;

        /** Loop over t */
        for ( unsigned int d = 0; d < this->m_G; ++d )
        {
            /** Initialize some variables. */
            RealType movingImageValue;
            MovingImagePointType mappedPoint;

            /** Set fixed point's last dimension to lastDimPosition. */
            voxelCoord[this->m_LastDimIndex] = d;

            /** Transform sampled point back to world coordinates. */
            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );

            /** Transform point and check if it is inside the B-spline support region. */
            /** Only for d == G-1 **/
            bool sampleOk = true;
            this->EvaluateMovingImageValueAndDerivative( fixedPoint, movingImageValue, 0 );

            if( d > (this->m_G - this->m_NumSingleSubjects - 1) )
            {
                sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

                /** Check if point is inside mask. */
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }

                if( sampleOk )

                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(
                                mappedPoint, movingImageValue, 0 );
                }

                if( sampleOk )
                {
                    numSamplesOk++;
                    datablock( pixelIndex, d ) = movingImageValue;
                }// end if sampleOk
            }

        } // end loop over t
        if( numSamplesOk == this->m_G )
        {
            SamplesOK.push_back(fixedPoint);
            pixelIndex++;
            this->m_NumberOfPixelsCounted++;
        }

    }/** end first loop over image sample container */

    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples(	sampleContainer->Size(), this->m_NumberOfPixelsCounted );

    MatrixType A( datablock.extract( this->m_NumberOfPixelsCounted, this->m_G ) );

    /** Calculate mean of from columns */
    vnl_vector< RealType > mean( this->m_G );
    mean.fill( NumericTraits< RealType >::Zero );
    for( int i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        for( int j = 0; j < this->m_G; j++)
        {
            mean(j) += A(i,j);
        }
    }
    mean /= RealType(this->m_NumberOfPixelsCounted);

    /** Calculate standard deviation from columns */
    MatrixType Amm( this->m_NumberOfPixelsCounted, this->m_G );
    Amm.fill( NumericTraits< RealType >::Zero );
    for( int i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        for( int j = 0; j < this->m_G; j++)
        {
            Amm(i,j) = A(i,j)-mean(j);
        }
    }

    /** Compute covariancematrix C */
    MatrixType Atmm = Amm.transpose();
    MatrixType C( Atmm*Amm );
    C /=  static_cast< RealType > ( RealType(this->m_NumberOfPixelsCounted) - 1.0 );

    vnl_diag_matrix< RealType > S( this->m_G );
    S.fill( NumericTraits< RealType >::Zero );
    for( int j = 0; j < this->m_G; j++)
    {
        S(j,j) = 1.0/sqrt(C(j,j));
    }

    MatrixType K(S*C*S);

    /** Compute first eigenvalue and eigenvector of K */
    vnl_symmetric_eigensystem< RealType > eig( K );

    RealType sumEigenValuesUsed = itk::NumericTraits< RealType >::Zero;
    for(unsigned int i = 1; i < this->m_NumEigenValues+1; i++)
    {
        sumEigenValuesUsed += eig.get_eigenvalue(this->m_G - i);
    }

    MatrixType eigenVectorMatrix( this->m_G, this->m_NumEigenValues );
    for(unsigned int i = 1; i < this->m_NumEigenValues+1; i++)
    {
        eigenVectorMatrix.set_column(i-1, (eig.get_eigenvector(this->m_G - i)).normalize() );
    }

    MatrixType eigenVectorMatrixTranspose( eigenVectorMatrix.transpose() );

    /** Create variables to store intermediate results in. */
    TransformJacobianType jacobian;
    DerivativeType imageJacobian( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
    NonZeroJacobianIndicesType nzjis( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );

    /** Sub components of metric derivative */
    vnl_diag_matrix< DerivativeValueType > dSdmu_part1( this->m_G );

    /** initialize */
    dSdmu_part1.fill( itk::NumericTraits< DerivativeValueType >::Zero );

    for(unsigned int d = 0; d < this->m_G; d++)
    {
        double S_sqr = S(d,d) * S(d,d);
        double S_qub = S_sqr * S(d,d);
        dSdmu_part1(d, d) = -S_qub;
    }

    this->m_vSAtmm = eigenVectorMatrixTranspose*S*this->m_Atmm;
    this->m_CSv = C*S*eigenVectorMatrix;
    this->m_Sv = S*eigenVectorMatrix;
    this->m_vdSdmu_part1 = eigenVectorMatrixTranspose*dSdmu_part1;

    /** Second loop over fixed image samples. */
    for ( pixelIndex = 0; pixelIndex < SamplesOK.size(); ++pixelIndex )
    {
        /** Read fixed coordinates. */
        FixedImagePointType fixedPoint = SamplesOK[ pixelIndex ];

        /** Transform sampled point to voxel coordinates. */
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

        /** Initialize some variables. */
        RealType movingImageValue;
        MovingImagePointType mappedPoint;
        MovingImageDerivativeType movingImageDerivative;

        /** Set fixed point's last dimension to lastDimPosition. */
        for( unsigned int d = (this->m_G-this->m_NumSingleSubjects); d < this->m_G; ++d)
        {
            voxelCoord[ this->m_LastDimIndex ] = d;

            /** Transform sampled point back to world coordinates. */
            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );

            this->TransformPoint( fixedPoint, mappedPoint );

            this->EvaluateMovingImageValueAndDerivative(
                        mappedPoint, movingImageValue, &movingImageDerivative );

            /** Get the TransformJacobian dT/dmu */
            this->EvaluateTransformJacobian( fixedPoint, jacobian, nzjis );

            /** Compute the innerproduct (dM/dx)^T (dT/dmu). */
            this->EvaluateTransformJacobianInnerProduct(
                        jacobian, movingImageDerivative, imageJacobian );

            /** build metric derivative components */
            for( unsigned int p = 0; p < nzjis.size(); ++p)
            {
                DerivativeValueType tmp = 0.0;
                for(unsigned int z = 0; z < this->m_NumEigenValues; z++)
                {
                    tmp += this->m_vSAtmm[ z ][ pixelIndex ] * imageJacobian[ p ] * this->m_Sv[ d ][ z ] +
                            this->m_vdSdmu_part1[ z ][ d ] * this->m_Atmm[ d ][ pixelIndex ] * imageJacobian[ p ] * this->m_CSv[ d ][ z ];
                }//end loop over eigenvalues
                derivative[ nzjis[ p ] ] += tmp;
            }//end loop over non-zero jacobian indices
        }

    } // end second for loop over sample container

    derivative *= -(2.0/(DerivativeValueType(this->m_NumberOfPixelsCounted) - 1.0)); //normalize

    value = this->m_G - sumEigenValuesUsed;

} // end GetValueAndDerivativeSingleThreaded()

/**
  * ******************* GetValueAndDerivative *******************
  */

template< class TFixedImage, class TMovingImage >
void
PCAMetric_ss< TFixedImage, TMovingImage >
::GetValueAndDerivative(
        const TransformParametersType & parameters,
        MeasureType & value, DerivativeType & derivative ) const
{
    /** Option for now to still use the single threaded code. */
    if( !this->m_UseMultiThread )
    {
        return this->GetValueAndDerivativeSingleThreaded(
                    parameters, value, derivative );
    }

    /** Call non-thread-safe stuff, such as:
       *   this->SetTransformParameters( parameters );
       *   this->GetImageSampler()->Update();
       * Because of these calls GetValueAndDerivative itself is not thread-safe,
       * so cannot be called multiple times simultaneously.
       * This is however needed in the CombinationImageToImageMetric.
       * In that case, you need to:
       * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
       * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
       *   calling GetValueAndDerivative
       * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
       * - Now you can call GetValueAndDerivative multi-threaded.
       */
    this->BeforeThreadedGetValueAndDerivative( parameters );


    this->InitializeThreadingParameters();

    /** Launch multi-threading GetSamples */
    this->LaunchGetSamplesThreaderCallback();

    /** Get the metric value contributions from all threads. */
    this->AfterThreadedGetSamples( value );

    /** Launch multi-threading ComputeDerivative */
    this->LaunchComputeDerivativeThreaderCallback();

    /** Sum derivative contributions from all threads */
    this->AfterThreadedComputeDerivative( derivative );


} // end GetValueAndDerivative()

/**
 * ******************* ThreadedGetSamples *******************
 */

template< class TFixedImage, class TMovingImage >
void
PCAMetric_ss< TFixedImage, TMovingImage >
::ThreadedGetSamples( ThreadIdType threadId )
{
    /** Get a handle to the sample container. */
    ImageSampleContainerPointer sampleContainer     = this->GetImageSampler()->GetOutput();
    const unsigned long         sampleContainerSize = sampleContainer->Size();

    /** Get the samples for this thread. */
    const unsigned long nrOfSamplesPerThreads
            = static_cast< unsigned long >( vcl_ceil( static_cast< double >( sampleContainerSize )
                                                      / static_cast< double >( this->m_NumberOfThreads ) ) );
    unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
    unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
    pos_begin = ( pos_begin > sampleContainerSize ) ? sampleContainerSize : pos_begin;
    pos_end   = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator threader_fiter;
    typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator threader_fend   = sampleContainer->Begin();
    threader_fbegin += (int)pos_begin;
    threader_fend   += (int)pos_end;

    std::vector< FixedImagePointType > SamplesOK;
    MatrixType datablock( nrOfSamplesPerThreads, this->m_G );

    unsigned int pixelIndex = 0;
    for( threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter )
    {
        /** Read fixed coordinates. */
        FixedImagePointType fixedPoint = (*threader_fiter).Value().m_ImageCoordinates;

        /** Transform sampled point to voxel coordinates. */
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

        unsigned int numSamplesOk = 0;

        /** Loop over t */
        for ( unsigned int d = 0; d < this->m_G; ++d )
        {
            /** Initialize some variables. */
            RealType movingImageValue;
            MovingImagePointType mappedPoint;

            /** Set fixed point's last dimension to lastDimPosition. */
            voxelCoord[ this->m_LastDimIndex ] = d;

            /** Transform sampled point back to world coordinates. */
            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );

            /** Transform point and check if it is inside the B-spline support region. */
            /** Only for d > G-num_ss **/
            bool sampleOk = true;
            this->EvaluateMovingImageValueAndDerivative( fixedPoint, movingImageValue, 0 );

            if( d > (this->m_G - this->m_NumSingleSubjects - 1) )
            {
                sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

                /** Check if point is inside mask. */
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }

                if( sampleOk )

                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(
                                mappedPoint, movingImageValue, 0 );
                }
            }
            if( sampleOk )
            {
                numSamplesOk++;
                datablock( pixelIndex, d ) = movingImageValue;
            }// end if sampleOk

        } // end loop over t
        if( numSamplesOk == this->m_G )
        {
            SamplesOK.push_back(fixedPoint);
            pixelIndex++;
        }

    }/** end first loop over image sample container */
    /** Only update these variables at the end to prevent unnecessary "false sharing". */
    this->m_PCAMetricssGetSamplesPerThreadVariables[ threadId ].st_NumberOfPixelsCounted = pixelIndex;
    this->m_PCAMetricssGetSamplesPerThreadVariables[ threadId ].st_DataBlock             = datablock.extract( pixelIndex, this->m_G );
    this->m_PCAMetricssGetSamplesPerThreadVariables[ threadId ].st_ApprovedSamples       = SamplesOK;

} // end ThreadedGetSamples()

/**
 * ******************* AfterThreadedGetSamples *******************
 */

template< class TFixedImage, class TMovingImage >
void
PCAMetric_ss< TFixedImage, TMovingImage >
::AfterThreadedGetSamples(MeasureType & value) const
{
    /** Accumulate the number of pixels. */
    this->m_NumberOfPixelsCounted = this->m_PCAMetricssGetSamplesPerThreadVariables[ 0 ].st_NumberOfPixelsCounted;
    for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
    {
        this->m_NumberOfPixelsCounted += this->m_PCAMetricssGetSamplesPerThreadVariables[ i ].st_NumberOfPixelsCounted;
    }

    /** Check if enough samples were valid. */
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
    this->CheckNumberOfSamples(
                sampleContainer->Size(), this->m_NumberOfPixelsCounted );

    MatrixType A( this->m_NumberOfPixelsCounted, this->m_G );
    unsigned int row_start = 0;
    for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
    {
        A.update(this->m_PCAMetricssGetSamplesPerThreadVariables[ i ].st_DataBlock, row_start, 0);
        this->m_PixelStartIndex[ i ] = row_start;
        row_start += this->m_PCAMetricssGetSamplesPerThreadVariables[ i ].st_DataBlock.rows();
    }


    /** Calculate mean of from columns */
    vnl_vector< RealType > mean( this->m_G );
    mean.fill( NumericTraits< RealType >::Zero );
    for( int i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        for( int j = 0; j < this->m_G; j++)
        {
            mean(j) += A(i,j);
        }
    }
    mean /= RealType(this->m_NumberOfPixelsCounted);

    /** Calculate standard deviation from columns */
    MatrixType Amm( this->m_NumberOfPixelsCounted, this->m_G );
    Amm.fill( NumericTraits< RealType >::Zero );
    for( int i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        for( int j = 0; j < this->m_G; j++)
        {
            Amm(i,j) = A(i,j)-mean(j);
        }
    }

    /** Compute covariancematrix C */
    this->m_Atmm = Amm.transpose();
    MatrixType C( this->m_Atmm*Amm );
    C /=  static_cast< RealType > ( RealType(this->m_NumberOfPixelsCounted) - 1.0 );

    vnl_diag_matrix< RealType > S( this->m_G );
    S.fill( NumericTraits< RealType >::Zero );
    for( int j = 0; j < this->m_G; j++)
    {
        S(j,j) = 1.0/sqrt(C(j,j));
    }

    MatrixType K(S*C*S);

    /** Compute first eigenvalue and eigenvector of K */
    vnl_symmetric_eigensystem< RealType > eig( K );

    RealType sumEigenValuesUsed = itk::NumericTraits< RealType >::Zero;
    MatrixType eigenVectorMatrix( this->m_G, this->m_NumEigenValues );
    for(unsigned int i = 1; i < this->m_NumEigenValues+1; i++)
    {
        sumEigenValuesUsed += eig.get_eigenvalue(this->m_G - i);
        eigenVectorMatrix.set_column(i-1, (eig.get_eigenvector(this->m_G - i)).normalize() );
    }

    value = this->m_G - sumEigenValuesUsed;

    MatrixType eigenVectorMatrixTranspose( eigenVectorMatrix.transpose() );

    /** Sub components of metric derivative */
    vnl_diag_matrix< DerivativeValueType > dSdmu_part1( this->m_G );

    for(unsigned int d = 0; d < this->m_G; d++)
    {
        double S_sqr = S(d,d) * S(d,d);
        double S_qub = S_sqr * S(d,d);
        dSdmu_part1(d, d) = -S_qub;
    }

    this->m_vSAtmm = eigenVectorMatrixTranspose*S*this->m_Atmm;
    this->m_CSv = C*S*eigenVectorMatrix;
    this->m_Sv = S*eigenVectorMatrix;
    this->m_vdSdmu_part1 = eigenVectorMatrixTranspose*dSdmu_part1;

} // end AfterThreadedGetSamples()


/**
 * **************** GetSamplesThreaderCallback *******
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
PCAMetric_ss< TFixedImage, TMovingImage >
::GetSamplesThreaderCallback( void * arg )
{
    ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
    ThreadIdType     threadId   = infoStruct->ThreadID;

    PCAMetricssMultiThreaderParameterType * temp
            = static_cast< PCAMetricssMultiThreaderParameterType * >( infoStruct->UserData );

    temp->m_Metric->ThreadedGetSamples( threadId );

    return ITK_THREAD_RETURN_VALUE;

} // GetSamplesThreaderCallback()


/**
 * *********************** LaunchGetSamplesThreaderCallback***************
 */

template< class TFixedImage, class TMovingImage >
void
PCAMetric_ss< TFixedImage, TMovingImage >
::LaunchGetSamplesThreaderCallback( void ) const
{
    /** Setup local threader. */
    // \todo: is a global threader better performance-wise? check
    typename ThreaderType::Pointer local_threader = ThreaderType::New();
    local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
    local_threader->SetSingleMethod( this->GetSamplesThreaderCallback,
                                     const_cast< void * >( static_cast< const void * >(
                                                               &this->m_PCAMetricssThreaderParameters ) ) );
    /** Launch. */
    local_threader->SingleMethodExecute();

} // end LaunchGetSamplesThreaderCallback()


/**
 * ******************* ThreadedComputeDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
PCAMetric_ss< TFixedImage, TMovingImage >
::ThreadedComputeDerivative( ThreadIdType threadId )
{
    /** Create variables to store intermediate results in. */
    DerivativeType & derivative = this->m_PCAMetricssGetSamplesPerThreadVariables[ threadId ].st_Derivative;
    derivative.Fill(0.0);

    /** Initialize some variables. */
    RealType movingImageValue;
    MovingImagePointType mappedPoint;
    MovingImageDerivativeType movingImageDerivative;

    TransformJacobianType jacobian;
    DerivativeType imageJacobian( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
    NonZeroJacobianIndicesType nzjis( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );

    unsigned int dummyindex = 0;
    /** Second loop over fixed image samples. */
    for ( unsigned int pixelIndex = this->m_PixelStartIndex[ threadId ]; pixelIndex < (this->m_PixelStartIndex[ threadId ]+this->m_PCAMetricssGetSamplesPerThreadVariables[ threadId ].st_ApprovedSamples.size()); ++pixelIndex )
    {
        /** Read fixed coordinates. */
        FixedImagePointType fixedPoint = this->m_PCAMetricssGetSamplesPerThreadVariables[ threadId ].st_ApprovedSamples[ dummyindex ];

        /** Transform sampled point to voxel coordinates. */
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

        /** Set fixed point's last dimension to lastDimPosition. */
        for( unsigned int d = (this->m_G - this->m_NumSingleSubjects); d < this->m_G; ++d)
        {
            voxelCoord[ this->m_LastDimIndex ] = d;

            /** Transform sampled point back to world coordinates. */
            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
            this->TransformPoint( fixedPoint, mappedPoint );

            this->EvaluateMovingImageValueAndDerivative(
                        mappedPoint, movingImageValue, &movingImageDerivative );

            /** Get the TransformJacobian dT/dmu */
            this->EvaluateTransformJacobian( fixedPoint, jacobian, nzjis );

            /** Compute the innerproduct (dM/dx)^T (dT/dmu). */
            this->EvaluateTransformJacobianInnerProduct(
                        jacobian, movingImageDerivative, imageJacobian );

            /** build metric derivative components */
            for( unsigned int p = 0; p < nzjis.size(); ++p)
            {
                DerivativeValueType tmp = 0.0;
                for(unsigned int z = 0; z < this->m_NumEigenValues; z++)
                {
                    tmp += this->m_vSAtmm[ z ][ pixelIndex ] * imageJacobian[ p ] * this->m_Sv[ d ][ z ] +
                            this->m_vdSdmu_part1[ z ][ d ] * this->m_Atmm[ d ][ pixelIndex ] * imageJacobian[ p ] * this->m_CSv[ d ][ z ];
                }//end loop over eigenvalues
                derivative[ nzjis[ p ] ] += tmp;
            }//end loop over non-zero jacobian indices
        }

        dummyindex++;

    } // end second for loop over sample container


} // end ThreadedGetValueAndDerivative()

/**
 * ******************* AfterThreadedComputeDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
PCAMetric_ss< TFixedImage, TMovingImage >
::AfterThreadedComputeDerivative(
        DerivativeType & derivative ) const
{
    derivative = this->m_PCAMetricssGetSamplesPerThreadVariables[ 0 ].st_Derivative;
    for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
    {
        derivative += this->m_PCAMetricssGetSamplesPerThreadVariables[ i ].st_Derivative;
    }

    derivative *= -(2.0/(DerivativeValueType(this->m_NumberOfPixelsCounted) - 1.0)); //normalize

}// end AftherThreadedComputeDerivative()

/**
 * **************** ComputeDerivativeThreaderCallback *******
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
PCAMetric_ss< TFixedImage, TMovingImage >
::ComputeDerivativeThreaderCallback( void * arg )
{
    ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
    ThreadIdType     threadId   = infoStruct->ThreadID;

    PCAMetricssMultiThreaderParameterType * temp
            = static_cast< PCAMetricssMultiThreaderParameterType * >( infoStruct->UserData );

    temp->m_Metric->ThreadedComputeDerivative( threadId );

    return ITK_THREAD_RETURN_VALUE;

} // end omputeDerivativeThreaderCallback()

/**
 * ************** LaunchComputeDerivativeThreaderCallback **********
 */

template< class TFixedImage, class TMovingImage >
void
PCAMetric_ss< TFixedImage, TMovingImage >
::LaunchComputeDerivativeThreaderCallback( void ) const
{
    /** Setup local threader. */
    // \todo: is a global threader better performance-wise? check
    typename ThreaderType::Pointer local_threader = ThreaderType::New();
    local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
    local_threader->SetSingleMethod( this->ComputeDerivativeThreaderCallback,
                                     const_cast< void * >( static_cast< const void * >(
                                                               &this->m_PCAMetricssThreaderParameters ) ) );

    /** Launch. */
    local_threader->SingleMethodExecute();

} // end LaunchComputeDerivativeThreaderCallback()

} // end namespace itk

#endif // ITKPCAMETRIC_F_MULTITHREADED_HXX

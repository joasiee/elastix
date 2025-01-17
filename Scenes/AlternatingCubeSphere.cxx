#include "itkSpatialObject.h"
#include "itkEllipseSpatialObject.h"
#include "itkBoxSpatialObject.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkSpatialObjectToImageFilter.h"
#include "elx_config.h"

constexpr char         SceneId[]{ "02" };
constexpr unsigned int Dimension = 3;

constexpr int Padding = 5;
constexpr int CubeSizeMoving = 20;
constexpr int CubeSizeFixed = 0.6f * CubeSizeMoving;
constexpr int ImageSize = CubeSizeMoving + 2 * Padding;

constexpr int SphereCenter = CubeSizeMoving / 2.0f + Padding;
constexpr int SphereRadiusMoving = CubeSizeFixed / 2.0f;
constexpr int SphereRadiusFixed = 1.3f * CubeSizeMoving / 2.0f;

constexpr int CubeIntensity = 200;
constexpr int SphereIntensity = 100;
constexpr int OutsideIntensity = 0;

using PixelType = unsigned char;
using ImageType = itk::Image<PixelType, Dimension>;
using SpatialObjectType = itk::SpatialObject<Dimension>;
using EllipseType = itk::EllipseSpatialObject<Dimension>;
using BoxType = itk::BoxSpatialObject<Dimension>;

static void
CreateFixedImage(ImageType::Pointer image);
static void
CreateFixedImageMask(ImageType::Pointer image);
static void
CreateMovingImage(ImageType::Pointer image);

int
main(int, char *[])
{
  ImageType::Pointer fixedImage = ImageType::New();
  ImageType::Pointer fixedImageMask = ImageType::New();
  ImageType::Pointer movingImage = ImageType::New();
  std::string        fixedPath;
  std::string        fixedMaskPath;
  std::string        movingPath;

  CreateFixedImage(fixedImage);
  CreateFixedImageMask(fixedImageMask);
  CreateMovingImage(movingImage);

  {
    std::ostringstream oss;
    oss << elastix_BINARY_DIR << "/Scenes/" << SceneId << "_Fixed.mhd";
    fixedPath = oss.str();
  }

  {
    std::ostringstream oss;
    oss << elastix_BINARY_DIR << "/Scenes/" << SceneId << "_FixedMask.mhd";
    fixedMaskPath = oss.str();
  }

  {
    std::ostringstream oss;
    oss << elastix_BINARY_DIR << "/Scenes/" << SceneId << "_Moving.mhd";
    movingPath = oss.str();
  }

  // Write the two synthetic inputs
  itk::WriteImage(fixedImage, fixedPath);
  itk::WriteImage(fixedImageMask, fixedMaskPath);
  itk::WriteImage(movingImage, movingPath);

  return EXIT_SUCCESS;
}

void
CreateMovingImage(ImageType::Pointer image)
{
  using SpatialObjectToImageFilterType = itk::SpatialObjectToImageFilter<EllipseType, ImageType>;

  // image filter
  SpatialObjectToImageFilterType::Pointer imageFilter = SpatialObjectToImageFilterType::New();

  ImageType::SizeType size;
  size.Fill(ImageSize);
  imageFilter->SetSize(size);

  ImageType::SpacingType spacing;
  spacing.Fill(1);
  imageFilter->SetSpacing(spacing);

  // cube
  BoxType::Pointer cube = BoxType::New();

  BoxType::SizeType boxSize;
  boxSize.Fill(CubeSizeMoving - 1);
  cube->SetSizeInObjectSpace(boxSize);

  BoxType::PointType boxPosition;
  boxPosition.Fill(Padding);
  cube->SetPositionInObjectSpace(boxPosition);

  // inner sphere
  EllipseType::Pointer sphere = EllipseType::New();

  EllipseType::ArrayType sphereRadius;
  sphereRadius.Fill(SphereRadiusMoving);
  sphere->SetRadiusInObjectSpace(sphereRadius);

  EllipseType::PointType sphereCenter;
  sphereCenter.Fill(SphereCenter - 0.5);
  sphere->SetCenterInObjectSpace(sphereCenter);

  // image intensities
  cube->SetDefaultInsideValue(CubeIntensity);
  cube->SetDefaultOutsideValue(OutsideIntensity);
  sphere->SetDefaultInsideValue(SphereIntensity);

  sphere->AddChild(cube);
  sphere->Update();

  // write to image
  imageFilter->SetUseObjectValue(true);

  imageFilter->SetInput(sphere);
  imageFilter->Update();
  image->Graft(imageFilter->GetOutput());
}

void
CreateFixedImage(ImageType::Pointer image)
{
  using SpatialObjectToImageFilterType = itk::SpatialObjectToImageFilter<BoxType, ImageType>;

  // image filter
  SpatialObjectToImageFilterType::Pointer imageFilter = SpatialObjectToImageFilterType::New();

  ImageType::SizeType size;
  size.Fill(ImageSize);
  imageFilter->SetSize(size);

  ImageType::SpacingType spacing;
  spacing.Fill(1);
  imageFilter->SetSpacing(spacing);

  // outer sphere
  EllipseType::Pointer sphere = EllipseType::New();

  EllipseType::ArrayType sphereRadius;
  sphereRadius.Fill(SphereRadiusFixed);
  sphere->SetRadiusInObjectSpace(sphereRadius);

  EllipseType::PointType sphereCenter;
  sphereCenter.Fill(SphereCenter - 0.5);
  sphere->SetCenterInObjectSpace(sphereCenter);

  // inner cube
  BoxType::Pointer cube = BoxType::New();

  BoxType::SizeType boxSize;
  boxSize.Fill(CubeSizeFixed - 1);
  cube->SetSizeInObjectSpace(boxSize);

  BoxType::PointType boxPosition;
  boxPosition.Fill(Padding + (CubeSizeMoving - CubeSizeFixed) / 2.0f);
  cube->SetPositionInObjectSpace(boxPosition);

  // image intensities
  sphere->SetDefaultInsideValue(CubeIntensity);
  sphere->SetDefaultOutsideValue(OutsideIntensity);
  cube->SetDefaultInsideValue(SphereIntensity);

  cube->AddChild(sphere);
  cube->Update();

  // write to image
  imageFilter->SetUseObjectValue(true);

  imageFilter->SetInput(cube);
  imageFilter->Update();
  image->Graft(imageFilter->GetOutput());
}

void
CreateFixedImageMask(ImageType::Pointer image)
{
  using SpatialObjectType = itk::SpatialObject<Dimension>;
  using EllipseType = itk::EllipseSpatialObject<Dimension>;
  using SpatialObjectToImageFilterType = itk::SpatialObjectToImageFilter<EllipseType, ImageType>;

  // image filter
  SpatialObjectToImageFilterType::Pointer imageFilter = SpatialObjectToImageFilterType::New();

  ImageType::SizeType size;
  size.Fill(ImageSize);
  imageFilter->SetSize(size);

  ImageType::SpacingType spacing;
  spacing.Fill(1);
  imageFilter->SetSpacing(spacing);

  // outer sphere
  EllipseType::Pointer sphere = EllipseType::New();

  EllipseType::ArrayType sphereRadius;
  sphereRadius.Fill(SphereRadiusFixed);
  sphere->SetRadiusInObjectSpace(sphereRadius);

  EllipseType::PointType sphereCenter;
  sphereCenter.Fill(SphereCenter - 0.5);
  sphere->SetCenterInObjectSpace(sphereCenter);

  // image intensities
  sphere->SetDefaultInsideValue(1);
  sphere->SetDefaultOutsideValue(0);

  sphere->Update();

  // write to image
  imageFilter->SetUseObjectValue(true);

  imageFilter->SetInput(sphere);
  imageFilter->Update();
  image->Graft(imageFilter->GetOutput());
}
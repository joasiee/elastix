#include "itkSpatialObject.h"
#include "itkEllipseSpatialObject.h"
#include "itkBoxSpatialObject.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkSpatialObjectToImageFilter.h"

constexpr unsigned int Dimension = 3;

constexpr int Padding = 2;
constexpr int CubeSize = 25;
constexpr int ImageSize = CubeSize + 2 * Padding;

constexpr int SphereCenter = CubeSize / 2.0f + Padding;
constexpr int SphereRadiusMoving = 0.9f * CubeSize / 2.0f;
constexpr int SphereRadiusFixed = SphereRadiusMoving / 2.0f;

constexpr int CubeIntensity = 200;
constexpr int SphereIntensity = 100;
constexpr int OutsideIntensity = 0;

using PixelType = unsigned char;

using ImageType = itk::Image<PixelType, Dimension>;

static void
CreateFixedImage(ImageType::Pointer image);
static void
CreateMovingImage(ImageType::Pointer image);

int
main(int, char *[])
{
  // Get the two images
  ImageType::Pointer fixedImage = ImageType::New();
  ImageType::Pointer movingImage = ImageType::New();

  CreateFixedImage(fixedImage);
  CreateMovingImage(movingImage);

  // Write the two synthetic inputs
  itk::WriteImage(fixedImage, "../Scenes/01_Fixed.mhd");
  itk::WriteImage(movingImage, "../Scenes/01_Moving.mhd");

  return EXIT_SUCCESS;
}

void
CreateMovingImage(ImageType::Pointer image)
{
  using SpatialObjectType = itk::SpatialObject<Dimension>;
  using EllipseType = itk::EllipseSpatialObject<Dimension>;
  using BoxType = itk::BoxSpatialObject<Dimension>;
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
  boxSize.Fill(CubeSize - 1);
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
  sphereCenter.Fill(SphereCenter);
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
  using SpatialObjectType = itk::SpatialObject<Dimension>;
  using EllipseType = itk::EllipseSpatialObject<Dimension>;
  using BoxType = itk::BoxSpatialObject<Dimension>;
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
  boxSize.Fill(CubeSize - 1);
  cube->SetSizeInObjectSpace(boxSize);

  BoxType::PointType boxPosition;
  boxPosition.Fill(Padding);
  cube->SetPositionInObjectSpace(boxPosition);

  // inner sphere
  EllipseType::Pointer sphere = EllipseType::New();

  EllipseType::ArrayType sphereRadius;
  sphereRadius.Fill(SphereRadiusFixed);
  sphere->SetRadiusInObjectSpace(sphereRadius);

  EllipseType::PointType sphereCenter;
  sphereCenter.Fill(SphereCenter);
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
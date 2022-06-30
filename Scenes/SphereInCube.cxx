#include "itkSpatialObject.h"
#include "itkEllipseSpatialObject.h"
#include "itkBoxSpatialObject.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkSpatialObjectToImageFilter.h"

constexpr unsigned int Dimension = 3;
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
  itk::WriteImage(fixedImage, "../Scenes/fixed.mhd");
  itk::WriteImage(movingImage, "../Scenes/moving.mhd");

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
  ImageType::SizeType                     size;
  size.Fill(40);
  imageFilter->SetSize(size);
  ImageType::SpacingType spacing;
  spacing.Fill(1);
  imageFilter->SetSpacing(spacing);

  // cube
  BoxType::Pointer cube = BoxType::New();

  BoxType::SizeType boxSize;
  boxSize.Fill(35);
  cube->SetSizeInObjectSpace(boxSize);

  BoxType::PointType boxPosition;
  boxPosition.Fill(2.5);
  cube->SetPositionInObjectSpace(boxPosition);

  // inner sphere
  EllipseType::Pointer sphere = EllipseType::New();

  EllipseType::ArrayType sphereRadius;
  sphereRadius.Fill(14);
  sphere->SetRadiusInObjectSpace(sphereRadius);

  EllipseType::PointType sphereCenter;
  sphereCenter.Fill(20);
  sphere->SetCenterInObjectSpace(sphereCenter);

  // image intensities
  cube->SetDefaultInsideValue(255);
  cube->SetDefaultOutsideValue(0);
  sphere->SetDefaultInsideValue(100);

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
  ImageType::SizeType                     size;
  size.Fill(40);
  imageFilter->SetSize(size);
  ImageType::SpacingType spacing;
  spacing.Fill(1);
  imageFilter->SetSpacing(spacing);

  // cube
  BoxType::Pointer cube = BoxType::New();

  BoxType::SizeType boxSize;
  boxSize.Fill(35);
  cube->SetSizeInObjectSpace(boxSize);

  BoxType::PointType boxPosition;
  boxPosition.Fill(2.5);
  cube->SetPositionInObjectSpace(boxPosition);

  // inner sphere
  EllipseType::Pointer sphere = EllipseType::New();

  EllipseType::ArrayType sphereRadius;
  sphereRadius.Fill(7);
  sphere->SetRadiusInObjectSpace(sphereRadius);

  EllipseType::PointType sphereCenter;
  sphereCenter.Fill(20);
  sphere->SetCenterInObjectSpace(sphereCenter);

  // image intensities
  cube->SetDefaultInsideValue(255);
  cube->SetDefaultOutsideValue(0);
  sphere->SetDefaultInsideValue(100);

  sphere->AddChild(cube);
  sphere->Update();

  // write to image
  imageFilter->SetUseObjectValue(true);

  imageFilter->SetInput(sphere);
  imageFilter->Update();
  image->Graft(imageFilter->GetOutput());
}
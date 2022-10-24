#include "itkSpatialObject.h"
#include "itkEllipseSpatialObject.h"
#include "itkBoxSpatialObject.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkSpatialObjectToImageFilter.h"
#include "elx_config.h"

constexpr char SceneId[]{ "01" };

constexpr unsigned int Dimension = 3;

constexpr int Padding = 2;
constexpr int CubeSize = 20;
constexpr int ImageSize = CubeSize + 2 * Padding;

constexpr int SphereCenter = CubeSize / 2.0f + Padding;
constexpr int SphereRadiusMoving = 0.8f * CubeSize / 2.0f;
constexpr int SphereRadiusFixed = 0.5f * CubeSize / 2.0f;

constexpr int CubeIntensity = 50;
constexpr int SphereIntensity = 100;
constexpr int OutsideIntensity = 0;

using PixelType = unsigned char;

using ImageType = itk::Image<PixelType, Dimension>;

static void
CreateFixedImage(ImageType::Pointer image);
static void
CreateFixedImageMask(ImageType::Pointer image);
static void
CreateMovingImage(ImageType::Pointer image);

int
main(int, char *[])
{
  // Get the two images
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
CreateFixedImageMask(ImageType::Pointer image)
{
  using SpatialObjectType = itk::SpatialObject<Dimension>;
  using BoxType = itk::BoxSpatialObject<Dimension>;
  using SpatialObjectToImageFilterType = itk::SpatialObjectToImageFilter<BoxType, ImageType>;

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

  // image intensities
  cube->SetDefaultInsideValue(1);
  cube->SetDefaultOutsideValue(0);

  cube->Update();

  // write to image
  imageFilter->SetUseObjectValue(true);

  imageFilter->SetInput(cube);
  imageFilter->Update();
  image->Graft(imageFilter->GetOutput());
}
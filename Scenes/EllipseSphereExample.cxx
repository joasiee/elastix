#include "itkEllipseSpatialObject.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkSpatialObjectToImageFilter.h"

constexpr unsigned int Dimension = 3;
using PixelType = unsigned char;

using ImageType = itk::Image<PixelType, Dimension>;

static void
CreateEllipseImage(ImageType::Pointer image);
static void
CreateSphereImage(ImageType::Pointer image);

int
main(int, char *[])
{
  // Get the two images
  ImageType::Pointer fixedImage = ImageType::New();
  ImageType::Pointer movingImage = ImageType::New();

  CreateSphereImage(fixedImage);
  CreateEllipseImage(movingImage);

  // Write the two synthetic inputs
  itk::WriteImage(fixedImage, "../Scenes/fixed.mha");
  itk::WriteImage(movingImage, "../Scenes/moving.mha");

  return EXIT_SUCCESS;
}

void
CreateEllipseImage(ImageType::Pointer image)
{
  using EllipseType = itk::EllipseSpatialObject<Dimension>;

  using SpatialObjectToImageFilterType = itk::SpatialObjectToImageFilter<EllipseType, ImageType>;

  SpatialObjectToImageFilterType::Pointer imageFilter = SpatialObjectToImageFilterType::New();

  ImageType::SizeType size;
  size.Fill(100);

  imageFilter->SetSize(size);

  ImageType::SpacingType spacing;
  spacing.Fill(1);
  imageFilter->SetSpacing(spacing);

  EllipseType::Pointer   ellipse = EllipseType::New();
  EllipseType::ArrayType radiusArray;
  for (int i = 0; i < Dimension; ++i)
  {
    radiusArray[i] = 10 + (i > 0) * 10;
  }
  ellipse->SetRadiusInObjectSpace(radiusArray);

  using TransformType = EllipseType::TransformType;
  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();

  TransformType::OutputVectorType translation;
  for (int i = 0; i < Dimension; ++i)
  {
    translation[i] = 65 - (i * 15);
  }
  transform->Translate(translation, false);

  ellipse->SetObjectToParentTransform(transform);

  imageFilter->SetInput(ellipse);

  ellipse->SetDefaultInsideValue(255);
  ellipse->SetDefaultOutsideValue(0);
  imageFilter->SetUseObjectValue(true);
  imageFilter->SetOutsideValue(0);

  imageFilter->Update();

  image->Graft(imageFilter->GetOutput());
}

void
CreateSphereImage(ImageType::Pointer image)
{
  using EllipseType = itk::EllipseSpatialObject<Dimension>;

  using SpatialObjectToImageFilterType = itk::SpatialObjectToImageFilter<EllipseType, ImageType>;

  SpatialObjectToImageFilterType::Pointer imageFilter = SpatialObjectToImageFilterType::New();

  ImageType::SizeType size;
  size.Fill(100);

  imageFilter->SetSize(size);

  ImageType::SpacingType spacing;
  spacing.Fill(1);
  imageFilter->SetSpacing(spacing);

  EllipseType::Pointer   ellipse = EllipseType::New();
  EllipseType::ArrayType radiusArray;
  for (int i = 0; i < Dimension; ++i)
  {
    radiusArray[i] = 10;
  }
  ellipse->SetRadiusInObjectSpace(radiusArray);

  using TransformType = EllipseType::TransformType;
  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();

  TransformType::OutputVectorType translation;
  for (int i = 0; i < Dimension; ++i)
  {
    translation[i] = 50;
  }
  transform->Translate(translation, false);

  ellipse->SetObjectToParentTransform(transform);

  imageFilter->SetInput(ellipse);

  ellipse->SetDefaultInsideValue(255);
  ellipse->SetDefaultOutsideValue(0);
  imageFilter->SetUseObjectValue(true);
  imageFilter->SetOutsideValue(0);

  imageFilter->Update();

  image->Graft(imageFilter->GetOutput());
}
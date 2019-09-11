// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

namespace Windows::AI::MachineLearning::Internal {
const UINT kImageTensorDimensionCountMax = 4;  // NCHW format

enum ImageTensorDataType {
  kImageTensorDataTypeFloat32,
  kImageTensorDataTypeFloat16,
  kImageTensorDataTypeUInt32,
  kImageTensorDataTypeUInt16,
  kImageTensorDataTypeUInt8,
  kImageTensorDataTypeInt32,
  kImageTensorDataTypeInt16,
  kImageTensorDataTypeInt8,
  kImageTensorDataTypeCount
};

enum ImageTensorChannelType {
  kImageTensorChannelTypeRGB8,
  kImageTensorChannelTypeBGR8,
  kImageTensorChannelTypeGRAY8,
  ImageTensorChannelType_COUNT
};

struct ImageTensorDescription {
  ImageTensorDataType dataType;
  ImageTensorChannelType channelType;
  UINT sizes[kImageTensorDimensionCountMax];
};
}  // namespace Windows::AI::MachineLearning::Internal
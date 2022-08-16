// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"

#include "shaper.h"

namespace onnxruntime {
namespace nnapi {

void Shaper::AddShape(const std::string& name, const Shape& shape) {
  shape_map_[name] = shape;
}

Status Shaper::UpdateShape(const std::string& name, const Shape& new_shape) {
  const Shape& old_shape = shape_map_.at(name);
  if (old_shape != new_shape) {
    ORT_RETURN_IF_NOT(Product(old_shape) == 0 || !old_shape.empty(),
                      "The shape should be same size or old shape has size 0 (dynamic shape)");

    shape_map_[name] = new_shape;
  }

  return Status::OK();
}

Status Shaper::UpdateDynamicDimensions() {
  for (auto& shape_op : shape_ops_)
    ORT_RETURN_IF_ERROR(shape_op(*this));

  return Status::OK();
}

void Shaper::Clear() {
  shape_map_.clear();
  shape_ops_.clear();
}

}  // namespace nnapi
}  // namespace onnxruntime

// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//fhfang@openailab.com

#include <cmath>
#include <vector>

#include "mediapipe/calculators/tengine/retina_anchors_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

float CalculateScale(float min_scale, float max_scale, int stride_index,
                     int num_strides) {
  if (num_strides == 1) {
    return (min_scale + max_scale) * 0.5f;
  } else {
    return min_scale +
           (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
  }
}

}  // namespace

// Generate anchors for Retina object detection model.
// Output:
//   ANCHORS: A list of anchors. Model generates predictions based on the
//   offsets of these anchors.
//
// Usage example:
// node {
//   calculator: "RetinaAnchorsCalculator"
//   output_side_packet: "anchors"
//   options {
//     [mediapipe.RetinaAnchorsCalculatorOptions.ext] {
//       num_layers: 6
//       min_scale: 0.2
//       max_scale: 0.95
//       input_size_height: 300
//       input_size_width: 300
//       anchor_offset_x: 0.5
//       anchor_offset_y: 0.5
//       strides: 16
//       strides: 32
//       strides: 64
//       strides: 128
//       strides: 256
//       strides: 512
//       aspect_ratios: 1.0
//       aspect_ratios: 2.0
//       aspect_ratios: 0.5
//       aspect_ratios: 3.0
//       aspect_ratios: 0.3333
//       reduce_boxes_in_lowest_layer: true
//     }
//   }
// }
class RetinaAnchorsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->OutputSidePackets().Index(0).Set<std::vector<Anchor>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    const RetinaAnchorsCalculatorOptions& options =
        cc->Options<RetinaAnchorsCalculatorOptions>();

    auto anchors = absl::make_unique<std::vector<Anchor>>();
    MP_RETURN_IF_ERROR(GenerateAnchors(anchors.get(), options));
    cc->OutputSidePackets().Index(0).Set(Adopt(anchors.release()));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  static absl::Status GenerateAnchors(
      std::vector<Anchor>* anchors, const RetinaAnchorsCalculatorOptions& options);
};
REGISTER_CALCULATOR(RetinaAnchorsCalculator);

absl::Status RetinaAnchorsCalculator::GenerateAnchors(
    std::vector<Anchor>* anchors, const RetinaAnchorsCalculatorOptions& options) {
  // Verify the options.
  if (!options.feature_map_height_size() && !options.strides_size()) {
    return absl::InvalidArgumentError(
        "Both feature map shape and strides are missing. Must provide either "
        "one.");
  }
  if (options.feature_map_height_size()) {
    if (options.strides_size()) {
      LOG(ERROR) << "Found feature map shapes. Strides will be ignored.";
    }
    CHECK_EQ(options.feature_map_height_size(), options.num_layers());
    CHECK_EQ(options.feature_map_height_size(),
             options.feature_map_width_size());
  } else {
    CHECK_EQ(options.strides_size(), options.num_layers());
  }

  std::vector<float> base_sizes({10.0,16.0,16.0});
  std::vector<std::vector<float>> cur_scales({{8.0f,4.0f,2.0f,1.0f},{8.0f,4.0f,2.0f,1.0f},{32.0f,16.0f,8.0f,4.0f}});
  for(int k=0; k<options.num_layers(); ++k){
    std::vector<std::vector<float>> min_sizes;
    const int stride = options.strides(k);
    int feat_h = std::ceil(1.0f * options.input_size_height() / stride);
    int feat_w = std::ceil(1.0f * options.input_size_width() / stride);

    float s_kx = base_sizes[k];
    float s_ky = base_sizes[k];
    float box_area = s_kx*s_ky;
    std::vector<float> scales = cur_scales[k];
    for(size_t n=0; n<options.aspect_ratios_size(); n++){
      for(size_t m=0; m<scales.size(); m++){
        float size_ratio = box_area / options.aspect_ratios(n);
        float ws = sqrt(size_ratio);
        float hs = ws * options.aspect_ratios(n);
        float scale_ws = ws * scales[m];
        float scale_hs = hs * scales[m];
        std::vector<float> min_size;
        min_size.push_back(scale_ws);
        min_size.push_back(scale_hs);
        min_sizes.push_back(min_size);
      }
    }

    for (int i = 0; i < feat_h; ++i)
    {
      for (int j = 0; j < feat_w; ++j)
      {
        for (size_t l = 0; l < min_sizes.size(); ++l)
        {
          float s_kx = min_sizes[l][0]*1.0/options.input_size_width();
          float s_ky = min_sizes[l][1]*1.0/options.input_size_height();
          float cx = (j + 0.5) / feat_w;
          float cy = (i + 0.5) / feat_h;
          // float s_kx = min_sizes[l][0]*1.0;
          // float s_ky = min_sizes[l][1]*1.0;
          // float cx = (j + 0.5) * stride;
          // float cy = (i + 0.5) * stride;
          Anchor new_anchor;
          new_anchor.set_x_center(cx);
          new_anchor.set_y_center(cy);
          new_anchor.set_w(s_kx);
          new_anchor.set_h(s_ky);
          anchors->push_back(new_anchor);
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace mediapipe

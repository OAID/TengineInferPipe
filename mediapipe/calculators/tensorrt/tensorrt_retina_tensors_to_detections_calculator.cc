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

#include <unordered_map>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/tensorrt/tensorrt_retina_tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/tflite/config.h"


namespace {
constexpr int kNumInputTensorsWithAnchors = 3;
constexpr int kNumCoordsPerBox = 4;

constexpr char kArrayTag[] = "ARRAYS";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
constexpr char kShapeTag[] = "TENSOR_SHAPE";
}  // namespace

namespace mediapipe {
  
class TensorrtTensorsToDetectionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status ProcessCPU(CalculatorContext* cc,
                          std::vector<Detection>* output_detections);

  absl::Status LoadOptions(CalculatorContext* cc);
  Detection ConvertToDetection(float box_ymin, float box_xmin, float box_ymax,
                               float box_xmax, float score, int class_id,
                               bool flip_vertically);

  std::pair<int,float> argmax(const float * begin, int step);
  std::pair<int,uint8_t> argmax(const uint8_t * begin, int step);

  int num_classes_ = 0;
  int num_boxes_ = 0;
  int num_coords_ = 0;
  std::set<int> ignore_classes_;

  ::mediapipe::TensorrtTensorsToDetectionsCalculatorOptions options_;
  std::vector<Anchor> anchors_;
  bool side_packet_anchors_{};

  bool gpu_input_ = false;
  bool anchors_init_ = false;
};
REGISTER_CALCULATOR(TensorrtTensorsToDetectionsCalculator);

absl::Status TensorrtTensorsToDetectionsCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  bool use_gpu = false;

  if (cc->Inputs().HasTag(kArrayTag)) {
      cc->Inputs().Tag(kArrayTag).Set<std::vector<std::vector<float>>>();
  }

  // if (cc->Inputs().HasTag(kTensorsGpuTag)) {
  //   cc->Inputs().Tag(kTensorsGpuTag).Set<std::vector<GpuTensor>>();
  //   use_gpu |= true;
  // }

  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
  }

  if (cc->InputSidePackets().UsesTags()) {
    if (cc->InputSidePackets().HasTag("ANCHORS")) {
      cc->InputSidePackets().Tag("ANCHORS").Set<std::vector<Anchor>>();
    }
  }

  return absl::OkStatus();
}

absl::Status TensorrtTensorsToDetectionsCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  MP_RETURN_IF_ERROR(LoadOptions(cc));
  side_packet_anchors_ = cc->InputSidePackets().HasTag("ANCHORS");

  return absl::OkStatus();
}

absl::Status TensorrtTensorsToDetectionsCalculator::Process(
    CalculatorContext* cc) {
  if ((!gpu_input_ && cc->Inputs().Tag(kArrayTag).IsEmpty())) {
    return absl::OkStatus();
  }

  auto output_detections = absl::make_unique<std::vector<Detection>>();

  if (gpu_input_) {
    absl::UnavailableError("Gpu tensor to detections not available.");
  } else {
    MP_RETURN_IF_ERROR(ProcessCPU(cc, output_detections.get()));
  }

  // Output
  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs()
        .Tag("DETECTIONS")
        .Add(output_detections.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

std::pair<int,float> TensorrtTensorsToDetectionsCalculator::argmax(const float * begin, int step){
    float max_val = *begin;
    int max_index = 0;
    for(int i=0; i<step; i++){
        if(*(begin+i) > max_val){
            max_val = *(begin+i);
            max_index = i;
        }
    }
    return std::make_pair(max_index, max_val);
}

std::pair<int,uint8_t> TensorrtTensorsToDetectionsCalculator::argmax(const uint8_t * begin, int step){
    uint8_t max_val = *begin;
    int max_index = 0;
    for(int i=0; i<step; i++){
        if(*(begin+i) > max_val){
            max_val = *(begin+i);
            max_index = i;
        }
    }
    return std::make_pair(max_index, max_val);
}

absl::Status TensorrtTensorsToDetectionsCalculator::ProcessCPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
  const auto& input_tensors =
      cc->Inputs().Tag(kArrayTag).Get<std::vector<std::vector<float>>>();

  if (input_tensors.size() == 2 ||
      input_tensors.size() == kNumInputTensorsWithAnchors) {
    // Postprocessing on CPU for model without postprocessing op. E.g. output
    // raw score tensor and box tensor. Anchor decoding will be handled below.
    const std::vector<float>* raw_score_tensor = &input_tensors[1];
    const std::vector<float>* raw_box_tensor = &input_tensors[0];
    
    const float* raw_boxes = raw_box_tensor->data();
    const float* raw_scores = raw_score_tensor->data();

    // TODO: Support other options to load anchors.
    if (!anchors_init_) {
      if (side_packet_anchors_) {
        CHECK(!cc->InputSidePackets().Tag("ANCHORS").IsEmpty());
        anchors_ =
            cc->InputSidePackets().Tag("ANCHORS").Get<std::vector<Anchor>>();
      } else {
        return absl::UnavailableError("No anchor data available.");
      }
      anchors_init_ = true;
    }
    
    int anchor_num = anchors_.size();
    for (int i = 0; i < anchor_num; ++i) {
      std::pair<int, float> max_result = argmax(raw_scores+1, num_classes_-1);
      float score_val = max_result.second;
      if (options_.sigmoid_score()) 
        score_val = 1/(1+exp(0-score_val));
      if (options_.has_min_score_thresh() &&
        score_val > options_.min_score_thresh()) {
        float anchor_cx = anchors_[i].x_center();
        float anchor_cy = anchors_[i].y_center();
        float anchor_w = anchors_[i].w();
        float anchor_h = anchors_[i].h();

        float tmp_cx = anchor_cx+*raw_boxes *0.1*anchor_w;
        float tmp_cy = anchor_cy+*(raw_boxes+1) *0.1*anchor_h;
        float tmp_w = anchor_w*exp(*(raw_boxes+2)*0.2);
        float tmp_h = anchor_h*exp(*(raw_boxes+3)*0.2);

        float x1 = tmp_cx - tmp_w/2;
        float y1 = tmp_cy - tmp_h/2;
        float x2 = tmp_cx + tmp_w/2;
        float y2 = tmp_cy + tmp_h/2;
        x1 = x1>0 ? x1 : 0;
        y1 = y1>0 ? y1 : 0;
        if(x1>=0 && x1<1 && x2>=0 && x2<1 && y1>=0 && y1<1 && y2>=0 && y2<1){
          Detection detection = ConvertToDetection(y1, x1, y2, x2,
            score_val, max_result.first, options_.flip_vertically());
          output_detections->emplace_back(detection);
        }
      }
      raw_scores += num_classes_;
      raw_boxes += 4;
    }
    
  }
  return absl::OkStatus();
}


absl::Status TensorrtTensorsToDetectionsCalculator::Close(CalculatorContext* cc) {

  return absl::OkStatus();
}

absl::Status TensorrtTensorsToDetectionsCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ =
      cc->Options<::mediapipe::TensorrtTensorsToDetectionsCalculatorOptions>();

  num_classes_ = options_.num_classes();
  num_boxes_ = options_.num_boxes();
  num_coords_ = options_.num_coords();

  // Currently only support 2D when num_values_per_keypoint equals to 2.
  CHECK_EQ(options_.num_values_per_keypoint(), 2);

  // Check if the output size is equal to the requested boxes and keypoints.
  CHECK_EQ(options_.num_keypoints() * options_.num_values_per_keypoint() +
               kNumCoordsPerBox,
           num_coords_);

  for (int i = 0; i < options_.ignore_classes_size(); ++i) {
    ignore_classes_.insert(options_.ignore_classes(i));
  }

  return absl::OkStatus();
}

Detection TensorrtTensorsToDetectionsCalculator::ConvertToDetection(
    float box_ymin, float box_xmin, float box_ymax, float box_xmax, float score,
    int class_id, bool flip_vertically) {
  Detection detection;
  detection.add_score(score);
  detection.add_label_id(class_id);

  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();
  relative_bbox->set_xmin(box_xmin);
  relative_bbox->set_ymin(flip_vertically ? 1.f - box_ymax : box_ymin);
  relative_bbox->set_width(box_xmax - box_xmin);
  relative_bbox->set_height(box_ymax - box_ymin);
  return detection;
}

}  // namespace mediapipe

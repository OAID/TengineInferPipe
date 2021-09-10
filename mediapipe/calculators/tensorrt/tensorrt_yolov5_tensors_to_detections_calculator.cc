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
#include "mediapipe/calculators/tensorrt/tensorrt_yolov5_tensors_to_detections_calculator.pb.h"
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
constexpr char kPaddingTag[] = "PADDING";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
constexpr char kShapeTag[] = "TENSOR_SHAPE";
}  // namespace

namespace mediapipe {

void ConvertAnchorsToRawValues(const std::vector<Anchor>& anchors,
                               int num_boxes, float* raw_anchors) {
  CHECK_EQ(anchors.size(), num_boxes);
  int box = 0;
  for (const auto& anchor : anchors) {
    raw_anchors[box * kNumCoordsPerBox + 0] = anchor.y_center();
    raw_anchors[box * kNumCoordsPerBox + 1] = anchor.x_center();
    raw_anchors[box * kNumCoordsPerBox + 2] = anchor.h();
    raw_anchors[box * kNumCoordsPerBox + 3] = anchor.w();
    ++box;
  }
}

class TensorrtYolov5TensorsToDetectionsCalculator : public CalculatorBase {
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

  int num_classes_ = 0;
  int img_width_ = 0;
  int img_height_ = 0;
  std::set<int> ignore_classes_;

  ::mediapipe::TensorrtYolov5TensorsToDetectionsCalculatorOptions options_;

  bool gpu_input_ = false;
  bool anchors_init_ = false;
};
REGISTER_CALCULATOR(TensorrtYolov5TensorsToDetectionsCalculator);

absl::Status TensorrtYolov5TensorsToDetectionsCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  bool use_gpu = false;

  if (cc->Inputs().HasTag(kArrayTag)) {
    cc->Inputs().Tag(kArrayTag).Set<std::vector<std::vector<float>>>();
  }

  if (cc->Inputs().HasTag(kPaddingTag)) {
    cc->Inputs().Tag(kPaddingTag).Set<std::array<float, 4>>();
  }

  if (cc->Inputs().HasTag(kTensorsGpuTag)) {
    cc->Inputs().Tag(kTensorsGpuTag).Set<std::vector<GpuTensor>>();
    use_gpu |= true;
  }

  if (cc->Inputs().HasTag(kShapeTag)) {
    cc->Inputs().Tag(kShapeTag).Set<std::vector<std::vector<int>>>();
  }

  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
  }
  return absl::OkStatus();
}

absl::Status TensorrtYolov5TensorsToDetectionsCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  MP_RETURN_IF_ERROR(LoadOptions(cc));

  img_width_ = options_.img_width();
  img_height_ = options_.img_height();

  return absl::OkStatus();
}

absl::Status TensorrtYolov5TensorsToDetectionsCalculator::Process(
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

std::pair<int,float> TensorrtYolov5TensorsToDetectionsCalculator::argmax(const float * begin, int step){
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


absl::Status TensorrtYolov5TensorsToDetectionsCalculator::ProcessCPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
  const auto& input_tensors =
      cc->Inputs().Tag(kArrayTag).Get<std::vector<std::vector<float>>>();
  
  const auto& input_pads = cc->Inputs().Tag(kPaddingTag).Get<std::array<float, 4>>();
  float ori_width = 1.f - input_pads[2] - input_pads[0];
  float ori_height = 1.f - input_pads[3] - input_pads[1];
  int raw_stride = num_classes_ + 5;
  if (input_tensors.size() == 1) {
    // Postprocessing on CPU for model without postprocessing op. E.g. output
    // raw score tensor and box tensor. Anchor decoding will be handled below.
    const std::vector<float>* raw_tensor = &input_tensors[0];
    int num_boxes = raw_tensor->size()/raw_stride;
    
    const float* raw_data = raw_tensor->data();

    // TODO: Support other options to load anchors.
    for (int i = 0; i < num_boxes; ++i) {
      if(*(raw_data + 4) > options_.min_score_thresh()){
        std::pair<int, float> max_result = argmax(raw_data+5, num_classes_);
        float score_val = *(raw_data + 4) * max_result.second;
        if (score_val > options_.min_score_thresh()) {
          float x1 = (*raw_data*1.0 / img_width_ - input_pads[0])/ori_width;
          float y1 = (*(raw_data+1)*1.0 / img_height_ - input_pads[1])/ori_height;
          float x2 = (*(raw_data+2)*1.0 / img_width_ - input_pads[0])/ori_width;
          float y2 = (*(raw_data+3)*1.0 / img_height_ - input_pads[1])/ori_height;
          if(x1>=0 && x1<1 && x2>=0 && x2<1 && y1>=0 && y1<1 && y2>=0 && y2<1){
            Detection detection = ConvertToDetection(y1, x1, y2, x2,
            score_val, max_result.first, options_.flip_vertically());
            output_detections->emplace_back(detection);
          }
        }
      }
      
      raw_data += raw_stride;
    }
    // LOG(INFO)<<"output_detections:"<<output_detections->size();
  }
  return absl::OkStatus();
}


absl::Status TensorrtYolov5TensorsToDetectionsCalculator::Close(CalculatorContext* cc) {

  return absl::OkStatus();
}

absl::Status TensorrtYolov5TensorsToDetectionsCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ =
      cc->Options<::mediapipe::TensorrtYolov5TensorsToDetectionsCalculatorOptions>();

  num_classes_ = options_.num_classes();

  return absl::OkStatus();
}

Detection TensorrtYolov5TensorsToDetectionsCalculator::ConvertToDetection(
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

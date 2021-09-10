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
#include "mediapipe/calculators/tengine/tengine_yolov5_tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace {
constexpr int kNumInputTensorsWithAnchors = 3;
constexpr int kNumCoordsPerBox = 4;

constexpr char kArrayTag[] = "ARRAYS";
constexpr char kPaddingTag[] = "PADDING";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
constexpr char kShapeTag[] = "TENSOR_SHAPE";
constexpr char kPARAMTag[] = "QUANT_PARAM";
}  // namespace

namespace mediapipe {

class TengineYolov5TensorsToDetectionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status ProcessCPU(CalculatorContext* cc,
                          std::vector<Detection>* output_detections);
  absl::Status ProcessUintCPU(CalculatorContext* cc,
                          std::vector<Detection>* output_detections);                    

  absl::Status LoadOptions(CalculatorContext* cc);
  Detection ConvertToDetection(float box_ymin, float box_xmin, float box_ymax,
                               float box_xmax, float score, int class_id,
                               bool flip_vertically);

  std::pair<int,uint8_t> argmax(const uint8_t * begin, int step);
  std::pair<int,float> argmax(const float * begin, int step);
  float sigmoid(float x);

  int num_classes_ = 0;
  int img_width_ = 0;
  int img_height_ = 0;
  std::set<int> ignore_classes_;
  float detect_threshold_ = 0.5;
  std::vector<std::vector<std::vector<int>>> m_anchors{{{10, 13}, {16, 30}, {33, 23}},
        {{30, 61}, {62, 45}, {59, 119}},
        {{116, 90}, {156, 198}, {373, 326}}};

  ::mediapipe::TengineYolov5TensorsToDetectionsCalculatorOptions options_;
  std::vector<Anchor> anchors_;

  bool gpu_input_ = false;
  bool anchors_init_ = false;

  std::string data_type_;
};
REGISTER_CALCULATOR(TengineYolov5TensorsToDetectionsCalculator);

absl::Status TengineYolov5TensorsToDetectionsCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  bool use_gpu = false;
  auto options =
      cc->Options<::mediapipe::TengineYolov5TensorsToDetectionsCalculatorOptions>();
  if (cc->Inputs().HasTag(kArrayTag)) {
    if(options.data_type()=="uint8"){
      cc->Inputs().Tag(kArrayTag).Set<std::vector<std::vector<uint8_t>>>();
    }else{
      cc->Inputs().Tag(kArrayTag).Set<std::vector<std::vector<float>>>();
    }
  }

  if (cc->Inputs().HasTag(kPaddingTag)) {
    cc->Inputs().Tag(kPaddingTag).Set<std::array<float, 4>>();
  }

  if (cc->Inputs().HasTag(kShapeTag)) {
    cc->Inputs().Tag(kShapeTag).Set<std::vector<std::vector<int>>>();
  }

  if (cc->Inputs().HasTag(kPARAMTag)) {
    cc->Inputs().Tag(kPARAMTag).Set<std::vector<std::array<float,2>>>();
  }

  // if (cc->Inputs().HasTag(kTensorsGpuTag)) {
  //   cc->Inputs().Tag(kTensorsGpuTag).Set<std::vector<GpuTensor>>();
  //   use_gpu |= true;
  // }

  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
  }

  return absl::OkStatus();
}

absl::Status TengineYolov5TensorsToDetectionsCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  MP_RETURN_IF_ERROR(LoadOptions(cc));

  img_width_ = options_.img_width();
  img_height_ = options_.img_height();

  LOG(INFO)<<"TengineYolov5TensorsToDetectionsCalculator open";

  return absl::OkStatus();
}

absl::Status TengineYolov5TensorsToDetectionsCalculator::Process(
    CalculatorContext* cc) {
  if ((!gpu_input_ && cc->Inputs().Tag(kArrayTag).IsEmpty())) {
    return absl::OkStatus();
  }

  auto output_detections = absl::make_unique<std::vector<Detection>>();
  if (gpu_input_) {
    absl::UnavailableError("Gpu tensor to detections not available.");
  } else {
    if(data_type_=="uint8") {
      MP_RETURN_IF_ERROR(ProcessUintCPU(cc, output_detections.get()));
    }else{
      MP_RETURN_IF_ERROR(ProcessCPU(cc, output_detections.get()));
    }
  }

  // Output
  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs()
        .Tag("DETECTIONS")
        .Add(output_detections.release(), cc->InputTimestamp());
  }
  return absl::OkStatus();
}

float TengineYolov5TensorsToDetectionsCalculator::sigmoid(float x)
{
    return 1.0 / (1 + expf(-x));
}

std::pair<int,uint8_t> TengineYolov5TensorsToDetectionsCalculator::argmax(const uint8_t * begin, int step){
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

std::pair<int,float> TengineYolov5TensorsToDetectionsCalculator::argmax(const float * begin, int step){
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

absl::Status TengineYolov5TensorsToDetectionsCalculator::ProcessCPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
  const auto& input_tensors =
      cc->Inputs().Tag(kArrayTag).Get<std::vector<std::vector<float>>>();

  const auto& input_pads = cc->Inputs().Tag(kPaddingTag).Get<std::array<float, 4>>();
  float ori_width = 1.f - input_pads[2] - input_pads[0];
  float ori_height = 1.f - input_pads[3] - input_pads[1];

  const auto& input_shapes =
      cc->Inputs().Tag(kShapeTag).Get<std::vector<std::vector<int>>>();

  for(size_t tidx=0; tidx < input_shapes.size(); tidx++) {
    int n_out = static_cast<int>(input_shapes[tidx][4]);
    int class_num = n_out - 5;
    
    std::vector<float> cur_tensor = input_tensors[tidx];
    int anchor_num = static_cast<int>(input_shapes[tidx][1]);
    int feat_h = static_cast<int>(input_shapes[tidx][2]);
    int feat_w = static_cast<int>(input_shapes[tidx][3]);
    int area = feat_h * feat_w;

    float* raw_data = cur_tensor.data();
    for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++){
      int feature_size = feat_h*feat_w*n_out;
      float *ptr = raw_data + anchor_idx*feature_size;
      for(int i=0; i<area; i++){
        float det_score = ptr[4];
        if(det_score > detect_threshold_){
          std::pair<int, float> max_result = argmax(ptr+5, class_num);
          float class_score = max_result.second;
          float score = sigmoid(det_score)*sigmoid(class_score);

          if(score > options_.min_score_thresh()){
            int class_id = max_result.first;
            float centerX = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w;
            float centerY = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h;
            float width   = pow((sigmoid(ptr[2]) * 2), 2) * m_anchors[tidx][anchor_idx][0] / img_width_; //w
            float height  = pow((sigmoid(ptr[3]) * 2), 2) * m_anchors[tidx][anchor_idx][1] / img_height_;
            float x1 = (centerX-width/2-input_pads[0])/ori_width;
            float y1 = (centerY-height/2 - input_pads[1])/ori_height;
            float x2 = (centerX+width/2 - input_pads[0])/ori_width;
            float y2 = (centerY+height/2 - input_pads[1])/ori_height;
            if(x1>=0 && x1<1 && x2>=0 && x2<1 && y1>=0 && y1<1 && y2>=0 && y2<1){
              Detection detection = ConvertToDetection(y1, x1, y2, x2,
              score, max_result.first, options_.flip_vertically());
              output_detections->emplace_back(detection);
            }
          }
        }
        ptr += n_out;
      }
    }
  }
  return absl::OkStatus();
}

absl::Status TengineYolov5TensorsToDetectionsCalculator::ProcessUintCPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
  const auto& input_tensors =
      cc->Inputs().Tag(kArrayTag).Get<std::vector<std::vector<uint8_t>>>();

  const auto& input_pads = cc->Inputs().Tag(kPaddingTag).Get<std::array<float, 4>>();
  float ori_width = 1.f - input_pads[2] - input_pads[0];
  float ori_height = 1.f - input_pads[3] - input_pads[1];

  const auto& output_shapes =
      cc->Inputs().Tag(kShapeTag).Get<std::vector<std::vector<int>>>();

  const auto& quant_params =
      cc->Inputs().Tag(kPARAMTag).Get<std::vector<std::array<float,2>>>();

  for(size_t tidx=0; tidx < output_shapes.size(); tidx++) {
    int n_out = static_cast<int>(output_shapes[tidx][4]);
    int class_num = n_out - 5;
    
    std::vector<uint8_t> cur_tensor = input_tensors[tidx];
    int anchor_num = static_cast<int>(output_shapes[tidx][1]);
    int feat_h = static_cast<int>(output_shapes[tidx][2]);
    int feat_w = static_cast<int>(output_shapes[tidx][3]);
    int area = feat_h * feat_w;
    
    float output_scale = quant_params[tidx][0];
    float output_zero_point = quant_params[tidx][1];
    uint8_t* raw_data = cur_tensor.data();
    for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++){
      int feature_size = feat_h*feat_w*n_out;
      uint8_t *ptr = raw_data + anchor_idx*feature_size;
      for(int i=0; i<area; i++){
        uint8_t det_val = ptr[4];
        float det_score = (det_val*1.0-output_zero_point)*output_scale;
        // LOG(INFO)<<"det_score:"<<det_score<<" "<<det_val<<" "<<output_zero_point<<" "<<output_scale;
        if(det_score > detect_threshold_){
          std::pair<int, uint8_t> max_result = argmax(ptr+5, class_num);
          float class_score = (max_result.second*1.0-output_zero_point)*output_scale;
          float score = sigmoid(det_score)*sigmoid(class_score);
          if(score > options_.min_score_thresh()){
            int class_id = max_result.first;
            float centerX = (sigmoid((ptr[0]*1.0-output_zero_point)*output_scale) * 2 - 0.5 + i % feat_w) / feat_w;
            float centerY = (sigmoid((ptr[1]*1.0-output_zero_point)*output_scale) * 2 - 0.5 + i / feat_w) / feat_h;
            float width   = pow((sigmoid((ptr[2]*1.0-output_zero_point)*output_scale) * 2), 2) * m_anchors[tidx][anchor_idx][0] / img_width_; //w
            float height  = pow((sigmoid((ptr[3]*1.0-output_zero_point)*output_scale) * 2), 2) * m_anchors[tidx][anchor_idx][1] / img_height_;
            float x1 = (centerX-width/2-input_pads[0])/ori_width;
            float y1 = (centerY-height/2 - input_pads[1])/ori_height;
            float x2 = (centerX+width/2 - input_pads[0])/ori_width;
            float y2 = (centerY+height/2 - input_pads[1])/ori_height;
            if(x1>=0 && x1<1 && x2>=0 && x2<1 && y1>=0 && y1<1 && y2>=0 && y2<1){
              Detection detection = ConvertToDetection(y1, x1, y2, x2,
              score, max_result.first, options_.flip_vertically());
              output_detections->emplace_back(detection);
            }
          }
        }
        ptr += n_out;
      }
    }
  }
  std::cout<<"output_detections:"<<output_detections->size()<<std::endl;
  return absl::OkStatus();
}


absl::Status TengineYolov5TensorsToDetectionsCalculator::Close(CalculatorContext* cc) {

  return absl::OkStatus();
}

absl::Status TengineYolov5TensorsToDetectionsCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ =
      cc->Options<::mediapipe::TengineYolov5TensorsToDetectionsCalculatorOptions>();
  data_type_ = options_.data_type();
  num_classes_ = options_.num_classes();

  for (int i = 0; i < options_.ignore_classes_size(); ++i) {
    ignore_classes_.insert(options_.ignore_classes(i));
  }

  if(options_.sigmoid_score()){
    detect_threshold_ = 0-log(1.0/options_.min_score_thresh()-1);
  }else{
    detect_threshold_ = options_.min_score_thresh();
  }

  return absl::OkStatus();
}

Detection TengineYolov5TensorsToDetectionsCalculator::ConvertToDetection(
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

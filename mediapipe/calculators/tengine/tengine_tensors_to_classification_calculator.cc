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
#include "mediapipe/calculators/tengine/tengine_tensors_to_classification_calculator.pb.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/ret_check.h"

namespace {
constexpr int kNumInputTensorsWithAnchors = 3;
constexpr int kNumCoordsPerBox = 4;

constexpr char kArrayTag[] = "ARRAYS";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
constexpr char kShapeTag[] = "TENSOR_SHAPE";
constexpr char kPARAMTag[] = "QUANT_PARAM";
constexpr char kCLASSTag[] = "CLASSIFICATION";
}  // namespace

namespace mediapipe {

class TengineTensorsToClassificationsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status ProcessCPU(CalculatorContext* cc,
                          std::vector<std::pair<int, float>>* output_classifications);
  absl::Status ProcessUintCPU(CalculatorContext* cc,
                          std::vector<std::pair<int, float>>* output_classifications);                    

  absl::Status LoadOptions(CalculatorContext* cc);

  ::mediapipe::TengineTensorsToClassificationsCalculatorOptions options_;

  bool gpu_input_ = false;

  std::string data_type_;
};
REGISTER_CALCULATOR(TengineTensorsToClassificationsCalculator);

absl::Status TengineTensorsToClassificationsCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  bool use_gpu = false;
  auto options = cc->Options<::mediapipe::TengineTensorsToClassificationsCalculatorOptions>();

  if (cc->Inputs().HasTag(kArrayTag)) {
    if(options.data_type()=="uint8"){
      cc->Inputs().Tag(kArrayTag).Set<std::vector<std::vector<uint8_t>>>();
    }else{
      cc->Inputs().Tag(kArrayTag).Set<std::vector<std::vector<float>>>();
    }
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

  if (cc->Outputs().HasTag(kCLASSTag)) {
    cc->Outputs()
        .Tag(kCLASSTag)
        .Set<std::vector<std::pair<int, float>>>();
  }

  return absl::OkStatus();
}

absl::Status TengineTensorsToClassificationsCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  MP_RETURN_IF_ERROR(LoadOptions(cc));
  return absl::OkStatus();
}

absl::Status TengineTensorsToClassificationsCalculator::Process(
    CalculatorContext* cc) {
  if ((!gpu_input_ && cc->Inputs().Tag(kArrayTag).IsEmpty())) {
    return absl::OkStatus();
  }

  auto output_calssifications = absl::make_unique<std::vector<std::pair<int, float>>>();

  if (gpu_input_) {
    absl::UnavailableError("Gpu tensor to classification not available.");
  } else {
    if(data_type_=="uint8") {
      MP_RETURN_IF_ERROR(ProcessUintCPU(cc, output_calssifications.get()));
    }else{
      MP_RETURN_IF_ERROR(ProcessCPU(cc, output_calssifications.get()));
    }
  }

  // Output
  if (cc->Outputs().HasTag(kCLASSTag)) {
    cc->Outputs()
        .Tag(kCLASSTag)
        .Add(output_calssifications.release(), cc->InputTimestamp());
  }
  return absl::OkStatus();
}

absl::Status TengineTensorsToClassificationsCalculator::ProcessCPU(
    CalculatorContext* cc, std::vector<std::pair<int, float>>* output_calssifications) {
  const auto& input_tensors =
      cc->Inputs().Tag(kArrayTag).Get<std::vector<std::vector<float>>>();

  const auto& input_shapes =
      cc->Inputs().Tag(kShapeTag).Get<std::vector<std::vector<int>>>();

  for(size_t tidx=0; tidx < input_shapes.size(); tidx++) {
    int batch_size = input_shapes[tidx][0];
    int class_num = 1;
    for(size_t i=1; i<input_shapes[tidx].size(); i++){
        class_num *= input_shapes[tidx][i];
    }
    
    std::vector<float> cur_tensor = input_tensors[tidx];

    float* ptr = cur_tensor.data();
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++){
        float max_score = 0.f;
        int max_index = 0;
        float total_score = 0.f;
        for(int class_idx=0; class_idx<class_num; class_idx++){
            float cur_score = expf(ptr[class_idx]);
            if(cur_score > max_score){
                max_score = cur_score;
                max_index = class_idx;
            }
            total_score += cur_score;
        }
        std::pair<int, float> result = std::make_pair(max_index, max_score/total_score);
        output_calssifications->emplace_back(result);
        ptr += class_num;
    }
  }
  return absl::OkStatus();
}

absl::Status TengineTensorsToClassificationsCalculator::ProcessUintCPU(
    CalculatorContext* cc, std::vector<std::pair<int,float>>* output_calssifications) {
  const auto& input_tensors =
      cc->Inputs().Tag(kArrayTag).Get<std::vector<std::vector<uint8_t>>>();

  const auto& quant_params =
      cc->Inputs().Tag(kPARAMTag).Get<std::vector<std::array<float,2>>>();
  const auto& input_shapes =
      cc->Inputs().Tag(kShapeTag).Get<std::vector<std::vector<int>>>();

  for(size_t tidx=0; tidx < input_shapes.size(); tidx++) {
    int batch_size = input_shapes[tidx][0];
    int class_num = 1;
    for(size_t i=1; i<input_shapes[tidx].size(); i++){
        class_num *= input_shapes[tidx][i];
    }

    float output_scale = quant_params[tidx][0];
    float output_zero_point = quant_params[tidx][1];
    
    std::vector<uint8_t> cur_tensor = input_tensors[tidx];

    uint8_t* ptr = cur_tensor.data();
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++){
        float max_score = 0.f;
        int max_index = 0;
        float total_score = 0.f;
        for(int class_idx=0; class_idx<class_num; class_idx++){
            float net_out = (ptr[class_idx]*1.0-output_zero_point)*output_scale;
            float cur_score = expf(net_out);
            if(cur_score > max_score){
                max_score = cur_score;
                max_index = class_idx;
            }
            total_score += cur_score;
        }
        std::pair<int, float> result = std::make_pair(max_index, max_score/total_score);
        output_calssifications->emplace_back(result);
        ptr += class_num;
    }
  }
  return absl::OkStatus();
}


absl::Status TengineTensorsToClassificationsCalculator::Close(CalculatorContext* cc) {

  return absl::OkStatus();
}

absl::Status TengineTensorsToClassificationsCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ =
      cc->Options<::mediapipe::TengineTensorsToClassificationsCalculatorOptions>();
  data_type_ = options_.data_type();

  return absl::OkStatus();
}

}  // namespace mediapipe

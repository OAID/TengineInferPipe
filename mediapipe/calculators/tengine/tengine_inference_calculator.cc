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

#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <sys/time.h>

#include "absl/memory/memory.h"
#include "mediapipe/calculators/tengine/tengine_inference_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

#include "c_api.h"

#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
#include "mediapipe/util/cpu_util.h"
#endif  // !__EMSCRIPTEN__ || __EMSCRIPTEN_PTHREADS__


#if defined(MEDIAPIPE_ANDROID)
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/filesystem.h"
#include "mediapipe/util/android/file/base/helpers.h"
#endif  // ANDROID

namespace {
// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size) {  // NOLINT
  return (size + group_size - 1) / group_size;
}

// Round up n to next multiple of m.
size_t RoundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; }  // NOLINT

constexpr char kArrayTag[] = "ARRAYS";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
constexpr char kShapeTag[] = "TENSOR_SHAPE";
constexpr char kPARAMTag[] = "QUANT_PARAM";
}  // namespace

// TengineInferenceCalculator File Layout:
//  * Header
//  * Core
//  * Aux
namespace mediapipe {

namespace {

int GetXnnpackDefaultNumThreads() {
#if defined(MEDIAPIPE_ANDROID) || defined(MEDIAPIPE_IOS) || \
    defined(__EMSCRIPTEN_PTHREADS__)
  constexpr int kMinNumThreadsByDefault = 1;
  constexpr int kMaxNumThreadsByDefault = 4;
  return std::clamp(NumCPUCores() / 2, kMinNumThreadsByDefault,
                    kMaxNumThreadsByDefault);
#else
  return 1;
#endif  // MEDIAPIPE_ANDROID || MEDIAPIPE_IOS || __EMSCRIPTEN_PTHREADS__
}

// Returns number of threads to configure XNNPACK delegate with.
// Returns user provided value if specified. Otherwise, tries to choose optimal
// number of threads depending on the device.
int GetXnnpackNumThreads(
    const mediapipe::TengineInferenceCalculatorOptions& opts) {
  static constexpr int kDefaultNumThreads = -1;
  if (opts.has_delegate() && opts.delegate().has_xnnpack() &&
      opts.delegate().xnnpack().num_threads() != kDefaultNumThreads) {
    return opts.delegate().xnnpack().num_threads();
  }
  return GetXnnpackDefaultNumThreads();
}

}  // namespace

// Calculator Header Section

// Runs inference on the provided input vector<float> tensors and tengine model.
//
// Creates an interpreter with given model and calls invoke().
// Optionally run inference on CPU/GPU.
//
// This calculator is designed to be used with the TengineConverterCalculator,
// to get the appropriate inputs.
//
// When the input tensors are on CPU, gpu inference is optional and can be
// specified in the calculator options.
// When the input tensors are on GPU, inference is GPU and output can be CPU or
// GPU.
//  (i.e. after calling graph.WaitUntilDone()).
//  GPU tensor support rquires OpenGL ES 3.1+.
//  This calculator uses FixedSizeInputStreamHandler by default.
//
class TengineInferenceCalculator : public CalculatorBase {
 public:

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status LoadModel(CalculatorContext* cc);
  absl::Status ProcessInputsCpu(CalculatorContext* cc);
  absl::Status ProcessYolov5InputsCpu(CalculatorContext* cc);
  absl::Status ProcessOutputsCpu(
      CalculatorContext* cc,
      std::unique_ptr<std::vector<std::vector<float>>> output_tensors_cpu);

  absl::Status ProcessOutputsUintCpu(
      CalculatorContext* cc,
      std::unique_ptr<std::vector<std::vector<uint8_t>>> output_tensors_cpu);


  absl::Status RunInContextIfNeeded(std::function<absl::Status(void)> f) {
    return f();
  }

  Packet model_packet_;

  bool gpu_inference_ = false;
  bool gpu_input_ = false;
  bool gpu_output_ = false;
  bool use_quantized_tensors_ = false;

  bool use_advanced_gpu_api_ = false;
  bool allow_precision_loss_ = false;

  bool use_kernel_caching_ = false;
  std::string cached_kernel_filename_;

  graph_t top_graph;
  tensor_t input_tengine_tensor;
  std::vector<tensor_t> output_tensors;
  int input_len;
  int model_input_c_, model_input_w_, model_input_h_;
  std::vector<int> output_lens;

  uint8_t *uinput_data_ = nullptr;
  float *finput_data_ = nullptr;
  std::string data_type_;

  std::vector<std::vector<int>> output_shapes_;
  std::vector<std::array<float,2>> quant_param_;

  std::vector<std::array<float,2>> input_quant_param_;
  int output_num_;
  int max_dim_=4;
  bool use_yolov5_focus_=false;

  bool check_file_exist(const std::string file_name);
};
REGISTER_CALCULATOR(TengineInferenceCalculator);

// Calculator Core Section

namespace {
template <class CC>
bool ShouldUseGpu(CC* cc) {
  return false;
}
}  // namespace

absl::Status TengineInferenceCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kArrayTag) ^
            cc->Inputs().HasTag(kTensorsGpuTag));
  RET_CHECK(cc->Outputs().HasTag(kArrayTag) ^
            cc->Outputs().HasTag(kTensorsGpuTag));

  LOG(INFO)<<"BEGIN TengineInferenceCalculator::GetContract";

  const auto& options =
      cc->Options<::mediapipe::TengineInferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^
            cc->InputSidePackets().HasTag("MODEL"))
      << "Either model as side packet or model path in options is required.";

  if (cc->Inputs().HasTag(kArrayTag))
    cc->Inputs().Tag(kArrayTag).Set<std::vector<std::vector<float>>>();

  if (cc->Outputs().HasTag(kArrayTag)){
    if (options.data_type()=="uint8") {
      cc->Outputs().Tag(kArrayTag).Set<std::vector<std::vector<uint8_t>>>();
    }else if(options.data_type()=="fp32"){
      cc->Outputs().Tag(kArrayTag).Set<std::vector<std::vector<float>>>();
    }else{
      return absl::InternalError("unsupported data_type.");
    }
  }

  if (cc->Outputs().HasTag(kShapeTag)){
    cc->Outputs().Tag(kShapeTag).Set<std::vector<std::vector<int>>>();
  }
  if (cc->Outputs().HasTag(kPARAMTag)){
    cc->Outputs().Tag(kPARAMTag).Set<std::vector<std::array<float,2>>>();
  }


  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return absl::OkStatus();
}

absl::Status TengineInferenceCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  const auto& options =
      cc->Options<::mediapipe::TengineInferenceCalculatorOptions>();

  data_type_ = options.data_type();
  output_num_ = options.output_num();
  max_dim_ = options.max_dim();
  if(options.has_yolov5_focus()){
    use_yolov5_focus_ = options.yolov5_focus();
  }

  gpu_input_ = cc->Inputs().HasTag(kTensorsGpuTag);
  gpu_output_ = cc->Outputs().HasTag(kTensorsGpuTag);

  MP_RETURN_IF_ERROR(LoadModel(cc));

  // MP_RETURN_IF_ERROR(LoadDelegate(cc));
  return absl::OkStatus();
}

absl::Status TengineInferenceCalculator::Process(CalculatorContext* cc) {
  return RunInContextIfNeeded([this, cc]() -> absl::Status {
    struct timeval start;
    gettimeofday(&start,NULL);
    LOG(INFO)<<"TengineInferenceCalculator::Process";
    // 1. Receive pre-processed tensor inputs.
    if (gpu_input_) {
      return absl::InternalError("tengine inference gpu_input_ not supported.");
    } else {
      if(use_yolov5_focus_){
        MP_RETURN_IF_ERROR(ProcessYolov5InputsCpu(cc));
      }else{
        MP_RETURN_IF_ERROR(ProcessInputsCpu(cc));
      }
    }

    // 2. Run inference.
    int ret = run_graph(this->top_graph, 1);
    if (ret != 0) {
      return absl::InternalError("Run top graph failed.");
    }
    // 3. Output processed tensors.
    if(data_type_=="uint8"){
      auto output_tensors_cpu = absl::make_unique<std::vector<std::vector<uint8_t>>>();
      MP_RETURN_IF_ERROR(ProcessOutputsUintCpu(cc, std::move(output_tensors_cpu)));
    }else if(data_type_=="fp32"){
      auto output_tensors_cpu = absl::make_unique<std::vector<std::vector<float>>>();
      MP_RETURN_IF_ERROR(ProcessOutputsCpu(cc, std::move(output_tensors_cpu)));
    }

    struct timeval end;
    gettimeofday(&end,NULL);
    float time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    printf("time_use : %f\n",time_use);
    return absl::OkStatus();
  });
}


absl::Status TengineInferenceCalculator::Close(CalculatorContext* cc) {
  delete [] uinput_data_;
  return RunInContextIfNeeded([this]() -> absl::Status {
    return absl::OkStatus();
  });
}

// Calculator Auxiliary Section

absl::Status TengineInferenceCalculator::ProcessInputsCpu(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kArrayTag).IsEmpty()) {
    return absl::OkStatus();
  }
  // Read CPU input into tensors.
  const auto& input_tensors =
      cc->Inputs().Tag(kArrayTag).Get<std::vector<std::vector<float>>>();
  RET_CHECK_GT(input_tensors.size(), 0);
  for (int i = 0; i < input_tensors.size(); ++i) {
    std::vector<float> input_tensor = input_tensors[i];
    if(data_type_=="uint8") {
      float* input_tensor_buffer = input_tensor.data();
      for(int m=0; m<this->input_len; m++){
        int udata = (round)(input_tensor_buffer[m] / input_quant_param_[i][0] + input_quant_param_[i][1]);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        uinput_data_[m] = udata;
      }
      set_tensor_buffer(input_tengine_tensor, uinput_data_, this->input_len* sizeof(uint8_t));
    } else if(data_type_=="fp32"){
      float* input_tensor_buffer = input_tensor.data();
      set_tensor_buffer(input_tengine_tensor, input_tensor_buffer, this->input_len* sizeof(float));
    }
  }

  return absl::OkStatus();
}

absl::Status TengineInferenceCalculator::ProcessYolov5InputsCpu(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kArrayTag).IsEmpty()) {
    return absl::OkStatus();
  }
  // Read CPU input into tensors.
  const auto& input_tensors =
      cc->Inputs().Tag(kArrayTag).Get<std::vector<std::vector<float>>>();
  RET_CHECK_GT(input_tensors.size(), 0);
  for (int m = 0; m < input_tensors.size(); ++m) {
    std::vector<float> in_tensor = input_tensors[m];
    if (data_type_=="uint8") {
      float* input_tensor_buffer = in_tensor.data();
      /* focus process */
      int letterbox_rows = model_input_h_*2;
      int letterbox_cols = model_input_w_*2;
      for (int i = 0; i < 2; i++) // corresponding to rows
      {
        for (int g = 0; g < 2; g++) // corresponding to cols
        {
          for (int c = 0; c < 3; c++)
          {
            for (int h = 0; h < letterbox_rows / 2; h++)
            {
              for (int w = 0; w < letterbox_cols / 2; w++)
              {
                int in_index = i + g * letterbox_cols + c * letterbox_cols * letterbox_rows + h * 2 * letterbox_cols + w * 2;
                int out_index = i * 2 * 3 * (letterbox_cols / 2) * (letterbox_rows / 2) + g * 3 * (letterbox_cols / 2) * (letterbox_rows / 2) + c * (letterbox_cols / 2) * (letterbox_rows / 2) + h * (letterbox_cols / 2) + w;

                /* quant to uint8 */
                int udata = (round)(input_tensor_buffer[in_index] / input_quant_param_[m][0] + input_quant_param_[m][1]);
                if (udata > 255)
                    udata = 255;
                else if (udata < 0)
                    udata = 0;

                uinput_data_[out_index] = udata;
              }
            }
          }
        }
      }

      set_tensor_buffer(this->input_tengine_tensor, uinput_data_, this->input_len* sizeof(uint8_t));
    }else if(data_type_=="fp32") {
      float* input_tensor_buffer = in_tensor.data();

      /* focus process */
      int letterbox_rows = model_input_h_*2;
      int letterbox_cols = model_input_w_*2;
      for (int i = 0; i < 2; i++) // corresponding to rows
      {
        for (int g = 0; g < 2; g++) // corresponding to cols
        {
          for (int c = 0; c < 3; c++)
          {
            for (int h = 0; h < letterbox_rows / 2; h++)
            {
              for (int w = 0; w < letterbox_cols / 2; w++)
              {
                int in_index = i + g * letterbox_cols + c * letterbox_cols * letterbox_rows + h * 2 * letterbox_cols + w * 2;
                int out_index = i * 2 * 3 * (letterbox_cols / 2) * (letterbox_rows / 2) + g * 3 * (letterbox_cols / 2) * (letterbox_rows / 2) + c * (letterbox_cols / 2) * (letterbox_rows / 2) + h * (letterbox_cols / 2) + w;

                finput_data_[out_index] = input_tensor_buffer[in_index];
              }
            }
          }
        }
      }

      set_tensor_buffer(this->input_tengine_tensor, finput_data_, this->input_len* sizeof(float));      
    }
  }

  return absl::OkStatus();
}


absl::Status TengineInferenceCalculator::ProcessOutputsCpu(
    CalculatorContext* cc,
    std::unique_ptr<std::vector<std::vector<float>>> output_tensors_cpu) {
  // Output result tensors (CPU).

  auto output_tensors_shapes = absl::make_unique<std::vector<std::vector<int>>>();

  auto output_quant_param = absl::make_unique<std::vector<std::array<float,2>>>();

  for(int i=0; i<this->output_tensors.size(); ++i){
    float* fdata = ( float* )get_tensor_buffer(this->output_tensors[i]);
    std::vector<float> tensor;
    tensor.resize(this->output_lens[i]);
    for(int j=0; j<this->output_lens[i]; ++j)
      tensor[j] = fdata[j];
    output_tensors_cpu->emplace_back(tensor);
    output_tensors_shapes->emplace_back(output_shapes_[i]);
    if(data_type_=="uint8"){
      output_quant_param->emplace_back(quant_param_[i]);
    }
  }
  
  cc->Outputs()
      .Tag(kArrayTag)
      .Add(output_tensors_cpu.release(), cc->InputTimestamp());

  cc->Outputs()
      .Tag(kShapeTag)
      .Add(output_tensors_shapes.release(), cc->InputTimestamp());

  if (cc->Outputs().HasTag(kPARAMTag)){
    cc->Outputs()
        .Tag(kPARAMTag)
        .Add(output_quant_param.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

absl::Status TengineInferenceCalculator::ProcessOutputsUintCpu(
    CalculatorContext* cc,
    std::unique_ptr<std::vector<std::vector<uint8_t>>> output_tensors_cpu) {
  // Output result tensors (CPU).

  auto output_tensors_shapes = absl::make_unique<std::vector<std::vector<int>>>();

  auto output_quant_param = absl::make_unique<std::vector<std::array<float,2>>>();

  for(int i=0; i<this->output_tensors.size(); ++i){
    uint8_t* fdata = ( uint8_t* )get_tensor_buffer(this->output_tensors[i]);
    std::vector<uint8_t> tensor;
    tensor.resize(this->output_lens[i]);
    for(int j=0; j<this->output_lens[i]; ++j)
      tensor[j] = fdata[j];
    output_tensors_cpu->emplace_back(tensor);
    output_tensors_shapes->emplace_back(output_shapes_[i]);
    if(data_type_=="uint8"){
      output_quant_param->emplace_back(quant_param_[i]);
    }
  }
  
  cc->Outputs()
      .Tag(kArrayTag)
      .Add(output_tensors_cpu.release(), cc->InputTimestamp());

  cc->Outputs()
      .Tag(kShapeTag)
      .Add(output_tensors_shapes.release(), cc->InputTimestamp());

  if (cc->Outputs().HasTag(kPARAMTag)){
    cc->Outputs()
        .Tag(kPARAMTag)
        .Add(output_quant_param.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}


bool TengineInferenceCalculator::check_file_exist(const std::string file_name)
{
    FILE* fp = fopen(file_name.c_str(), "r");
    if(!fp)
    {
        std::cerr << "Input file not existed: " << file_name << "\n";
        return false;
    }
    fclose(fp);
    return true;
}

absl::Status TengineInferenceCalculator::LoadModel(CalculatorContext* cc) {
  if (use_advanced_gpu_api_) {
    return absl::InternalError("tengine inference use_advanced_gpu_api_ not supported.");
  }

  // init tengine
  if (init_tengine() < 0) {
      return absl::Status(absl::StatusCode::kNotFound,
                      "init tengine error.");
  }

  set_log_level(LOG_WARNING);
  printf("Tengine version: %s\n",get_tengine_version());
  if (request_tengine_version("1.0") < 0) {
    return absl::InternalError("request_tengine_version failed.");
  }

  const auto& options =
      cc->Options<::mediapipe::TengineInferenceCalculatorOptions>();
  if (!options.model_path().empty()) {
    std::string model_path = options.model_path();
    if (!check_file_exist(model_path)) {
      return absl::Status(absl::StatusCode::kNotFound,
                      "Tengine model path not exist.");
    }
    std::string backend = options.tengine_backend();
    if(backend=="cpu"){
      this->top_graph = create_graph(NULL, "tengine", model_path.c_str());
    }else if(backend=="timvx"){
      context_t timvx_context = create_context("timvx", 1);
      int rtt = add_context_device(timvx_context, "TIMVX");
      if (0 > rtt)
      {
          return absl::Status(absl::StatusCode::kNotFound,
                      "add_context_device VSI DEVICE failed.");
      }
      this->top_graph = create_graph(timvx_context, "tengine", model_path.c_str());
      if (this->top_graph == nullptr) {
          return absl::Status(absl::StatusCode::kNotFound,
                      "create_graph TIMVX failed.");
      }
    }else{
      return absl::Status(absl::StatusCode::kNotFound,
                    "unknown tengine backend.");
    }
    
  }else{
    return absl::InvalidArgumentError("Must specify Tengine model path.");
  }

  int ret = prerun_graph(this->top_graph);
  if (ret != 0) {
    return absl::InternalError("Prerun top graph failed.");
  }else{
      printf("Prerun top graph sucess\n");
  }

  this->input_tengine_tensor = get_graph_input_tensor(this->top_graph, 0,0);
  if(data_type_=="uint8"){
    input_quant_param_.resize(1);
  }
  if(data_type_=="uint8"){
    float output_scale;
    int output_zero_point;
    get_tensor_quant_param(this->input_tengine_tensor, &output_scale, &output_zero_point, 1);
    input_quant_param_[0][0] = output_scale;
    input_quant_param_[0][1] = output_zero_point*1.0;
  }

  int in_dims[4] = {0};
  int in_ndims = get_tensor_shape(this->input_tengine_tensor, in_dims, 4);
  this->input_len = 1;
  for(int m=0; m<in_ndims; ++m){
      this->input_len = this->input_len*in_dims[m];
  }


  model_input_c_ = in_dims[1];
  model_input_h_ = in_dims[2];
  model_input_w_ = in_dims[3];
  uinput_data_ = new uint8_t[this->input_len];
  
  this->output_tensors.resize(output_num_);
  this->output_lens.resize(output_num_);
  int* out_dims = new int[max_dim_];
  output_shapes_.resize(output_num_);
  if(data_type_=="uint8"){
    quant_param_.resize(output_num_);
  }
  for (int i = 0; i < output_num_; i++) {
    this->output_tensors[i] = get_graph_output_tensor(this->top_graph, i,0);
    int out_ndims = get_tensor_shape(this->output_tensors[i], out_dims, max_dim_);
    output_shapes_[i].resize(out_ndims);
    int data_len = 1;
    for(int m=0; m<out_ndims; ++m){
      data_len = data_len*out_dims[m];
      output_shapes_[i][m] = out_dims[m];
    }
    if(data_type_=="uint8"){
      float output_scale;
      int output_zero_point;
      get_tensor_quant_param(this->output_tensors[i], &output_scale, &output_zero_point, 1);
      quant_param_[i][0] = output_scale;
      quant_param_[i][1] = output_zero_point*1.0;
    }
    this->output_lens[i] = data_len;
  }

  delete [] out_dims;
  return absl::OkStatus();
}

}  // namespace mediapipe

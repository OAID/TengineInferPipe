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
#include "mediapipe/calculators/tensorrt/tensorrt_inference_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "common/buffers.h"
#include "common/logging.h"
#include <math.h>

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
}  // namespace

// TensorrtInferenceCalculator File Layout:
//  * Header
//  * Core
//  * Aux
namespace mediapipe {

namespace {

Logger gLogger{Logger::Severity::kINFO};
LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

void setReportableSeverity(Logger::Severity severity)
{
    gLogger.setReportableSeverity(severity);
    gLogVerbose.setReportableSeverity(severity);
    gLogInfo.setReportableSeverity(severity);
    gLogWarning.setReportableSeverity(severity);
    gLogError.setReportableSeverity(severity);
    gLogFatal.setReportableSeverity(severity);
}


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
    const mediapipe::TensorrtInferenceCalculatorOptions& opts) {
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
class TensorrtInferenceCalculator : public CalculatorBase {
  template <typename T>
    using TrtUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
 public:

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status LoadModel(CalculatorContext* cc);
  absl::Status ProcessInputsCpu(CalculatorContext* cc,
                                std::vector<std::vector<float>>* output_tensors_cpu);
  absl::Status ProcessOutputsCpu(
      CalculatorContext* cc,
      std::unique_ptr<std::vector<std::vector<float>>> output_tensors_cpu);

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

  TrtUniquePtr<nvinfer1::IExecutionContext> mContext;
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

  nvinfer1::Dims mInputDims;
  std::vector<nvinfer1::Dims> mOutputDims;

  float m_detThreshold = 0.5;
  float m_nmsThreshold = 0.4;

  int m_batchSize = 1;
  int mInputLen = 1;
  std::vector<int> mOutputLens;
  std::vector<std::vector<int>> output_shapes_;
  int output_num_;
  samplesCommon::BufferManager mBuffers;
  std::string m_input_name;
  std::vector<std::string> m_output_names;

  bool check_file_exist(const std::string file_name);
  bool load_engine(const std::string& engine_path);
  bool constructNetwork(const std::string& onnx_path, const std::string& engine_path, bool use_fp16);
};
REGISTER_CALCULATOR(TensorrtInferenceCalculator);

// Calculator Core Section

namespace {
template <class CC>
bool ShouldUseGpu(CC* cc) {
  return false;
}
}  // namespace

absl::Status TensorrtInferenceCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kArrayTag) ^
            cc->Inputs().HasTag(kTensorsGpuTag));
  RET_CHECK(cc->Outputs().HasTag(kArrayTag) ^
            cc->Outputs().HasTag(kTensorsGpuTag));

  const auto& options =
      cc->Options<::mediapipe::TensorrtInferenceCalculatorOptions>();
  RET_CHECK(!options.onnx_path().empty() ^
            cc->InputSidePackets().HasTag("MODEL"))
      << "Either model as side packet or model path in options is required.";

  if (cc->Inputs().HasTag(kArrayTag))
    cc->Inputs().Tag(kArrayTag).Set<std::vector<std::vector<float>>>();
  if (cc->Outputs().HasTag(kArrayTag))
    cc->Outputs().Tag(kArrayTag).Set<std::vector<std::vector<float>>>();

  if (cc->Outputs().HasTag(kShapeTag)){
    cc->Outputs().Tag(kShapeTag).Set<std::vector<std::vector<int>>>();
  }

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return absl::OkStatus();
}

absl::Status TensorrtInferenceCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  const auto& options =
      cc->Options<::mediapipe::TensorrtInferenceCalculatorOptions>();

  gpu_input_ = cc->Inputs().HasTag(kTensorsGpuTag);
  gpu_output_ = cc->Outputs().HasTag(kTensorsGpuTag);
  output_num_ = options.output_num();

  MP_RETURN_IF_ERROR(LoadModel(cc));

  // MP_RETURN_IF_ERROR(LoadDelegate(cc));
  return absl::OkStatus();
}

absl::Status TensorrtInferenceCalculator::Process(CalculatorContext* cc) {
  return RunInContextIfNeeded([this, cc]() -> absl::Status {
    // struct timeval start;
    // gettimeofday(&start,NULL);
    // 0. Declare outputs
    auto output_tensors_cpu = absl::make_unique<std::vector<std::vector<float>>>();

    // 1. Receive pre-processed tensor inputs.
    if (gpu_input_) {
      return absl::InternalError("tengine inference gpu_input_ not supported.");
    } else {
      MP_RETURN_IF_ERROR(ProcessInputsCpu(cc, output_tensors_cpu.get()));
    }
    // 2. Run inference.
    bool label = mContext->execute(m_batchSize, mBuffers.getDeviceBindings().data());
    // struct timeval execute;
    // gettimeofday(&execute,NULL);
    // float time_execute_use=(execute.tv_sec-start.tv_sec)*1000000+(execute.tv_usec-start.tv_usec);
    // printf("time_execute_use : %f\n",time_execute_use);
    // Memcpy from device output buffers to host output buffers
    mBuffers.copyOutputToHost();
    

    // if (ret != 0) {
    //   return absl::InternalError("Run top graph failed.");
    // }

    // 3. Output processed tensors.

    MP_RETURN_IF_ERROR(ProcessOutputsCpu(cc, std::move(output_tensors_cpu)));
    // struct timeval end;
    // gettimeofday(&end,NULL);
    // float time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    // printf("time_use : %f\n",time_use);
    return absl::OkStatus();
  });
}


absl::Status TensorrtInferenceCalculator::Close(CalculatorContext* cc) {

  return RunInContextIfNeeded([this]() -> absl::Status {
    return absl::OkStatus();
  });
}

// Calculator Auxiliary Section

absl::Status TensorrtInferenceCalculator::ProcessInputsCpu(
    CalculatorContext* cc, std::vector<std::vector<float>>* output_tensors_cpu) {
  if (cc->Inputs().Tag(kArrayTag).IsEmpty()) {
    return absl::OkStatus();
  }
  // Read CPU input into tensors.
  const auto& input_tensors =
      cc->Inputs().Tag(kArrayTag).Get<std::vector<std::vector<float>>>();
  RET_CHECK_GT(input_tensors.size(), 0);
  for (int i = 0; i < input_tensors.size(); ++i) {
    std::vector<float> input_tensor = input_tensors[i];
    if (use_quantized_tensors_) {
      return absl::InternalError("tensorrt inference quantized tensors not supported.");
    } else {
      float* input_tensor_buffer = input_tensor.data();
      float* hostDataBuffer = (float*)(mBuffers.getHostBuffer(m_input_name));
      memcpy(hostDataBuffer, input_tensor_buffer, mInputLen*sizeof(float));
      mBuffers.copyInputToDevice();
    }
  }

  return absl::OkStatus();
}


absl::Status TensorrtInferenceCalculator::ProcessOutputsCpu(
    CalculatorContext* cc,
    std::unique_ptr<std::vector<std::vector<float>>> output_tensors_cpu) {
  // Output result tensors (CPU).
  
  auto output_tensors_shapes = absl::make_unique<std::vector<std::vector<int>>>();

  for(size_t k=0; k<m_output_names.size(); k++){
    std::string output_name = m_output_names[k];
    int output_len = mOutputLens[k];

    void* predHost = mBuffers.getHostBuffer(m_output_names[k]);
    for(int i=0; i<mOutputDims[k].d[0]; ++i){
      float* fdata = ( float* )predHost;
      std::vector<float> tensor(fdata, fdata+output_len);
      output_tensors_cpu->emplace_back(tensor);
      predHost += output_len;
    }
    output_tensors_shapes->emplace_back(output_shapes_[k]);
  }
  
  cc->Outputs()
      .Tag(kArrayTag)
      .Add(output_tensors_cpu.release(), cc->InputTimestamp());

  cc->Outputs()
      .Tag(kShapeTag)
      .Add(output_tensors_shapes.release(), cc->InputTimestamp());

  return absl::OkStatus();
}


bool TensorrtInferenceCalculator::check_file_exist(const std::string file_name)
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

bool TensorrtInferenceCalculator::constructNetwork(const std::string& onnx_path, const std::string& engine_path, bool use_fp16){
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }
    
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    
    auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto parsed = parser->parseFromFile(
        onnx_path.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(100_MiB);
    if (use_fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), -1);

    std::shared_ptr<nvinfer1::ICudaEngine> tmpEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!tmpEngine)
    {
        return false;
    }
    
    std::ofstream engineFile(engine_path, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Cannot open engine file: " << engine_path << std::endl;
        return false;
    }

    TrtUniquePtr<IHostMemory> serializedEngine{tmpEngine->serialize()};
    if (serializedEngine == nullptr)
    {
        std::cout << "Engine serialization failed" << std::endl;
        return false;
    }
    tmpEngine->destroy();
    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    return !engineFile.fail();
}

bool TensorrtInferenceCalculator::load_engine(const std::string& engine_path){
    TrtUniquePtr<IRuntime> tmpRuntime{createInferRuntime(gLogger.getTRTLogger())};
        
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Error opening engine file: " << std::endl;
        return false;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        std::cout << "Error loading engine file: " << std::endl;
        return false;
    }
    engineFile.close();
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(tmpRuntime->deserializeCudaEngine(engineData.data(), fsize, nullptr),samplesCommon::InferDeleter());
    // mEngine.reset(tmpRuntime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    
    mBuffers.init_buffer(mEngine, 1);
    const int inputIndex = 0;
    mInputDims =  mEngine->getBindingDimensions(inputIndex);
    m_input_name = mEngine->getBindingName(inputIndex);
    // assert(mInputDims.nbDims == 4);
    mInputLen = mInputDims.d[1]*mInputDims.d[2]*mInputDims.d[3];

    mOutputDims.resize(output_num_);
    mOutputLens.resize(output_num_);
    m_output_names.resize(output_num_);
    output_shapes_.resize(output_num_);

    for(int i=0; i<output_num_; i++){
      int outputIndex = i+1;
      nvinfer1::Dims output_dims =  mEngine->getBindingDimensions(outputIndex);
      int output_len = 1;
      std::vector<int> output_shape(output_dims.nbDims);
      output_shape[0] = output_dims.d[0];
      for(int m=1; m<output_dims.nbDims; m++){
        output_len *= output_dims.d[m];
        output_shape[m] = output_dims.d[m];
      }
      mOutputDims[i] = output_dims;

      mOutputLens[i] = output_len;
      output_shapes_[i] = output_shape;

      std::string output_name = mEngine->getBindingName(outputIndex);
      m_output_names[i] = output_name;
    }
    
    mContext = TrtUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext)
    {
        return false;
    }
    
    return true;
}

absl::Status TensorrtInferenceCalculator::LoadModel(CalculatorContext* cc) {
  if (use_advanced_gpu_api_) {
    return absl::InternalError("tengine inference use_advanced_gpu_api_ not supported.");
  }

  // init tensorrt
  const auto& options =
      cc->Options<::mediapipe::TensorrtInferenceCalculatorOptions>();
  if (!options.onnx_path().empty()  && !options.engine_path().empty()) {
    std::string onnx_path = options.onnx_path();
    if (!check_file_exist(onnx_path)) {
      return absl::Status(absl::StatusCode::kNotFound,
                      "Onnx model path not exist.");
    }

    std::string engine_path = options.engine_path();
    if (!check_file_exist(engine_path)) {
      constructNetwork(onnx_path, engine_path, options.use_fp16());
    }
  }else{
    return absl::InvalidArgumentError("Must specify onnx and engine path.");
  }

  // std::vector<int> h_vecs{};
  m_detThreshold = options.detect_threshold();
  m_nmsThreshold = options.nms_threshold();

  m_batchSize = options.batch_size();
  bool ret = load_engine(options.engine_path());
  if (!ret) {
    return absl::InternalError("load_engine failed.");
  }else{
    printf("load_engine sucess\n");
  }

  return absl::OkStatus();
}

}  // namespace mediapipe

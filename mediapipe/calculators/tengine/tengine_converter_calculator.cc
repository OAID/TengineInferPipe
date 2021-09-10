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

#include <string>
#include <vector>

#include <sys/time.h>

#include "mediapipe/calculators/tengine/tengine_converter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace {
constexpr int kWorkgroupSize = 8;  // Block size for GPU shader.
// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size) {  // NOLINT
  return (size + group_size - 1) / group_size;
}

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
    ColMajorMatrixXf;

constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kGpuBufferTag[] = "IMAGE_GPU";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kArrayTag[] = "ARRAYS";
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
constexpr char kMatrixTag[] = "MATRIX";
}  // namespace

namespace mediapipe {

class TengineConverterCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status InitGpu(CalculatorContext* cc);
  absl::Status LoadOptions(CalculatorContext* cc);
  template <class T>
  absl::Status NormalizeImage(const ImageFrame& image_frame,
                              bool flip_vertically, std::vector<float>& tensor_ptr);
  absl::Status CopyMatrixToTensor(const Matrix& matrix, float* tensor_ptr);
  absl::Status ProcessCPU(CalculatorContext* cc);
  absl::Status ProcessGPU(CalculatorContext* cc);

  bool initialized_ = false;
  bool use_gpu_ = false;
  bool flip_vertically_ = false;
  bool row_major_matrix_ = false;
  bool use_quantized_tensors_ = false;
  int max_num_channels_ = 3;

  std::vector<float> mean_={0.f,0.f,0.f};
  std::vector<float> scale_={1.f,1.f,1.f};
};
REGISTER_CALCULATOR(TengineConverterCalculator);

namespace {
template <class CC>
bool ShouldUseGpu(CC* cc) {
  return false;
}
}  // namespace

absl::Status TengineConverterCalculator::GetContract(CalculatorContract* cc) {
  // Confirm only one of the input streams is present.
  RET_CHECK(cc->Inputs().HasTag(kImageFrameTag) ^
            cc->Inputs().HasTag(kGpuBufferTag) ^
            cc->Inputs().HasTag(kMatrixTag));

  // Confirm only one of the output streams is present.
  RET_CHECK(cc->Outputs().HasTag(kArrayTag) ^
            cc->Outputs().HasTag(kTensorsGpuTag));

  if (cc->Inputs().HasTag(kImageFrameTag)) {
    cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
  }
  if (cc->Inputs().HasTag(kMatrixTag)) {
    cc->Inputs().Tag(kMatrixTag).Set<Matrix>();
  }
#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kGpuBufferTag)) {
    cc->Inputs().Tag(kGpuBufferTag).Set<mediapipe::GpuBuffer>();
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  if (cc->Outputs().HasTag(kArrayTag)) {
    cc->Outputs().Tag(kArrayTag).Set<std::vector<std::vector<float>>>();
  }

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return absl::OkStatus();
}

absl::Status TengineConverterCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  MP_RETURN_IF_ERROR(LoadOptions(cc));

  use_gpu_ = ShouldUseGpu(cc);

  if (use_gpu_) {
    // Cannot mix CPU/GPU streams.
    RET_CHECK(cc->Inputs().HasTag(kGpuBufferTag) &&
              cc->Outputs().HasTag(kTensorsGpuTag));
    // Cannot use quantization.
    use_quantized_tensors_ = false;
  }

  return absl::OkStatus();
}

absl::Status TengineConverterCalculator::Process(CalculatorContext* cc) {
  // struct timeval start;
  // gettimeofday(&start,NULL);
  // printf("Tengine convert start.tv_sec:%d\n",start.tv_sec);
  // printf("Tengine convert start.tv_usec:%d\n",start.tv_usec);

  if (use_gpu_) {
    if (cc->Inputs().Tag(kGpuBufferTag).IsEmpty()) {
      return absl::OkStatus();
    }
    if (!initialized_) {
      MP_RETURN_IF_ERROR(InitGpu(cc));
      initialized_ = true;
    }
    // Convert to GPU tensors type.
    MP_RETURN_IF_ERROR(ProcessGPU(cc));
  } else {
    // Convert to CPU tensors or Matrix type.
    MP_RETURN_IF_ERROR(ProcessCPU(cc));
  }
  return absl::OkStatus();
}

absl::Status TengineConverterCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status TengineConverterCalculator::ProcessCPU(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kImageFrameTag)) {
    if (cc->Inputs().Tag(kImageFrameTag).IsEmpty()) {
      return absl::OkStatus();
    }

    const auto& image_frame =
        cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
    const int height = image_frame.Height();
    const int width = image_frame.Width();
    const int channels = image_frame.NumberOfChannels();
    const int channels_preserved = std::min(channels, max_num_channels_);
    const mediapipe::ImageFormat::Format format = image_frame.Format();

    if (!initialized_) {
      if (!(format == mediapipe::ImageFormat::SRGBA ||
            format == mediapipe::ImageFormat::SRGB ||
            format == mediapipe::ImageFormat::GRAY8 ||
            format == mediapipe::ImageFormat::VEC32F1))
        RET_CHECK_FAIL() << "Unsupported CPU input format.";

      initialized_ = true;
    }

    // Copy image data into tensor.
    auto output_tensors = absl::make_unique<std::vector<std::vector<float>>>();
    if (use_quantized_tensors_) {
      return absl::InternalError("tengine converter quantized tensors not supported.");
    } else {
      std::vector<float> tensor;
      tensor.resize(height*width*channels_preserved);
      if (image_frame.ByteDepth() == 1) {
        MP_RETURN_IF_ERROR(NormalizeImage<uint8>(image_frame, flip_vertically_,
                                                 tensor));
      } else if (image_frame.ByteDepth() == 4) {
        MP_RETURN_IF_ERROR(NormalizeImage<float>(image_frame, flip_vertically_,
                                                 tensor));
      } else {
        return absl::InternalError(
            "Only byte-based (8 bit) and float (32 bit) images supported.");
      }
      output_tensors->emplace_back(tensor);
    }

    cc->Outputs()
        .Tag(kArrayTag)
        .Add(output_tensors.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

absl::Status TengineConverterCalculator::ProcessGPU(CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status TengineConverterCalculator::InitGpu(CalculatorContext* cc) {

  return absl::OkStatus();
}

absl::Status TengineConverterCalculator::LoadOptions(CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  const auto& options =
      cc->Options<::mediapipe::TengineConverterCalculatorOptions>();

  // Get y-flip mode.
  flip_vertically_ = options.flip_vertically();

  // Get row_major_matrix mode.
  row_major_matrix_ = options.row_major_matrix();

  // Get desired way to handle input channels.
  max_num_channels_ = options.max_num_channels();
  CHECK_GE(max_num_channels_, 1);
  CHECK_LE(max_num_channels_, 4);
  CHECK_NE(max_num_channels_, 2);
#if defined(MEDIAPIPE_IOS)
  if (cc->Inputs().HasTag(kGpuBufferTag))
    max_num_channels_ = 4;
#endif

  // Get tensor type, float or quantized.
  use_quantized_tensors_ = options.use_quantized_tensors();

  if (options.has_tensor_mean()) {
    mean_[0] = options.tensor_mean().val1();
    mean_[1] = options.tensor_mean().val2();
    mean_[2] = options.tensor_mean().val3();
  }

  if (options.has_tensor_scale()) {
    scale_[0] = options.tensor_scale().val1();
    scale_[1] = options.tensor_scale().val2();
    scale_[2] = options.tensor_scale().val3();
  }

  return absl::OkStatus();
}

template <class T>
absl::Status TengineConverterCalculator::NormalizeImage(
    const ImageFrame& image_frame, bool flip_vertically, std::vector<float>& tensor_ptr) {
  const int height = image_frame.Height();
  const int width = image_frame.Width();
  const int channels = image_frame.NumberOfChannels();
  const int channels_preserved = std::min(channels, max_num_channels_);
  const int channels_ignored = channels - channels_preserved;

  const float scale = 1.0f / 255.0f;
  for (int i = 0; i < height; ++i) {
    const T* image_ptr = reinterpret_cast<const T*>(
        image_frame.PixelData() +
        (flip_vertically ? height - 1 - i : i) * image_frame.WidthStep());
    for (int j = 0; j < width; ++j) {
      for (int c = 0; c < channels_preserved; ++c) {
        int dst_index = j+i*width+c*height*width;
        tensor_ptr[dst_index] = (*image_ptr++ - mean_[c])*scale_[c];
      }
      image_ptr += channels_ignored;
    }
  }

  return absl::OkStatus();
}

absl::Status TengineConverterCalculator::CopyMatrixToTensor(const Matrix& matrix,
                                                           float* tensor_ptr) {
  if (row_major_matrix_) {
    auto matrix_map =
        Eigen::Map<RowMajorMatrixXf>(tensor_ptr, matrix.rows(), matrix.cols());
    matrix_map = matrix;
  } else {
    auto matrix_map =
        Eigen::Map<ColMajorMatrixXf>(tensor_ptr, matrix.rows(), matrix.cols());
    matrix_map = matrix;
  }

  return absl::OkStatus();
}

}  // namespace mediapipe

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "absl/status/status.h"

typedef struct abox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
} abox;

typedef struct acategory {
    float score;
    int class_id;
} acategory;

class TensorrtOperation;

class InferTensorrtPipe
{
public:
    InferTensorrtPipe();
    ~InferTensorrtPipe();
    absl::Status init_pipe(std::string calculator_graph_config_file, std::string input_stream, std::string output_stream);

    absl::Status infer(const cv::Mat& frame, std::vector<abox>& result);

    absl::Status infer(const cv::Mat& frame, std::vector<acategory>& result);

    absl::Status release_pipe();

private:
    std::unique_ptr<TensorrtOperation> mOperation;
};


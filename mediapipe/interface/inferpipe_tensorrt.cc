#include <cstdlib>

#include "absl/status/status.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/calculators/tensorrt/inferpipe_tensorrt.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"

class TensorrtOperation{
public:
    TensorrtOperation(){};
    ~TensorrtOperation();
    mediapipe::CalculatorGraph graph;
    // mediapipe::StatusOrPoller status_or_poller;
    std::unique_ptr<mediapipe::OutputStreamPoller> allow_poller;
    std::string input_stream_name;
    std::string output_stream_name;
};
TensorrtOperation::~TensorrtOperation() = default;

InferTensorrtPipe::InferTensorrtPipe(){
    mOperation = std::unique_ptr<TensorrtOperation>(new TensorrtOperation());
}

InferTensorrtPipe::~InferTensorrtPipe() = default;

absl::Status InferTensorrtPipe::init_pipe(std::string calculator_graph_config_file, std::string input_stream, std::string output_stream){
    std::string calculator_graph_config_contents;
    mediapipe::file::GetContents(
        calculator_graph_config_file,
        &calculator_graph_config_contents);
    // LOG(INFO) << "Get calculator graph config contents: "
    //             << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);
    LOG(INFO) << "MP_RETURN_IF_ERROR Initialize";
    MP_RETURN_IF_ERROR(mOperation->graph.Initialize(config));
    mOperation->input_stream_name = input_stream;
    mOperation->output_stream_name = output_stream;
    LOG(INFO) << "Start running the calculator graph.";
    // mOperation->status_or_poller =
    //   mOperation->graph.AddOutputStreamPoller(mOperation->output_stream_name);
    mOperation->allow_poller.reset(
        new mediapipe::OutputStreamPoller(mOperation->graph.AddOutputStreamPoller(mOperation->output_stream_name).value()));
    MP_RETURN_IF_ERROR(mOperation->graph.StartRun({}));
    LOG(INFO) << "MP_RETURN_IF_ERROR StartRun";
    return absl::OkStatus();
}

absl::Status InferTensorrtPipe::infer(const cv::Mat& frame, std::vector<abox>& result){
    cv::Mat camera_frame;
    cv::cvtColor(frame, camera_frame, cv::COLOR_BGR2RGB);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

    MP_RETURN_IF_ERROR(mOperation->graph.AddPacketToInputStream(
        mOperation->input_stream_name, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    // mediapipe::OutputStreamPoller& poller = mOperation->status_or_poller.value();

    mediapipe::Packet packet;
    // if (!poller.Next(&packet)){
    if (!mOperation->allow_poller->Next(&packet)){
        LOG(INFO) << "CancelledError.";
        return absl::CancelledError();
    }
        
    auto output_result = packet.Get<std::vector<mediapipe::Detection>>();
    for(size_t i=0; i<output_result.size(); i++){
        abox cur_loc;
        cur_loc.score = output_result[i].score(0);
        cur_loc.class_id = output_result[i].label_id(0);

        const mediapipe::LocationData::RelativeBoundingBox* relative_bbox =
            output_result[i].mutable_location_data()->mutable_relative_bounding_box();

        cur_loc.x1 = relative_bbox->xmin();
        cur_loc.y1 = relative_bbox->ymin();
        cur_loc.x2 = relative_bbox->xmin()+relative_bbox->width();
        cur_loc.y2 = relative_bbox->ymin()+relative_bbox->height();
        result.push_back(cur_loc);
    }
    
    return absl::OkStatus();
}

absl::Status InferTensorrtPipe::infer(const cv::Mat& frame, std::vector<acategory>& result){
    cv::Mat camera_frame;
    cv::cvtColor(frame, camera_frame, cv::COLOR_BGR2RGB);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

    MP_RETURN_IF_ERROR(mOperation->graph.AddPacketToInputStream(
        mOperation->input_stream_name, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    mediapipe::Packet packet;
    if (!mOperation->allow_poller->Next(&packet)){
        LOG(INFO) << "CancelledError.";
        return absl::CancelledError();
    }
        
    auto output_result = packet.Get<std::vector<std::pair<int, float>>>();
    for(size_t i=0; i<output_result.size(); i++){
        acategory cur_item;
        cur_item.score = output_result[i].second;
        cur_item.class_id = output_result[i].first;

        result.push_back(cur_item);
    }
    
    return absl::OkStatus();
}

absl::Status InferTensorrtPipe::release_pipe(){
    return mOperation->graph.WaitUntilDone();
}
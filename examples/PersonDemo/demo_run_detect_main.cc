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
//
// An example of reading OpenCV video frames into a InferPipe graph
#include <cstdlib>
#include "absl/status/status.h"
#include "inferpipe_tengine.h"

int main(int argc, char** argv) {

  std::vector<cv::Scalar> color_map;
  color_map.push_back(cv::Scalar(0,255,0));
  color_map.push_back(cv::Scalar(0,0,255));
  color_map.push_back(cv::Scalar(255,0,0));

  InferTenginePipe test_pipe;

  test_pipe.init_pipe(argv[1],argv[2],argv[3]);

  cv::VideoCapture capture;
  capture.open(argv[4]);

  int frame_num=0;
  while (true) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    // cv::Mat camera_frame_raw = cv::imread(argv[4]);
    if (camera_frame_raw.empty()) {
      break;
    }
    // camera_frame_raw = cv::imread("../test.jpg");
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    std::vector<abox> result;
    absl::Status run_status = test_pipe.infer(camera_frame, result);
    if (!run_status.ok()) {
      std::cout << "Failed to run the graph: " << run_status.message()<<std::endl;
      return EXIT_FAILURE;
    } 
    std::cout<<"result: "<<result.size()<<std::endl;
    int width = camera_frame.cols;
    int height = camera_frame.rows;
    for(size_t i=0; i<result.size(); i++){
      int x1 = static_cast<int>(floor(result[i].x1*width));
      int y1 = static_cast<int>(floor(result[i].y1*height));
      int x2 = static_cast<int>(floor(result[i].x2*width));
      int y2 = static_cast<int>(floor(result[i].y2*height));
      cv::rectangle(camera_frame_raw, cv::Point(x1,y1), cv::Point(x2,y2), color_map[result[i].class_id], 2);
    }
    cv::imwrite("./result_"+std::to_string(frame_num)+".jpg",camera_frame_raw);
    frame_num ++;
  }

  return EXIT_SUCCESS;
}

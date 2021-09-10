# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "desktop_tengine_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tengine:retina_anchors_calculator",
        "//mediapipe/calculators/tengine:tengine_converter_calculator",
        "//mediapipe/calculators/tengine:tengine_inference_calculator",
        "//mediapipe/calculators/tengine:tengine_retina_tensors_to_detections_calculator",
        "//mediapipe/calculators/tengine:tengine_yolov5_tensors_to_detections_calculator",
        "//mediapipe/calculators/tengine:tengine_tensors_to_classification_calculator",
        "//mediapipe/interface:inferpipe_tengine_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
    ],
    alwayslink=1,
)


cc_library(
    name = "desktop_jetson_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tengine:retina_anchors_calculator",
        "//mediapipe/calculators/tensorrt:tensorrt_converter_calculator",
        "//mediapipe/calculators/tensorrt:tensorrt_inference_calculator",
        "//mediapipe/calculators/tensorrt:tensorrt_yolov5_tensors_to_detections_calculator",
        "//mediapipe/calculators/tensorrt:tensorrt_retina_tensors_to_detections_calculator",
        "//mediapipe/calculators/tensorrt:inferpipe_tensorrt_calculator",
        "//mediapipe/calculators/tensorrt:tensorrt_tensors_to_classification_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
    ],
    alwayslink=1,
)

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)
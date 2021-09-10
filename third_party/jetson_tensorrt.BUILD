# Description:
#   OpenCV libraries for video/image processing on Linux

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

cc_library(
    name = "tensorrt",
    srcs = glob(
        [
            "lib/aarch64-linux-gnu/libnvinfer.so",
            "lib/aarch64-linux-gnu/libnvinfer_plugin.so",
            "lib/aarch64-linux-gnu/libnvonnxparser.so",
        ],
    ),
    hdrs = glob([
        "include/aarch64-linux-gnu/**/*.h*",
    ]),
    includes = [
        "include/aarch64-linux-gnu/",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

# Description:
#   cuda libraries on jetson

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

cc_library(
    name = "cuda_runtime",
    srcs = glob(
        [
            "lib64/libcudart.so",
            "lib64/libcublas.so",
            "lib64/libcudnn.so",
        ],
    ),
    hdrs = glob([
        "include/**/*.h*",
    ]),
    includes = [
        "include/",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

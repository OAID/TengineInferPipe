# Description:
#   tengine libraries for inference

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

cc_library(
    name = "tengine_runtime",
    srcs = glob(
        [
            "lib/libtengine-lite.so",
        ],
    ),
    hdrs = glob([
        "include/tengine/c_api.h",
    ]),
    includes = [
        "include/",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

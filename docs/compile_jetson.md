## inferpipe测试说明

#### 环境需求

* bazel >= 3.7	参考 [bazel安装说明](https://docs.bazel.build/versions/main/install.html) 安装bazel到系统中，或者下载对应版本[Releases · bazelbuild/bazel · GitHub](https://github.com/bazelbuild/bazel/releases)的可执行文件放到/usr/bin目录下
* CMake >= 3.10
* gcc >= 7.4
* opencv >= 3.4
* JetPack SDK >= 4.4    JetPack为nvidia

#### 编译

> <inferpipe-root-dir>表示TengineInferPipe项目根目录

1. 编译

   ```
   mkdir bazel_build
   bazel --output_base=./bazel_build build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:libdesktop_jetson_calculators.so
   ```

   生成的库文件在` <inferpipe-root-dir>/bazel-bin/mediapipe/examples/desktop/object_detection`目录下

   头文件见`<inferpipe-root-dir>/mediapipe/interface/inferpipe_tensorrt.h`


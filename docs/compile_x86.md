## compile-x86

#### 环境需求

* bazel >= 3.7	参考 [bazel安装说明](https://docs.bazel.build/versions/main/install.html) 安装bazel到系统中，或者下载对应版本[Releases · bazelbuild/bazel · GitHub](https://github.com/bazelbuild/bazel/releases)的可执行文件放到/usr/bin目录下

* CMake >= 3.10

* gcc >= 7.4

* opencv >= 3.4

  

#### 编译

> <tengine-lite-root-dir>表示tengine项目根目录，<inferpipe-root-dir>表示TengineInferPipe项目根目录

1. 参考[tengine-linux编译](./tengine_compile_linux.md)生成Tengine的x86平台的动态库

2. 编译TengineInferPipe 动态库

   ```
   mkdir tengine
   cp -r <tengine-lite-root-dir>/build/install/* ./tengine/
   mkdir bazel_build
   bazel --output_base=./bazel_build build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:libdesktop_tengine_calculators.so
   ```

生成的库文件在` <inferpipe-root-dir>/bazel-bin/mediapipe/examples/desktop/object_detection`目录下

头文件见`<inferpipe-root-dir>/mediapipe/interface/inferpipe_tengine.h`


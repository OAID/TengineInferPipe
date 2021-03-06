# 源码编译（TIM-VX ）

## 1. 简介

[TIM-VX](https://github.com/VeriSilicon/TIM-VX) 是 [VeriSilicon](https://www.verisilicon.com) 的 [OpenVX](https://www.khronos.org/openvx/) 张量接口模块(Tensor Interface Module for OpenVX，可以视作 OpenVX 的扩展支持)。
[Tengine Lite](https://github.com/OAID/Tengine) 已经完成 TIM-VX 的支持和集成， 在典型的 [VeriSilicon Vivante NPU](https://www.verisilicon.com/en/IPPortfolio/VivanteNPUIP) 硬件设备上，比如 [Khadas VIM3](https://www.khadas.com/vim3) (Amlogic A311D)、[Khadas VIM3L](https://www.khadas.com/vim3l) 上已经可以完成 Tengine 模型的推理。

**本文档以[Amlogic](https://www.amlogic.com) 的 [A311D](https://www.amlogic.com/#Products/393/index.html)为例，更多平台的Tengine编译请参考[Tengine compile_timvx]([Tengine/compile_timvx.md at tengine-lite · OAID/Tengine · GitHub](https://github.com/OAID/Tengine/blob/tengine-lite/doc/docs_zh/source_compile/compile_timvx.md))**

## 2. 如何编译
### 2.1 依赖项
依赖项有三部分：
> 第一部分是 TIM-VX 的源码，代码仓库在下方；
> 第二部分是 芯片对应板卡的 galcore.ko 的版本，对于 linux 平台，最低版本是 6.4.3.p0.286725；对于 Android 平台，最低版本是 6.4.3.279124+1。
> 第三部分是 TIM-VX 的依赖库，主要是直接依赖的 libCLC.so libGAL.so libOpenVX.so libOpenVXU.so libVSC.so libArchModelSw.so 等，不同的芯片最后的库文件依赖有可能是不完全相同的(比如 Android 上依赖的是 libarchmodelSw.so)，要根据拿到的 SDK 进行灵活调整。

### 2.2 编译过程
为了方便理解全流程的过程，首先描述编译的完整过程的流程。
在编译过程中，Tengine 将会先编译 TIM-VX 的代码，然后编译 Tengine-lite 的代码，并进行链接，链接时需要找到对应的各个芯片的用户态依赖库。需要注意的是，芯片板卡内部已经集成好的 galcore.ko 可能并不和依赖 so 的版本一致，编译时成功了也会有运行时错误打印提示版本不匹配。
此外，较早发布的系统，通常集成有较早版本的库文件，此时可以考虑烧入最新的镜像或运行升级命令进行更新。

### 2.3 拉取代码
这里假设是直接从 github 拉取代码，并且是拉取到同一个文件夹里。

#### 2.3.1 拉取 TIM-VX
```bash 
$ git clone https://github.com/VeriSilicon/TIM-VX.git
```

#### 2.3.2 拉取 Tengine-Lite
```bash
$ git clone https://github.com/OAID/Tengine.git tengine-lite
$ cd tengine-lite
```

### 2.4 选择 Tengine-Lite 集成编译 TIM-VX 方法
Tengine-Lite 支持三种 TIM-VX 的集成编译方法，具体如下：

> 第一种是将 TIM-VX 代码主要部分包含在 Tengine-Lite 的代码里，一并编译，最后得到单一 libtengine-lite.so，该 so 依赖 libCLC.so 等一系列 so；

> 第二种是不进行代码复制，但指定 CMake 选项 `-DTIM_VX_SOURCE_DIR=<你的拉取位置>/TIM-VX`，此时 Tengine-Lite 会查找 `TIM_VX_SOURCE_DIR` 指定的位置，并自动引用正确的 TIM-VX 代码文件，其他方面和第一种一致；

> 第三种是不进行集成编译，指定 CMake 选项 `-DTENGINE_ENABLE_TIM_VX_INTEGRATION=OFF`，TIM-VX 编译为单独的 libtim-vx.so，编译完成后，libtegine-lite.so 依赖 libtim-vx.so，libtim-vx.so 依赖其他的用户态驱动 libCLC.so 等一系列 so。

一般地，Tengine 推荐第一种方法以获得单一 so，并进行集成，这样可以在 Android APK 编译时减少一个依赖。这三种方法里，都需要准备 3rdparty 依赖，对于 Linux 和 Android 准备是有区别的，请注意分别进行区分。下面的几部分编译都是按照方法一进行描述的，其他方法的编译请根据方法一进行适当修改。

### 2.4 准备编译 x86_64 仿真环境
TIM-VX 提供了在 x86_64 宿主系统上的预编译依赖库，此部分依赖库可以在没有 NPU 的情况下，在 PC 上进行算法的开发和验证，其功能和板卡中是一致的，精度因计算路径区别略有影响但不影响验证。

#### 2.4.1 准备代码
这部分的目的是将 TIM-VX 的 include 和 src 目录复制到 Tengine-Lite 的 source/device/tim-vx 目录下，以便于 CMake 查找文件完成编译，参考命令如下：
``` bash
$ cd <tengine-lite-root-dir>
$ cp -rf ../TIM-VX/include  ./source/device/tim-vx/
$ cp -rf ../TIM-VX/src      ./source/device/tim-vx/
```

#### 2.4.3 准备 x86_64 3rdparty 依赖
这部分的目的是将 TIM-VX 的 x86_64 平台用户态驱动和其他头文件放入 Tengine-Lite 的 3rdparty 准备好，以便于链接阶段查找。预编译好的库文件和头文件已经在拉取下来的 TIM-VX 的 `prebuilt-sdk/x86_64_linux` 文件夹下，复制的参考命令如下：
``` bash
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/include
$ mkdir -p ./3rdparty/tim-vx/lib/x86_64
$ cp -rf ../TIM-VX/prebuilt-sdk/x86_64_linux/include/* ./3rdparty/tim-vx/include/
$ cp -rf ../TIM-VX/prebuilt-sdk/x86_64_linux/lib/*     ./3rdparty/tim-vx/lib/x86_64/
```

#### 2.4.4 执行编译
编译时需要打开 TIM-VX 后端的支持，参考命令如下：
```bash
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j`nproc` && make install
```
编译完成后，在 build 目录下的 install 文件夹里就有编译好的 libtengine-lite.so 库文件和头文件，可用于集成开发了。
需要注意的是，如果此时还想直接运行测试 example，需要手动指定一下 `LD_LIBRARY_PATH` 以便找到依赖的预编译好的用户态驱动。参考命令如下，需要根据实际情况调整：
``` bash
$ export LD_LIBRARY_PATH=<tengine-lite-root-dir>/3rdparty/tim-vx/lib/x86_64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### 2.5 准备编译 Khadas VIM3/VIM3L Linux 平台
VIM3/VIM3L 的 linux 平台是有 NPU 预置驱动的，可以通过 `sudo apt list --installed` 查看已经安装的版本：
``` bash
khadas@Khadas:~$ sudo apt list --installed | grep aml-npu
WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
aml-npu/now 6.4.3CB-2 arm64
khadas@Khadas:~$ 
```
对于 `6.4.3CB-2` 的版本(galcore 内核打印为 `6.4.3.279124CB`)，推荐进行联网执行升级：
``` bash
sudo apt-get update
sudo apt-get upgrade 
sudo apt-get full-upgrade
```
**当前的升级版本是 `6.4.4.3AAA`(galcore 的内核打印是 `6.4.4.3.310723AAA`)，升级后编译时不需要准备 3rdparty 的对应 so，系统默认的版本就可以满足要求。**

下面针对这两种情况，分别会进行讨论；然而新的 npu 驱动版本支持更多的 OP，升级总是没错的(如果烧录的是较早的镜像，NPU 版本可能是 `6.4.2`，和 `6.4.3CB-2` 一样不支持 TIM-VX，视同 `6.4.3CB-2` 进行编译即可，或进行推荐的升级按 `6.4.4` 及以上版本的流程进行编译)。

#### 2.5.1 准备代码
准备代码环节不用考虑 VIM3/VIM3L 的 NPU 版本，参考命令如下：
```bash
$ cd <tengine-lite-root-dir>
$ cp -rf ../TIM-VX/include  ./source/device/tim-vx/
$ cp -rf ../TIM-VX/src      ./source/device/tim-vx/
```

#### 2.5.2 准备 VIM3/VIM3L 较早版本 3rdparty 依赖
如果是较早版本的 NPU 依赖库 (**6.4.3.p0.286725**)，不打算/不可能进行升级，那么参考准备步骤如下：
```bash
$ wget -c https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz
$ tar zxvf aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz
$ mv aarch64_A311D_D312513_A294074_R311680_T312233_O312045 prebuild-sdk-a311d
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/include
$ mkdir -p ./3rdparty/tim-vx/lib/aarch64
$ cp -rf ../prebuild-sdk-a311d/include/*  ./3rdparty/tim-vx/include/
$ cp -rf ../prebuild-sdk-a311d/lib/*      ./3rdparty/tim-vx/lib/aarch64/
```
上面的命令是假设板子是 VIM3，对于 VIM3L，参考命令如下：
```bash
$ wget -c https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_S905D3_D312513_A294074_R311680_T312233_O312045.tgz
$ tar zxvf aarch64_S905D3_D312513_A294074_R311680_T312233_O312045.tgz
$ mv aarch64_S905D3_D312513_A294074_R311680_T312233_O312045 prebuild-sdk-s905d3
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/include
$ mkdir -p ./3rdparty/tim-vx/lib/aarch64
$ cp -rf ../prebuild-sdk-s905d3/include/*  ./3rdparty/tim-vx/include/
$ cp -rf ../prebuild-sdk-s905d3/lib/*      ./3rdparty/tim-vx/lib/aarch64/
```
注意，以上步骤都是假设 PC 的操作系统是 Linux，或者准备在板卡的 Linux 系统上编译；如果宿主系统是 WSL，请务必全部流程在 WSL 的命令行里面执行，不要在 windows 下执行，避免软连接问题。

#### 2.5.3 准备 VIM3/VIM3L 较早版本的编译
在这一步骤里，需要注意区分是要在 PC 上进行交叉编译还是在板卡上进行本地编译，本地编译比较简单，交叉编译复杂一些，但比较快。
假定前面的操作都是在板卡上进行的，那么执行的就是本地编译。本地编译时，尤其要注意，系统中的 lib 如果和 3rdparty 里面的有所不同时，那么链接时有可能链接到系统里面的版本，而不是 3rdparty 下面的版本。运行出错时需要用 ldd 命令检查一下链接情况。优先推荐先进行一下文件替换，然后再进行本地编译(如何替换请见 FAQ)。假设准备工作已经参考命令如下：
```bash
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j`nproc` && make install
```
编译完成后，在 build 目录下的 install 文件夹里就有编译好的 libtengine-lite.so 库文件和头文件，可用于集成开发了。
需要注意的是，如果此时还想直接运行测试 example，并且没有替换文件，需要手动指定一下 `LD_LIBRARY_PATH` 以便找到依赖的预编译好的用户态驱动。参考命令如下，需要根据实际情况调整：
``` bash
$ export LD_LIBRARY_PATH=<tengine-lite-root-dir>/3rdparty/tim-vx/lib/x86_64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
执行后，请再用 `ldd libtengine-lite.so` 命令检查一下，确保 NPU 驱动指向 3rdparty 目录的 so 一系列文件(替换是优先的选择)。

如果是交叉编译，那么请注意检查交叉编译的编译器不要高于板卡上的 gcc 版本，否则会导致在班子运行时，符号找不到的问题。确认这一点后，就可以进行交叉编译了。
如果手头没有合适的交叉编译工具，或者系统安装的版本较高，可以使用如下命令进行下载 linaro aarch64 工具链(或 arm release 的版本，二选一即可)：
```bash
$ wget -c http://releases.linaro.org/components/toolchain/binaries/7.3-2018.05/aarch64-linux-gnu/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz
$ tar xf gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz
```
下载完成后，进行交叉编译，参考命令如下：
``` bash
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_C_COMPILER=`pwd`/../../gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=`pwd`/../../gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++ ..
$ make -j`nproc` && make install
```
如果系统安装的 gcc-aarch64-linux-gnu/g++-aarch64-linux-gnu 满足板子的 gcc 版本要求，也可以通过如下参考命令进行编译：
``` bash
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
$ make -j`nproc` && make install
```

#### 2.5.4 准备 VIM3/VIM3L 最新版本 3rdparty 依赖
最佳实践是升级系统到最新版本，使得 NPU 的版本 >= 6.4.4。此时没有预置的 3rdparty，所以优先推荐的是采用在板上编译的方式进行编译，这时由于必要的 TIM-VX 依赖用户态驱动的 so 都已在系统目录下，不用准备 3rdparty 下的 lib 目录。参考命令如下：
``` bash
$ wget -c https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_S905D3_D312513_A294074_R311680_T312233_O312045.tgz
$ tar zxvf aarch64_S905D3_D312513_A294074_R311680_T312233_O312045.tgz
$ mv aarch64_S905D3_D312513_A294074_R311680_T312233_O312045 prebuild-sdk-s905d3
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/include
$ mkdir -p ./3rdparty/tim-vx/lib/aarch64
$ cp -rf ../prebuild-sdk-s905d3/include/*  ./3rdparty/tim-vx/include/
```
可以看出，只需要准备 include 文件夹到 3rdparty 即可。
如果确要进行交叉编译，此时下载到的 lib 目录下的 so 和板子是不匹配的，那么只需要按文件列表进行提取，这些文件在板卡的 `/usr/lib` 目录下；提取完成后，放入 `./3rdparty/tim-vx/lib/aarch64` 目录下即可。文件列表可见下载到的压缩包里面的 lib 目录，或 FAQ 部分的文件列表。

#### 2.5.5 准备 VIM3/VIM3L 最新版本的编译
在板子上进行本地编译很简单，参考命令如下：
``` bash
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j`nproc` && make install
```
如果是交叉编译，那么请参考前面 [2.5.3 准备 VIM3/VIM3L 较早版本的编译] 部分准备交叉工具链并进行编译即可。

## 3.  uint8 量化模型
The TIM-VX NPU backend needs the uint8 tmfile as it's input model file, you can **quantize** the tmfile from **float32** to **uint8** from here. 
- [Tengine Post Training Quantization Tools](../tools/quantize/README.md)
- [Download the uint8 quant tool](https://github.com/OAID/Tengine/releases/download/lite-v1.3/quant_tool_uint8)


## FAQ
Q：如何查看 NPU 驱动已经加载？  
A：用 lsmod 命令查看相关的驱动模块加载情况；以 VIM3 为例，检查 Galcore 内核驱动是否正确加载：  

``` bash
khadas@Khadas:~$ sudo lsmod
Module                  Size  Used by
iv009_isp_sensor      270336  0
iv009_isp_lens         69632  0
iv009_isp_iq          544768  0
galcore               663552  0
mali_kbase            475136  0
iv009_isp             540672  2
vpu                    49152  0
encoder                53248  0
# 中间打印略过
dhd                  1404928  0
sunrpc                446464  1
btrfs                1269760  0
xor                    20480  1 btrfs
raid6_pq              106496  1 btrfs
khadas@Khadas:~$
```
可以看到，`galcore 663552  0` 的打印说明了 galcore.ko 已经成功加载。

Q：如何查看 Galcore 的版本？  
A：使用 dmesg 命令打印驱动加载信息，由于信息较多，可以通过 grep 命令进行过滤。  
Linux 系统典型命令和打印如下：

``` bash
khadas@Khadas:~$ sudo dmesg | grep Galcore
[sudo] password for khadas: 
[   17.817600] Galcore version 6.4.3.p0.286725
khadas@Khadas:~$
```

Android 典型命令打印如下：

``` bash
kvim3:/ $ dmesg | grep Galcore
[   25.253842] <6>[   25.253842@0] Galcore version 6.4.3.279124+1
kvim3:/ $
```

可以看出，这个 linux 的 A311D 板卡加载的 galcore.ko 版本是 6.4.3.p0.286725，满足 linux 的版本最低要求。

Q：如何替换 galcore.ko？  
A：在 SDK 和内核版本升级过程中，有可能有需要升级对应的 NPU 部分的驱动，尽管推荐这一部分由板卡厂商完成，但实际上也有可能有测试或其他需求，需要直接使用最新的 NPU 版本进行测试。这时需要注意的是首先卸载 galcore.ko，然后再加载新的版本。具体命令为(假设新版本的 galcore.ko 就在当前目录)：  

``` bash
khadas@Khadas:~$ ls
galcore.ko
khadas@Khadas:~$ sudo rmmod galcore
khadas@Khadas:~$ sudo insmod galcore.ko
khadas@Khadas:~$ sudo dmesg | grep Galcore
[   17.817600] Galcore version 6.4.3.p0.286725
khadas@Khadas:~$
```

这样完成的是临时替换，临时替换在下次系统启动后就会加载回系统集成的版本；想要直接替换集成的版本可以通过 `sudo find /usr/lib -name galcore.ko` 查找一下默认位置，一个典型的路径是 `/usr/lib/modules/4.9.241/kernel/drivers/amlogic/npu/galcore.ko`，将 galcore.ko 替换到这个路径即可。
替换完成后，还需要替换用户态的相关驱动文件，一般有：

``` bash
libGAL.so
libNNGPUBinary.so
libOpenCL.so
libOpenVXU.so
libVSC.so
libCLC.so
libNNArchPerf.so
libNNVXCBinary.so
libOpenVX.so
libOvx12VXCBinary.so
libarchmodelSw.so
```

其中部分文件大小写、文件名、版本扩展名等可能不尽相同，需要保证替换前后旧版本的库及其软连接清理干净，新版本的库和软连接正确建立不疏失(有一两个 so 可能在不同的版本间是多出来或少掉的，是正常情况)。
这些文件一般在 `/usr/lib/` 文件夹里面(一些板卡可能没有预置用户态的驱动和内核驱动，这时自行添加后增加启动脚本加载内核驱动即可)。

Q：替换 galcore.ko 后，怎么检查细节状态？  
A：有时 insmod galcore.ko 后，lsmod 时还是有 galcore 模块的，但确实没加载成功。此时可以用 dmesg 命令确认下返回值等信息，核查是否有其他错误发生。  
Linux 典型打印如下：

``` bash
khadas@Khadas:~$ sudo dmesg | grep galcore
[    0.000000] OF: reserved mem: initialized node linux,galcore, compatible id shared-dma-pool
[   17.793965] galcore: no symbol version for module_layout
[   17.793997] galcore: loading out-of-tree module taints kernel.
[   17.817595] galcore irq number is 37.
khadas@Khadas:~$
```

Android 典型打印如下：

``` bash
kvim3:/ $ dmesg | grep galcore
[    0.000000] <0>[    0.000000@0]      c6c00000 - c7c00000,    16384 KB, linux,galcore
[   25.253838] <4>[   25.253838@0] galcore irq number is 53.
kvim3:/ $
```

Q：打印提示依赖库是未识别的 ELF 格式？  
A：目前 3rdparty 目录下的 include 目录几乎是通用的，lib 目录和平台有关；提示这个问题有可能是解压缩或复制过程中软连接断掉了(windows 系统下常见)，或者是准备的相关库文件和平台不匹配。  



## 附：支持的板卡链接
A311D:  [Khadas VIM3](https://www.khadas.com/vim3)  

## 附：其他
* 限于许可，Tengine-Lite 不能二次分发已经准备好的 3rdparty，请谅解。  
* 如果本文档描述的过程和 FAQ 没有覆盖您的问题，也欢迎加入 QQ 群 829565581 进一步咨询。  
* 不同版本的 TIM-VX 和 Tengine 对 OP 支持的情况有一定区别，请尽可能拉取最新代码进行测试评估。  
* 如果已有 OP 没有满足您的应用需求，可以分别在 TIM-VX 和 Tengine 的 issue 里创建一个新的 issue 要求支持；紧急或商业需求可以加入 QQ 群联系管理员申请商业支持。  
* Tengine 和 OPEN AI LAB 对文档涉及的板卡和芯片不做单独的保证，诸如芯片或板卡工作温度、系统定制、配置细节、价格等请与各自芯片或板卡供应商协商。  
* 如果贵司有板卡想要合作，可以加入 OPEN AI LAB 的 QQ 群联系管理员进一步沟通。
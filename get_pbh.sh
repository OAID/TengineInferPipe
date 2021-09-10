find ./bazel-bin/mediapipe/ -name '*.pb.h' |xargs tar -cvf media_pbh.tgz
find ./mediapipe/ -name '*.h' | xargs tar -cvf media_h.tgz

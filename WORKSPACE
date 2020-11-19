workspace(name = "com_google_sentencepiece")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    ],
)

http_archive(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "https://mirror.bazel.build/zlib.net/zlib-1.2.11.tar.gz",
        "https://zlib.net/zlib-1.2.11.tar.gz",
    ],
)

# proto_library, cc_proto_library, and java_proto_library rules implicitly
# depend on @com_google_protobuf for protoc and proto runtimes.
# This statement defines the @com_google_protobuf repo.
PROTOBUF_URLS = [
    "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
    "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
]

PROTOBUF_SHA256 = "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b"

PROTOBUF_STRIP_PREFIX = "protobuf-3.9.2"

http_archive(
    name = "com_google_protobuf",
    sha256 = PROTOBUF_SHA256,
    strip_prefix = PROTOBUF_STRIP_PREFIX,
    urls = PROTOBUF_URLS,
)

http_archive(
    name = "com_google_protobuf_cc",
    sha256 = PROTOBUF_SHA256,
    strip_prefix = PROTOBUF_STRIP_PREFIX,
    urls = PROTOBUF_URLS,
)

# Bazel toolchains
http_archive(
    name = "bazel_toolchains",
    sha256 = "4329663fe6c523425ad4d3c989a8ac026b04e1acedeceb56aa4b190fa7f3973c",
    strip_prefix = "bazel-toolchains-bc09b995c137df042bb80a395b73d7ce6f26afbe",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/bc09b995c137df042bb80a395b73d7ce6f26afbe.tar.gz",
        "https://github.com/bazelbuild/bazel-toolchains/archive/bc09b995c137df042bb80a395b73d7ce6f26afbe.tar.gz",
    ],
)

# ABSL
http_archive(
    name = "com_google_absl",
    sha256 = "d1535b8bd6ac41a0f899b906c1b5ef375136475e34dd53fb6775eb287487eef7",
    strip_prefix = "abseil-cpp-666fc1266bccfd8e6eaaa084e7b42580bb8eb199",
    urls = [
        "http://mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/666fc1266bccfd8e6eaaa084e7b42580bb8eb199.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/666fc1266bccfd8e6eaaa084e7b42580bb8eb199.tar.gz",
    ],
)

# Provides flags support until ABSL Flags is released.
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "ae27cdbcd6a2f935baa78e4f21f675649271634c092b1be01469440495609d0e",
    strip_prefix = "gflags-2.2.1",
    urls = [
        "http://mirror.tensorflow.org/github.com/gflags/gflags/archive/v2.2.1.tar.gz",
        "https://github.com/gflags/gflags/archive/v2.2.1.tar.gz",
    ],
)

# GoogleTest/GoogleMock framework. Used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
    strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
    urls = ["https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip"],  # 2019-01-07
)

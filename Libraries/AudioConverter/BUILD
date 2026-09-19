load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

swift_library(
    name = "AudioConverter",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "AudioConverter",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "AudioConverterTests",
    srcs = glob(["Tests/**/*.swift"], exclude = ["Tests/MiniMaxH3AudioReferenceTests.swift"]),
    deps = [
        ":AudioConverter",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "MiniMaxH3AudioReferenceTests",
    srcs = ["Tests/MiniMaxH3AudioReferenceTests.swift"],
    tags = ["manual", "requires-gpu"],
    deps = [
        ":AudioConverter",
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)

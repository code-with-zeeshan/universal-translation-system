// ios/UniversalTranslationSDK/Package.swift
// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "UniversalTranslationSDK",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .watchOS(.v8),
        .tvOS(.v15)
    ],
    products: [
        .library(
            name: "UniversalTranslationSDK",
            targets: ["UniversalTranslationSDK"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-async-algorithms.git", from: "1.0.0"),
        .package(url: "https://github.com/Flight-School/MessagePack.git", from: "1.2.4"),
        .package(url: "https://github.com/tsolomko/SWCompression.git", from: "4.8.0")
    ],
    targets: [
        .target(
            name: "UniversalTranslationSDK",
            dependencies: [
                .product(name: "Logging", package: "swift-log"),
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
                .product(name: "MessagePack", package: "MessagePack"),
                .product(name: "SWCompression", package: "SWCompression"),
                "UniversalEncoderBridge"
            ],
            path: "Sources",
            resources: [
                .process("Resources"),
                .copy("Models")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "UniversalEncoderBridge",
            dependencies: [],
            path: "Sources/EncoderBridge",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("../../../../encoder_core/include"),
                .define("ONNX_ML", to: "1"),
                .define("ONNX_NAMESPACE", to: "onnx")
            ],
            linkerSettings: [
                .linkedFramework("CoreML"),
                .linkedFramework("Accelerate"),
                .linkedLibrary("c++"),
                .linkedLibrary("onnxruntime", .when(platforms: [.iOS, .macOS]))
            ]
        ),
        .testTarget(
            name: "UniversalTranslationSDKTests",
            dependencies: ["UniversalTranslationSDK"],
            path: "Tests"
        ),
    ],
    cxxLanguageStandard: .cxx17
)
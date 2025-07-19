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
    ],
    targets: [
        .target(
            name: "UniversalTranslationSDK",
            dependencies: [
                .product(name: "Logging", package: "swift-log"),
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
            ],
            path: "Sources",
            resources: [
                .process("Resources")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "UniversalTranslationSDKTests",
            dependencies: ["UniversalTranslationSDK"],
            path: "Tests"
        ),
    ]
)
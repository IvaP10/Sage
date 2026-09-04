// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "SageMac",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "SageMac", targets: ["SageMac"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/swift-protobuf.git",
            exact: "1.38.1"
        ),
    ],
    targets: [
        .executableTarget(
            name: "SageMac",
            dependencies: [
                .product(name: "SwiftProtobuf", package: "swift-protobuf"),
            ],
            path: "Sources/SageMac",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
    ]
)

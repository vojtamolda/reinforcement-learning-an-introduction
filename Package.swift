// swift-tools-version:5.3
import PackageDescription


let package = Package(
    name: "Reinforcement Learning - An Introduction",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .library(
            name: "ReinforcementLearning",
            targets: ["MultiArmedBandit"]
        ),
    ],
    dependencies: [
        .package(
            name: "OpenSpiel",
            url: "https://github.com/deepmind/open_spiel.git",
            .branch("master")
        )
    ],
    targets: [
        .target(
            name: "MultiArmedBandit",
            dependencies: ["OpenSpiel", "Utilities"],
            path: "Chapter 2/MultiArmedBandit"
        ),
        .testTarget(
            name: "MultiArmedBanditTests",
            dependencies: ["MultiArmedBandit"],
            path: "Chapter 2/MultiArmedBanditTests"
        ),
        .target(
            name: "Utilities",
            dependencies: ["OpenSpiel"],
            path: "Tests/Utilities"
        ),
        .testTarget(
            name: "UtilitiesTests",
            dependencies: ["Utilities"],
            path: "Tests/UtilitiesTests"
        )
    ]
)

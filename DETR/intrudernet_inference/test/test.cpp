#include "test.hpp"

#include "../inference/inference.h"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void Test::runStressTest(
    std::string& testDataPath
)
{
    using clock =
        std::chrono::high_resolution_clock;

    InferenceEngine engine;

    if(!engine.loadModel(
        "model/vista_motion_detr.pth"
    ))
    {
        throw std::runtime_error(
            "Cannot load model"
        );
    }

    std::vector<std::string> imagePaths;

    for(const auto& entry :
        fs::directory_iterator(testDataPath))
    {
        if(entry.is_regular_file())
        {
            imagePaths.push_back(
                entry.path().string()
            );
        }
    }

    std::sort(
        imagePaths.begin(),
        imagePaths.end()
    );

    double totalMs = 0.0;

    int totalDetections = 0;

    int validFrames = 0;

    for(size_t i=1;i<imagePaths.size();i++)
    {
        cv::Mat prev =
            cv::imread(imagePaths[i-1]);

        cv::Mat curr =
            cv::imread(imagePaths[i]);

        if(prev.empty() || curr.empty())
            continue;

        auto t0 = clock::now();

        auto detections =
            engine.run(curr, prev);

        auto t1 = clock::now();

        double ms =
            std::chrono::duration<
                double,
                std::milli
            >(t1 - t0).count();

        totalMs += ms;

        validFrames++;

        std::cout
            << "\nFRAME: "
            << imagePaths[i]
            << "\n";

        std::cout
            << "Inference: "
            << ms
            << " ms\n";

        std::cout
            << "Detections: "
            << detections.size()
            << "\n";

        for(size_t d=0;d<detections.size();d++)
        {
            const auto& det =
                detections[d];

            std::cout
                << "  DET "
                << d
                << " score="
                << det.score
                << " staticness="
                << det.staticness
                << " box=("
                << det.cx << ", "
                << det.cy << ", "
                << det.w << ", "
                << det.h << ")\n";
        }

        totalDetections +=
            detections.size();
    }

    std::cout
        << "\n========================\n";

    std::cout
        << "Avg inference: "
        << (totalMs / validFrames)
        << " ms\n";

    std::cout
        << "Total detections: "
        << totalDetections
        << "\n";

    std::cout
        << "Processed frames: "
        << validFrames
        << "\n";

    std::cout
        << "========================\n";

    assert(validFrames > 0);
}
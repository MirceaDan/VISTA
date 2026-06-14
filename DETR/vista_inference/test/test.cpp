#include "test.hpp"

#include "../inference/inference.h"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void Test::runStressTest(std::string& testDataPath)
{
    using clock = std::chrono::high_resolution_clock;
    InferenceEngine engine;
    if(!engine.loadModel("/home/mircea/Desktop/VISTA/DETR/vista_inference/model/vistadetr_best.pth"))
    {
        throw std::runtime_error("Cannot load model");
    }

    fs::create_directories("/home/mircea/Desktop/VISTA/DETR/vista_inference/output");
    std::vector<std::string> imagePaths;
    for(const auto& entry : fs::directory_iterator(testDataPath))
    {
        if(entry.is_regular_file())
        {
            imagePaths.push_back(entry.path().string());
        }
    }

    std::sort(imagePaths.begin(), imagePaths.end());
    double totalMs = 0.0;
    int totalDetections = 0;
    int validFrames = 0;
    for(size_t i=1; i<imagePaths.size(); i++)
    {
        cv::Mat prev = cv::imread(imagePaths[i-1]);
        cv::Mat curr = cv::imread(imagePaths[i]);
        if(prev.empty() || curr.empty())
            continue;

        auto t0 = clock::now();
        auto detections = engine.run(curr, prev);
        auto t1 = clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if( i!= 1)
        {
            totalMs += ms;
            validFrames++;
        }

        std::cout << "\nFRAME: " << imagePaths[i] << "\n";
        std::cout << "Inference: " << ms << " ms\n";
        std::cout << "Detections: " << detections.size() << "\n";

        cv::Mat vis = curr.clone();
        int imgW = vis.cols;
        int imgH = vis.rows;
        for(size_t d=0;d<detections.size();d++)
        {
            const auto& det = detections[d];
            std::cout << "  DET " << d << " score=" << det.score << " staticness=" << det.staticness << " box=(" << det.cx << ", " << det.cy << ", " << det.w << ", " << det.h << ")\n";

            int x1 = static_cast<int>((det.cx - det.w * 0.5f) * imgW);
            int y1 = static_cast<int>((det.cy - det.h * 0.5f) * imgH);
            int x2 = static_cast<int>((det.cx + det.w * 0.5f) * imgW);
            int y2 = static_cast<int>((det.cy + det.h * 0.5f) * imgH);

            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(imgW - 1, x2);
            y2 = std::min(imgH - 1, y2);

            cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

            int cx = static_cast<int>(det.cx * imgW);
            int cy = static_cast<int>(det.cy * imgH);
            cv::circle(vis, cv::Point(cx, cy), 4, cv::Scalar(0, 0, 255), -1);

            char text[128];
            snprintf(text, sizeof(text), "#%zu S=%.2f ST=%.2f", d, det.score, det.staticness);
            cv::putText(vis, text, cv::Point(x1, std::max(20, y1 - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5,cv::Scalar(0, 255, 0), 1);
        }

        std::string outFile = "output/frame_" + std::to_string(i) + ".jpg";
        cv::imwrite(outFile, vis);
        std::cout << "Saved: " << outFile << "\n";
        totalDetections += detections.size();
    }

    std::cout << "\n========================\n";
    std::cout << "Avg inference: " << (totalMs / validFrames) << " ms\n";
    std::cout << "Total detections: " << totalDetections << "\n";
    std::cout << "Processed frames: " << validFrames << "\n";
    std::cout << "========================\n";

    assert(validFrames > 0);
}
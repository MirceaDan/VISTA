#ifndef INFERENCE_H_
#define INFERENCE_H_

#include <cmath>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

using namespace std;

struct Detection
{
    float score;
    float cx;
    float cy;
    float w;
    float h;
    float staticness;
};

class InferenceEngine
{
public:

    bool loadModel(const std::string& modelPath);

    std::vector<Detection> run(
        const cv::Mat& currentFrame,
        const cv::Mat& previousFrame
    );

private:

    torch::jit::Module model;

    torch::Tensor preprocessImage(
        const cv::Mat& frame
    );

    torch::Tensor computeOpticalFlow(
        const cv::Mat& currentFrame,
        const cv::Mat& previousFrame
    );
};

#endif
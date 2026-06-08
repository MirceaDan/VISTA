/*intrudernet_inference/
├── CMakeLists.txt
├── inference/
│   ├── inference.h
│   ├── inference.cpp
│   ├── preprocessing.cpp
│   ├── postprocessing.cpp
├── model/
│   └── vistadetr_best.pth
├── src/
│   └── main.cpp
└── third_party/
    └── libtorch/*/

#include "../inference/inference.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#if not TEST_MODE

cv::Mat latestFrame;
std::mutex frameMutex;
void getLatestFrame(RaspiCamera& cam)
{
    cv::Mat tmp;
    while(true)
    {
        if (cam.getFrame(tmp)) 
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            latestFrame = tmp.clone();
        }
    }
}

void overlayTextToImage(cv::Mat& image, vector<float> confidences)
{
    char rabbitBuf[128];
    snprintf(rabbitBuf, sizeof(rabbitBuf), "rabbit: %.2f", confidences[0]);
    std::string rabbit(rabbitBuf);

    char notRabbitBuf[128];
    snprintf(notRabbitBuf, sizeof(notRabbitBuf), "not rabbit: %.2f", confidences[1]);
    std::string not_rabbit(notRabbitBuf);

    cv::putText(
        image,
        rabbit,
        cv::Point(10, 30),              // (x, y) — stânga sus
        cv::FONT_HERSHEY_SIMPLEX,       // font
        1.0,                            // scale
        cv::Scalar(0, 255, 0),          // culoare BGR
        2,                              // grosime
        cv::LINE_AA                     // anti-aliasing
    );

    cv::putText(
        image,
        not_rabbit,
        cv::Point(10, 60),              // (x, y) — stânga sus
        cv::FONT_HERSHEY_SIMPLEX,       // font
        1.0,                            // scale
        cv::Scalar(0, 255, 0),          // culoare BGR
        2,                              // grosime
        cv::LINE_AA                     // anti-aliasing
    );
}

std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

std::string getBaseImagePath()
{
    std::filesystem::path basePath = "/home/mircea/Desktop/IntruderNet/images";

    std::filesystem::create_directories(basePath / "rabbit");
    std::filesystem::create_directories(basePath / "not_rabbit");

    return basePath.string();
}

void saveFrame(const cv::Mat& frame, const std::vector<float>& conf)
{
    static std::string basePath = getBaseImagePath();

    std::string label;
    if (conf[0] > conf[1])
        label = "rabbit";
    else
        label = "not_rabbit";

    std::string filename = basePath + "/" + label + "/" + getCurrentTimestamp() + ".jpg";

    bool ok = cv::imwrite(filename, frame);
    if(!ok)
    {
        std::cerr << "Failed saving image: " << filename << std::endl;
    }
}

bool isDaylight(const cv::Mat& frame)
{
    std::vector<uchar> pixels;
    pixels.assign(frame.datastart, frame.dataend);

    // Median
    size_t mid = pixels.size() / 2;
    std::nth_element(pixels.begin(), pixels.begin() + mid, pixels.end());
    double median = pixels[mid];

    return median > 20.0;
}

int main() {
    using clock = std::chrono::high_resolution_clock;

    RaspiCamera cam;
    InferenceEngine engine;

    engine.loadModelInfo("model/model_info.json");
    if(engine.modelInfo.quantized)
    {
        torch::globalContext().setQEngine(at::QEngine::QNNPACK);

        torch::jit::setGraphExecutorOptimize(false);

        engine.loadModel("model/cgtat_q.pth");
    }
    else
    {
        engine.loadModel("model/cgtat.pth");
    }

    if (!cam.start()) {
        std::cerr << "Failed to start camera\n";
        return -1;
    }

    std::thread camThread(getLatestFrame, std::ref(cam));

    while (true) 
    {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frameMutex);

            if(latestFrame.empty())
                continue;

            frame = latestFrame.clone();
        }

        if(frame.empty() || !isDaylight(frame))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto start = clock::now();
        vector<float> confidenceScores = engine.run(frame);
        auto end = clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Inference time: " << duration.count() << " ms inference scores: rabbit [" << confidenceScores[0] <<"] not_rabbit [" << confidenceScores[2] << "]\n";
            
        saveFrame(frame, confidenceScores);
    }

    cam.stop();
    camThread.join();
    return 0;
}
#else

#include "../test/test.hpp"
int main()
{
    std::string data_path = "test/test_data/";
    Test test;
    test.runStressTest(data_path);
}

#endif
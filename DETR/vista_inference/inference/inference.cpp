#include "inference.h"

bool InferenceEngine::loadModel(
    const std::string& modelPath
)
{
    try
    {
        model = torch::jit::load(
            modelPath,
            torch::kCPU
        );

        model.eval();

        return true;
    }
    catch(...)
    {
        return false;
    }
}

torch::Tensor InferenceEngine::preprocessImage(
    const cv::Mat& frame
)
{
    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(640,640)
    );

    cv::Mat rgb;

    cv::cvtColor(
        resized,
        rgb,
        cv::COLOR_BGR2RGB
    );

    auto tensor = torch::from_blob(
        rgb.data,
        {
            1,
            rgb.rows,
            rgb.cols,
            3
        },
        torch::kUInt8
    );

    tensor = tensor.permute({0,3,1,2});
    tensor = tensor.to(torch::kFloat32);
    tensor = tensor.div(255.0);

    return tensor.clone();
}

torch::Tensor InferenceEngine::computeOpticalFlow(
    const cv::Mat& currentFrame,
    const cv::Mat& previousFrame
)
{
    cv::Mat currGray;
    cv::Mat prevGray;

    cv::cvtColor(
        currentFrame,
        currGray,
        cv::COLOR_BGR2GRAY
    );

    cv::cvtColor(
        previousFrame,
        prevGray,
        cv::COLOR_BGR2GRAY
    );

    cv::Mat flow;

    cv::calcOpticalFlowFarneback(
        prevGray,
        currGray,
        flow,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    );

    cv::resize(
        flow,
        flow,
        cv::Size(640,640)
    );

    auto tensor = torch::from_blob(
        flow.data,
        {
            1,
            flow.rows,
            flow.cols,
            2
        },
        torch::kFloat32
    );

    tensor = tensor.permute({0,3,1,2});

    return tensor.clone();
}

std::vector<Detection> InferenceEngine::run(const cv::Mat& currentFrame, const cv::Mat& previousFrame)
{
    if(currentFrame.empty())
    {
        throw std::runtime_error("Current frame empty");
    }

    if(previousFrame.empty())
    {
        throw std::runtime_error("Previous frame empty");
    }

    torch::Tensor imageTensor = preprocessImage(currentFrame);

    torch::Tensor flowTensor = computeOpticalFlow(currentFrame, previousFrame);

    // Shape expected by TorchScript:
    // [B, MAX_OBJECTS, 7]
    // B = 1
    torch::Tensor motionTensor = torch::zeros({1,64,7}, torch::kFloat32);
    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(imageTensor);
    inputs.push_back(flowTensor);

    // Tensor, NOT List<Tensor>
    inputs.push_back(motionTensor);
    auto output = model.forward(inputs).toGenericDict();
    torch::Tensor logits =  output.at("pred_logits").toTensor();
    torch::Tensor boxes = output.at("pred_boxes").toTensor();
    torch::Tensor staticness = output.at("staticness").toTensor();

    logits = logits.squeeze(0);
    boxes = boxes.squeeze(0);
    float staticScore = staticness.mean().item<float>();
    std::vector<Detection> detections;
    const int numQueries = static_cast<int>(logits.size(0));
    for(int i = 0; i < numQueries; i++)
    {
        auto probs = torch::softmax(logits[i], -1);
        // DETR:
        // class 0 = beacon
        // class 1 = no-object
        float score = probs[0].item<float>();
        if(score < 0.5f)
        {
            continue;
        }

        Detection det;
        det.score = score;
        det.cx = boxes[i][0].item<float>();
        det.cy = boxes[i][1].item<float>();
        det.w = boxes[i][2].item<float>();
        det.h = boxes[i][3].item<float>();
        det.staticness = staticScore;
        detections.push_back(det);
    }

    return detections;
}
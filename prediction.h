//
// Created by piston on 2020/5/21.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

class FeatureGenerator {
public:
    explicit FeatureGenerator(const std::string &paramPath, int gpu = 0);

    std::vector<float> flattenPredict(const torch::Tensor &batchOutput);

    std::vector<std::vector<float>> batchPredict(const torch::Tensor &batchOutput);

    torch::Tensor predict(const torch::Tensor &input);

    int getModelStatus();

    int getEmbeddingSize();

    ~FeatureGenerator();

private:
    int modelStatus;
    int embeddingSize;
    std::string device;
    torch::jit::script::Module model;
};
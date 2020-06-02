//
// Created by piston on 2020/5/21.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#ifndef STUDY_PREDICTION_H
#define STUDY_PREDICTION_H

#endif //STUDY_PREDICTION_H

class FeatureGenerator {
public:

    std::string paramPath = "";

    explicit FeatureGenerator(std::string paramPath);

    std::vector<float> flattenPredict(const torch::Tensor &batchOutput);

    std::vector<std::vector<float>> batchPredict(const torch::Tensor &batchOutput);

    torch::Tensor predict(const torch::Tensor &input);

    int initModel();

    ~FeatureGenerator();

private:
    torch::jit::script::Module model;
};
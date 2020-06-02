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

//    std::string paramPath = "";

    explicit FeatureGenerator(const std::string &paramPath);

    std::vector<float> flattenPredict(const torch::Tensor &batchOutput);

    std::vector<std::vector<float>> batchPredict(const torch::Tensor &batchOutput);

    torch::Tensor predict(const torch::Tensor &input);

    int getModelStatus();

    c10::ClassTypePtr getModelType();

    ~FeatureGenerator();

private:
    int modelStatus;
    int embeddingSize;
    std::string modelType;
    torch::jit::script::Module model;
};
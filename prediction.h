//
// Created by piston on 2020/5/21.
//
#include <iostream>
#include <torch/script.h>

#ifndef STUDY_PREDICTION_H
#define STUDY_PREDICTION_H

#endif //STUDY_PREDICTION_H

class FeatureGenerator {
public:
    FeatureGenerator(std::string paramPath);

    std::string paramPath = "";

    std::vector<float> flattenPredict(torch::Tensor batchOutput);

    std::vector<std::vector<float>> batchPredict(torch::Tensor batchOutput);

    int initModel();

    ~FeatureGenerator();

private:
    torch::jit::script::Module model;

    torch::Tensor predict(torch::Tensor input);
};
//
// Created by piston on 2020/5/21.
//

#include "prediction.h"

std::vector<float> FeatureGenerator::flattenPredict(const torch::Tensor &batchInput) {
    auto batchOutput = this->predict(batchInput);
    std::vector<float> flattenFeats(
            batchOutput.data_ptr<float>(), batchOutput.data_ptr<float>() + batchOutput.numel());
    return flattenFeats;
}

std::vector<std::vector<float>> FeatureGenerator::batchPredict(const torch::Tensor &batchInput) {
    auto batchOutput = this->predict(batchInput);
    std::vector<std::vector<float>> batchFeats;
    for (int i = 0; i < batchOutput.size(0); ++i) {
        auto tar_tensor = batchOutput.index({i, "..."});
        batchFeats.emplace_back(tar_tensor.data_ptr<float>(), tar_tensor.data_ptr<float>() + tar_tensor.numel()
        );
    }
    return batchFeats;
}

torch::Tensor FeatureGenerator::predict(const torch::Tensor &input) {
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input.cuda());
    at::Tensor output = model.forward(inputs).toTensor().to(torch::kCPU, true);
    return output;
}

FeatureGenerator::FeatureGenerator(const std::string &paramPath) {
    try {
        this->model = torch::jit::load(paramPath);
        this->model.eval();
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the module\n";
        this->modelStatus = -1;
    }
    this->modelStatus = 0;
    auto inp = torch::randn({1, 3, 224, 224}).cuda();
    this->embeddingSize = this->predict(inp).size(1);
}


FeatureGenerator::~FeatureGenerator() = default;

int FeatureGenerator::getModelStatus() {
    return this->modelStatus;
}

int FeatureGenerator::getEmbeddingSize() {
    return this->embeddingSize;
}


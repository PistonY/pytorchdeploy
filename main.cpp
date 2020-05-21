//#include <torch/script.h>
//#include <torch/torch.h>
//#include <iostream>
//#include <memory>
#include "prediction.h"

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
//    torch::jit::script::Module module;
//    try {
//        module = torch::jit::load(argv[1]);
//    }
//    catch (const c10::Error &e) {
//        std::cerr << "error loading the module\n";
//        return -1;
//    }
//    std::cout << "Ok\n";
//
//    std::vector<torch::jit::IValue> inputs;
//    inputs.push_back(torch::ones({32, 3, 224, 224}).cuda());
//
//    at::Tensor output = module.forward(inputs).toTensor().to(torch::kCPU, /*non_blocking=*/true);
// //    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
//
// //    std::vector<float> vc(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
//    std::vector<std::vector<float>> batch_feats;
//    for (int i = 0; i < output.size(0); ++i) {
//        auto tar_ten = output.index({i, "..."});
//        batch_feats.push_back(
//                std::vector<float>(tar_ten.data_ptr<float>(), tar_ten.data_ptr<float>() + tar_ten.numel()));
//    }
//
//    std::cout << batch_feats.capacity() << ' '<< batch_feats[0].capacity() << std::endl;
    FeatureGenerator *fg = new FeatureGenerator(argv[1]);
    int status = fg->initModel();
    std::cout << status << '\n';
    auto inp = torch::randn({2, 3, 224, 224});
    auto fp = fg->flattenPredict(inp);
    auto bp = fg->batchPredict(inp);
    std::cout << fp.capacity() << '\n';

    return 0;
}
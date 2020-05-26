//#include <torch/script.h>
//#include <torch/torch.h>
//#include <iostream>
//#include <memory>
#include "prediction.h"
#include "transform.h"

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    auto *fg = new FeatureGenerator(argv[1]);
    int status = fg->initModel();
    std::cout << status << '\n';
    auto inp = torch::randn({2, 3, 224, 224});
    std::vector<float> fp = fg->flattenPredict(inp);
    auto bp = fg->batchPredict(inp);
    std::cout << fp.capacity() << '\n';

    return 0;
}
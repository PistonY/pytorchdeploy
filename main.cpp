#include <torch/torch.h>
#include "prediction.h"
#include "transform.h"

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
//    Init model.
    auto *fg = new FeatureGenerator(argv[1]);
    int status = fg->initModel();
    std::cout << status << '\n';
    auto inp = torch::randn({2, 3, 224, 224});
    auto fp = fg->flattenPredict(inp);
    auto bp = fg->batchPredict(inp);
    std::cout << fp.capacity() << '\n';

// process images and stack to one tensor
    auto img = cv::imread("/media/piston/data/Data/refer-frames/309269600/00024.jpg");
    auto trans_img = Transform::transOneImage(img);
    auto cp_img = trans_img.clone();
    std::cout << trans_img.sizes() << 'n';
    std::vector<at::Tensor> stc;
    stc.push_back(trans_img);
    stc.push_back(cp_img);

    auto sk_img = torch::stack(stc);
    std::cout << sk_img.sizes() << 'n';

//    torch::data::transforms::Stack
//    torch::data::datasets::StreamDataset
    return 0;
}
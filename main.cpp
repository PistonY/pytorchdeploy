//#include <torch/script.h>
#include <torch/torch.h>

#include <utility>
//#include <iostream>
//#include <memory>
#include "prediction.h"
#include "transform.h"

torch::Tensor trans(cv::Mat img) {
    auto resized_img = Transform::Resize(std::move(img), 256);
    auto cropped_img = Transform::CenterCrop(resized_img, 224);
//    auto tensor_img = Transform::ToTensor(cropped_img, 1);
    auto tensor_img = Transform::ToTensor(cropped_img);
    return tensor_img;
}

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

    auto img = cv::imread("/media/piston/data/Data/refer-frames/309269600/00024.jpg");
    auto trans_img = trans(img);
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
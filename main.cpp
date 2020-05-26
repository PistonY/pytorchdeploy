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

    auto img = cv::imread("/media/piston/data/Data/refer-frames/309269600/00024.jpg");
    auto resized_img = Transform::Resize(img, 256, img.cols, img.rows);
    auto cropped_img = Transform::CenterCrop(resized_img, 224, resized_img.cols, resized_img.rows);

    return 0;
}
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
//    std::cout << status << '\n';
//    auto inp = torch::randn({2, 3, 224, 224}).cuda().toType(torch::kHalf);
//    auto fp = fg->flattenPredict(inp);
//    auto bp = fg->batchPredict(inp);
//    std::cout << fp.capacity() << '\n';

// process images and stack to one tensor
    auto img1 = cv::imread("/media/piston/data/AQIYI_VIDEO_DNA/val_new/0/0.jpg");
    auto img2 = cv::imread("/media/piston/data/AQIYI_VIDEO_DNA/val_new/0/1.jpg");
    auto trans_img1 = Transform::transOneImage(img1).toType(torch::kHalf);
    auto trans_img2 = Transform::transOneImage(img2).toType(torch::kHalf);

    auto cl_img = trans_img1.clone();

    std::cout << trans_img1.dtype() << '\n';
    std::cout << cl_img.dtype() << '\n';

    std::vector<at::Tensor> stc;
    stc.push_back(trans_img1);
    stc.push_back(cl_img);

    auto sk_img = torch::stack(stc);
    auto sk_out = fg->predict(sk_img).cuda();
    std::cout << torch::sum(torch::mul(sk_out.index({0}),  sk_out.index({1}))) << std::endl;
    return 0;
}
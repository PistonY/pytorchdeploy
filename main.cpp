#include <torch/torch.h>
#include "prediction.h"
#include "transform.h"

template<typename T>
float vectorSimilarity(std::vector<T> m1, std::vector<T> m2) {
    auto length = m1.capacity();
    if (length != m2.capacity()) {
        return -1;
    }
    float sum = 0;
    for (unsigned long i = 0; i < length; ++i) {
        sum += m1[i] * m2[i];
    }
    return sum;
}

void test(std::string &paramPath, int batchSize = 64, int testLoop = 10000) {
    auto *fg = new FeatureGenerator(paramPath);
    int status = fg->getModelStatus();
    int embs = fg->getEmbeddingSize();

    std::cout << "status:" << status << " EmbeddingSize:" << embs << '\n';
    for (int i = 0; i < testLoop; ++i) {
        std::vector<at::Tensor> stc;
        for (int j = 0; j < batchSize; ++j) {
            auto img = cv::imread("/media/piston/data/AQIYI_VIDEO_DNA/val_new/0/0.jpg");
            auto trans_img = Transform::transOneImage(img);
            stc.push_back(trans_img);
        }
        at::TensorList tl(stc);
        auto sk_img = torch::stack(tl);
        auto ten_out = fg->batchPredict(sk_img);
        std::cout << i << std::endl;
    }
}

//int main1(int argc, const char *argv[]) {
//    if (argc != 2) {
//        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
//        return -1;
//    }
////    Init model.
//    auto *fg = new FeatureGenerator(argv[1]);
//    int status = fg->getModelStatus();
//    int embs = fg->getEmbeddingSize();
//
//    std::cout << "status:" << status << " EmbeddingSize:" << embs << '\n';
//
//    for (int i = 0; i < 100000; ++i) {
//        auto inp = torch::randn({64, 3, 224, 224}).cuda();
//        auto fp = fg->flattenPredict(inp);
//        auto bp = fg->batchPredict(inp);
//        std::cout << fp.capacity() << '\n';
//    }
//
//// process images and stack to one tensor
//
//    auto img1 = cv::imread("/media/piston/data/AQIYI_VIDEO_DNA/val_new/0/0.jpg");
//    auto img2 = cv::imread("/media/piston/data/AQIYI_VIDEO_DNA/val_new/0/1.jpg");
//
//
//    auto trans_img1 = Transform::transOneImage(img1);
//    auto trans_img2 = Transform::transOneImage(img2);
//
//    std::cout << trans_img1.dtype() << '\n';
//
//    std::vector<at::Tensor> stc;
//    stc.push_back(trans_img1);
//    stc.push_back(trans_img2);
//
//    at::TensorList tl(stc);
//    auto sk_img = torch::stack(tl);
//
//    auto ten_out = fg->predict(sk_img).cuda();
//    std::cout << torch::sum(torch::mul(ten_out.index({0}), ten_out.index({1}))) << std::endl;
//
//    auto bp = fg->batchPredict(sk_img);
//    std::cout << vectorSimilarity(bp[0], bp[1]) << "\n";
//    return 0;
//
//}
int main(int argc, const char *argv[]){
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
    test((std::string &) argv[1]);
}
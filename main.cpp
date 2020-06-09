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


int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
//    Init model.
//    cudaSetDevice(1);
    cv::cuda::setDevice(1);
    int gpu = 1;
    auto *fg = new FeatureGenerator(argv[1], gpu);
    int status = fg->getModelStatus();
    int embs = fg->getEmbeddingSize();

    std::cout << "status:" << status << " EmbeddingSize:" << embs << '\n';


    auto inp = torch::randn({64, 3, 224, 224});
    auto fp = fg->flattenPredict(inp);
    auto bp = fg->batchPredict(inp);
    std::cout << fp.capacity() << '\n';


// process images and stack to one tensor

    auto img1 = cv::imread("/media/piston/data/AQIYI_VIDEO_DNA/val_new/32/0.jpg");
    auto img2 = cv::imread("/media/piston/data/AQIYI_VIDEO_DNA/val_new/32/1.jpg");


    auto gImg1 = cv::cuda::GpuMat(img1);
    auto gImg2 = cv::cuda::GpuMat(img2);


    auto trans_img1 = Transform::transOneImage(gImg1, gpu);
    auto trans_img2 = Transform::transOneImage(gImg2, gpu);

    std::cout << trans_img1.index({0, 0, "..."}).slice(/*dim=*/0, /*start=*/0, /*end=*/10) << '\n';

    std::vector<at::Tensor> stc;
    stc.push_back(trans_img1);
    stc.push_back(trans_img2);
    auto sk_img = torch::stack(stc);
    std::cout << sk_img.index({0, 0, 0, "..."}).slice(/*dim=*/0, /*start=*/0, /*end=*/10) << '\n';

    auto ten_out = fg->predict(sk_img);

    std::cout << torch::sum(torch::mul(ten_out.index({0}), ten_out.index({1}))) << std::endl;

    bp = fg->batchPredict(sk_img);
    std::cout << vectorSimilarity(bp[0], bp[1]) << "\n";
    return 0;
}

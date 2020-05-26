//
// Created by piston on 2020/5/26.
//
#include <torch/torch.h>
#include <opencv2/opencv.hpp>


namespace Transform {
    // Rect(left, top, w, h)
    cv::Mat Crop(const cv::Mat &img, const cv::Rect &rect) {
        cv::Mat roi_img = img(rect);
        return roi_img;
    }

    /*input shape should be BxHxWxC RGB format*/
    torch::Tensor ToTensor(const cv::Mat &img, int batch_size) {
        at::Tensor tensor = torch::from_blob(img.data, {batch_size, 224, 224, 3});
        tensor = tensor.permute({0, 3, 1, 2});
        return tensor;
    }

    cv::Mat CenterCrop(const cv::Mat &img, int size, int w, int h) {
        int th = size, tw = size;
        int i = (h - th) / 2;
        int j = (w - tw) / 2;
        auto roi_img = Crop(img, cv::Rect(j, i, tw, th));
        return roi_img;
    }

    cv::Mat Resize(cv::Mat img, int size, int w, int h) {
        if ((w <= h and w == size) or (h <= w and h == size)) {
            return img;
        } else {
            int ow, oh;
            if (w < h) {
                ow = size;
                oh = int(size * h / w);
            } else {
                oh = size;
                ow = int(size * w / h);
            }
            cv::resize(img, img, cv::Size(ow, oh));
            return img;
        }
    }
};
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

    torch::Tensor ToTensor(const cv::Mat &img, int batch_size) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        at::Tensor tensor = torch::from_blob(img.data, {batch_size, img.rows, img.cols, 3});
        tensor = tensor.permute({0, 3, 1, 2}) / 255.;
        return tensor;
    }

    torch::Tensor ToTensor(const cv::Mat &img) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        at::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols, 3});
        tensor = tensor.permute({2, 0, 1}) / 255.;
        return tensor;
    }

    cv::Mat CenterCrop(const cv::Mat &img, int size) {
        int th = size, tw = size;
        int w = img.cols, h = img.rows;
        int i = (h - th) / 2;
        int j = (w - tw) / 2;
        auto roi_img = Crop(img, cv::Rect(j, i, tw, th));
        return roi_img;
    }

    cv::Mat Resize(cv::Mat img, int size) {
        int w = img.cols, h = img.rows;
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

    torch::Tensor transOneImage(cv::Mat img, int size = 224, float ratio = 0.875) {
        auto resized_img = Resize(std::move(img), int(float(size) / ratio));
        auto cropped_img = CenterCrop(resized_img, size);
        auto tensor_img = ToTensor(cropped_img);
        return tensor_img;
    }
};

//
// Created by piston on 2020/5/26.
//
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <utility>
#include <cuda_runtime.h>

namespace Transform {
    torch::Tensor ToTensor(cv::Mat &img) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
        at::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols, 3},
                                             torch::TensorOptions().requires_grad(false));
        tensor = tensor.permute({2, 0, 1});
        return tensor.contiguous();
    }

//    torch::Tensor ToTensor(cv::cuda::GpuMat &img, int gpu) {
//        std::string device = std::string("cuda:").append(std::to_string(gpu));
//        cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);
//        cv::cuda::GpuMat img_i;
//        img.convertTo(img_i, CV_32FC3, 1.0f / 255.0f);
//        uchar *data;
//        cudaMalloc((void **) &data, img.rows * img.cols * 3 * 4);
//        for (int j = 0; j < img.rows; j++) {
//            cudaMemcpy(data + j * img.cols * 3 * 4, img_i.ptr<uchar>(j), img.cols * 3 * 4, cudaMemcpyDeviceToDevice);
//        }
//
//        at::Tensor tensor = torch::from_blob(data, {img.rows, img.cols, 3}, &cudaFree,
//                                             torch::TensorOptions()
//                                                     .requires_grad(false)
//                                                     .device(device));
//        tensor = tensor.permute({2, 0, 1});
//        return tensor.contiguous();
//    }
    void deleter(void *arg) {};

    torch::Tensor ToTensor(cv::cuda::GpuMat &img, int gpu) {
        std::string device = std::string("cuda:").append(std::to_string(gpu));
        cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::cuda::GpuMat img_i;
        img.convertTo(img_i, CV_32FC3, 1.0f / 255.0f);

        std::vector<int64_t> sizes = {static_cast<int64_t>(img_i.channels()),
                                      static_cast<int64_t>(img_i.rows),
                                      static_cast<int64_t>(img_i.cols)};

        long long step = img_i.step / sizeof(float);
        std::vector<int64_t> strides = {1, step, static_cast<int64_t>(img_i.channels())};
        auto options = torch::TensorOptions().requires_grad(false).device(device);
        torch::Tensor tensor = torch::from_blob(img_i.data,
                                                sizes,
                                                strides,
                                                deleter,
                                                options);
//        tensor = tensor.permute({2, 0, 1});
        return tensor.contiguous();
    }

    // Rect(left, top, w, h)
    template<typename T>
    T Crop(const T &img, const cv::Rect &rect) {
        T roi_img = img(rect);
        return roi_img;
    }

    template<typename T>
    T CenterCrop(const T &img, int size) {
        int th = size, tw = size;
        int w = img.cols, h = img.rows;
        int i = (h - th) / 2;
        int j = (w - tw) / 2;
        auto roi_img = Crop(img, cv::Rect(j, i, tw, th));
        return roi_img;
    }


    cv::Mat Resize(cv::Mat img, int size) {
        int w = img.cols, h = img.rows;
        if (!((w <= h and w == size) or (h <= w and h == size))) {
            int ow, oh;
            if (w < h) {
                ow = size;
                oh = int(size * h / w);
            } else {
                oh = size;
                ow = int(size * w / h);
            }
            cv::resize(img, img, cv::Size(ow, oh));
        }
        return img;
    }

    cv::cuda::GpuMat Resize(const cv::cuda::GpuMat &img, int size) {
        int w = img.cols, h = img.rows;
        if (!((w <= h and w == size) or (h <= w and h == size))) {
            int ow, oh;
            if (w < h) {
                ow = size;
                oh = int(size * h / w);
            } else {
                oh = size;
                ow = int(size * w / h);
            }
            cv::cuda::GpuMat aftImg;
            cv::cuda::resize(img, aftImg, cv::Size(ow, oh));
            return aftImg;
        } else {
            return img;
        }
    }

    torch::Tensor transOneImage(cv::Mat &img, int size = 224, float ratio = 0.875) {
        auto resized_img = Resize(img, int(float(size) / ratio));
        auto cropped_img = CenterCrop(resized_img, size);
        auto tensor_img = ToTensor(cropped_img);
        return tensor_img;
    }

    torch::Tensor transOneImage(cv::cuda::GpuMat &img, int gpu, int size = 224, float ratio = 0.875) {
        auto resized_img = Resize(img, int(float(size) / ratio));
        auto cropped_img = CenterCrop(resized_img, size);
        auto tensor_img = ToTensor(cropped_img, gpu);
        return tensor_img;
    }
};

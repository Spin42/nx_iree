// Writes the 1x3x640x640 float input blob of an image and OpenCV DNN's YOLOv5
// output for it as raw little-endian float files.
// Usage: yolo_dump model.onnx image.jpg blob.bin output.bin
#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: yolo_dump model.onnx image.jpg blob.bin output.bin\n";
        return 2;
    }
    cv::Mat image = cv::imread(argv[2]);
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    cv::dnn::Net net = cv::dnn::readNetFromONNX(argv[1]);
    net.setInput(blob);
    cv::Mat out = net.forward();
    std::ofstream(argv[3], std::ios::binary).write(reinterpret_cast<const char*>(blob.data), blob.total() * sizeof(float));
    std::ofstream(argv[4], std::ios::binary).write(reinterpret_cast<const char*>(out.data), out.total() * sizeof(float));
    std::cout << "blob " << blob.total() << " floats, output " << out.size[1] << "x" << out.size[2] << "\n";
    return 0;
}

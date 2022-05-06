#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>

using namespace std;

int main()
{
	string model_path = "resout.pt";
	string img_path = "1.jpg";


	//载入模型 
	torch::jit::script::Module module;
	module = torch::jit::load(model_path);


	//输入图像
	cv::Mat image = cv::imread(img_path);
	int img_h = image.rows;
	int img_w = image.cols;
	int depth = image.channels();
	cv::Mat img_float;
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	cv::resize(image, image, cv::Size(512, 512));
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
	auto img_tensor = torch::from_blob(image.data, { 1, 512, 512, depth });
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });


	//模型计算
	std::vector<torch::jit::IValue> inputs;
	torch::Tensor out;
	inputs.push_back(img_tensor);
	out = module.forward(std::move(inputs)).toTensor();

	//结果处理
	out = out[0];
	out = out.permute({ 1, 2, 0 }).detach();
	out = torch::softmax(out, 2);
	out = out.argmax(2);
	out = out.mul(10).clamp(0, 255).to(torch::kU8); 
	out = out.to(torch::kCPU);

	//保存图片
	int height, width;
	height = out.size(0);
	width = out.size(1);
	cv::Mat resultImg(cv::Size(512, 512), CV_8U, out.data_ptr()); // 将Tensor数据拷贝至Mat
	cv::resize(resultImg, resultImg, cv::Size(img_w, img_h));
	cv::imwrite("result.jpg", resultImg);

	
	return 0;
}


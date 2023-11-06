#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char ** argv)
{
	if (argc != 2)
		return -1;

	std::string image_path{argv[1]};
	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
	if(img.empty())
	{
		std::cout << "Could not read the image: " << image_path << std::endl;
		return 1;
	}
	cv::imshow("Display window", img);
	int k = cv::waitKey(0);
	return 0;
}
#include <iostream>
#include <opencv2/opencv.hpp>
#include <lib.h>


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

	cv::Mat NoiseSaltPepper(cv::Size(img.rows, img.cols), img.type());
	cv::Mat NoiseGaussian(cv::Size(img.rows, img.cols), img.type());
	cv::Mat NoisePoisson(cv::Size(img.rows, img.cols), img.type());


	cv::Mat filtered(cv::Size(img.rows, img.cols), img.type());
	cv::Mat filteredSaltPepper(cv::Size(img.rows, img.cols), img.type());
	cv::Mat filteredGaussian(cv::Size(img.rows, img.cols), img.type());
	cv::Mat filteredPoisson(cv::Size(img.rows, img.cols), img.type());

	
	ajouterBruitSelEtPoivre(img,NoiseSaltPepper,3);
	ajouterBruitGaussian(img,NoiseGaussian,20);
	ajouterBruitPoisson(img,NoisePoisson,20);

	bilateral(img, filtered, 5, 150, 150);
	bilateral(NoiseSaltPepper, filteredSaltPepper, 5, 150, 150);
	bilateral(NoiseGaussian, filteredGaussian, 5, 150, 150);
	bilateral(NoisePoisson, filteredPoisson, 5, 150, 150);

    cv::Mat result;
	cv::Mat vertical_base;
	cv::Mat vertical_SaltPepper;
	cv::Mat vertical_Gaussian;
	cv::Mat vertical_Poisson;

    cv::vconcat(img, filtered, vertical_base);
	cv::vconcat(NoiseSaltPepper,filteredSaltPepper, vertical_SaltPepper);
	cv::vconcat(NoiseGaussian, filteredGaussian, vertical_Gaussian);
	cv::vconcat(NoisePoisson, filteredPoisson, vertical_Poisson);

	cv::hconcat(vertical_base,vertical_SaltPepper,result);
	cv::hconcat(result,vertical_Gaussian,result);
	cv::hconcat(result,vertical_Poisson,result);
	

	cv::imshow("Display window", result);	
	int k = cv::waitKey(0);
	return 0;
}
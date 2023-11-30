#include <iostream>
#include <opencv2/opencv.hpp>
#include "lib.h"

int main(int argc, char** argv) {
    if (argc != 2)
        return -1;

    std::string image_path{argv[1]};
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    if (img.empty()) {
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

    ajouterBruitSelEtPoivre(img, NoiseSaltPepper, 3);
    ajouterBruitGaussian(img, NoiseGaussian, 20);
    ajouterBruitPoisson(img, NoisePoisson, 20);

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
    cv::vconcat(NoiseSaltPepper, filteredSaltPepper, vertical_SaltPepper);
    cv::vconcat(NoiseGaussian, filteredGaussian, vertical_Gaussian);
    cv::vconcat(NoisePoisson, filteredPoisson, vertical_Poisson);

    cv::hconcat(vertical_base, vertical_SaltPepper, result);
    cv::hconcat(result, vertical_Gaussian, result);
    cv::hconcat(result, vertical_Poisson, result);

    // Paramètres pour le calcul SSIM
    int windowSize = 8; // Taille de la fenêtre locale
    double C1 = 6.5025, C2 = 58.5225; // Constantes pour stabiliser la division

    // Calculer et afficher les PSNR et SSIM
    double psnrBase = calculatePSNR(img, filtered);
    double psnrNoiseSaltPepper = calculatePSNR(img, NoiseSaltPepper);
    double psnrFilteredSaltPepper = calculatePSNR(filtered, filteredSaltPepper);

    double psnrNoiseGaussian = calculatePSNR(img, NoiseGaussian);
    double psnrFilteredGaussian = calculatePSNR(filtered, filteredGaussian);

    double psnrNoisePoisson = calculatePSNR(img, NoisePoisson);
    double psnrFilteredPoisson = calculatePSNR(filtered, filteredPoisson);

    double ssimBase = calculateSSIM(img, filtered, windowSize, C1, C2);
    double ssimNoiseSaltPepper = calculateSSIM(img, NoiseSaltPepper, windowSize, C1, C2);
    double ssimFilteredSaltPepper = calculateSSIM(filtered, filteredSaltPepper, windowSize, C1, C2);

    double ssimNoiseGaussian = calculateSSIM(img, NoiseGaussian, windowSize, C1, C2);
    double ssimFilteredGaussian = calculateSSIM(filtered, filteredGaussian, windowSize, C1, C2);

    double ssimNoisePoisson = calculateSSIM(img, NoisePoisson, windowSize, C1, C2);
    double ssimFilteredPoisson = calculateSSIM(filtered, filteredPoisson, windowSize, C1, C2);

    // Afficher les résultats
    std::cout<< "SALT AND PEPPER "<<std::endl;
    std::cout << "PSNR (Original to Noisy): " << psnrNoiseSaltPepper << " dB" << std::endl;
    std::cout << "PSNR (Original to Filtered): " << psnrFilteredSaltPepper << " dB" << std::endl;
    std::cout << "Taux d'amélioration PSNR : " << (psnrFilteredSaltPepper - psnrNoiseSaltPepper) / psnrNoiseSaltPepper * 100 << "%" << std::endl;
    std::cout << "SSIM (Original to Noisy): " << ssimNoiseSaltPepper << std::endl;
    std::cout << "SSIM (Original to Filtered): " << ssimFilteredSaltPepper << std::endl;
    std::cout << "Taux d'amélioration SSIM : " << (ssimFilteredSaltPepper - ssimNoiseSaltPepper) / ssimNoiseSaltPepper * 100 << "%" << std::endl;
    std::cout<< std::endl;
    std::cout<< "GAUSSIAN "<<std::endl;
    std::cout << "PSNR (Original to Noisy): " << psnrNoiseGaussian << " dB" << std::endl;
    std::cout << "PSNR (Original to Filtered): " << psnrFilteredGaussian << " dB" << std::endl;
    std::cout << "Taux d'amélioration PSNR : " << (psnrFilteredGaussian - psnrNoiseGaussian) / psnrNoiseGaussian * 100 << "%" << std::endl;
    std::cout << "SSIM (Original to Noisy): " << ssimNoiseGaussian << std::endl;
    std::cout << "SSIM (Original to Filtered): " << ssimFilteredGaussian << std::endl;
    std::cout << "Taux d'amélioration SSIM : " << (ssimFilteredGaussian - ssimNoiseGaussian) / ssimNoiseGaussian * 100 << "%" << std::endl;
    std::cout<< std::endl;
    std::cout<< "POISSON "<<std::endl;
    std::cout << "PSNR (Original to Noisy): " << psnrNoisePoisson << " dB" << std::endl;
    std::cout << "PSNR (Original to Filtered): " << psnrFilteredPoisson << " dB" << std::endl;
    std::cout << "Taux d'amélioration PSNR : " << (psnrFilteredPoisson - psnrNoisePoisson) / psnrNoisePoisson * 100 << "%" << std::endl;
    std::cout << "SSIM (Original to Noisy): " << ssimNoisePoisson << std::endl;
    std::cout << "SSIM (Original to Filtered): " << ssimFilteredPoisson << std::endl;
    std::cout << "Taux d'amélioration SSIM : " << (ssimFilteredPoisson - ssimNoisePoisson) / ssimNoisePoisson * 100 << "%" << std::endl;
    std::cout<< std::endl;
    

    
    // Sauvegarder les images au format PNG
    cv::imwrite("result_base.png", result);
    cv::imwrite("result_salt_pepper.png", vertical_SaltPepper);
    cv::imwrite("result_gaussian.png", vertical_Gaussian);
    cv::imwrite("result_poisson.png", vertical_Poisson);

    // Sauvegarder les images bruitées au format PNG
    cv::imwrite("noisy_salt_pepper.png", NoiseSaltPepper);
    cv::imwrite("noisy_gaussian.png", NoiseGaussian);
    cv::imwrite("noisy_poisson.png", NoisePoisson);

    cv::imshow("Display window", result);
    int k = cv::waitKey(0);
    return 0;
}


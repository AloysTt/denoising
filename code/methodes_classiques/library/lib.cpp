#include <lib.h>
#include <random>

void bilateral(const cv::Mat & image, cv::Mat & out, int d, double color, double space)
{
	int w = image.cols;
	int h = image.rows;

	for (int row=0; row<h; ++row)
	{
		for (int col=0; col<w; ++col)
		{
			float norm = 0.0f;
			cv::Vec3i filtered;

			for (int dy=-d/2; dy<d/2; ++dy)
			{
				for (int dx=-d/2; dx<d/2; ++dx)
				{
					int x = col + dx;
					int y = row + dy;
					if (x < 0 || x >= w || y < 0 || y >= h)
						continue;

					const auto & px1 = image.at<cv::Vec3b>(row, col);
					const auto & px2 = image.at<cv::Vec3b>(y, x);
					int intensityDistance = std::abs(static_cast<int>(px1[0]+px1[1]+px1[2]) - static_cast<int>(px2[0]+px2[1]+px2[2]));

					float intensityWeight = gaussian(intensityDistance, color);
					float spatialWeight = gaussian(sqrt(dy*dy+dx*dx), space);

					float weight = spatialWeight * intensityWeight;
					norm += weight;
					filtered += weight * image.at<cv::Vec3b>(y, x);
				}
			}

			out.at<cv::Vec3b>(row, col) = filtered/norm;
		}
	}
}

float gaussian(float x, float sigma)
{
	return exp(-(x * x) / (2.0f * sigma * sigma));
}

void ajouterBruitPoisson(cv::Mat& imageRef, cv::Mat& image, double intensiteMoyenne)
{
	image = imageRef.clone();
    std::default_random_engine generateur;
    std::poisson_distribution<int> distributionPoisson(intensiteMoyenne);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {

            int echantillon = distributionPoisson(generateur);

            if (image.channels() == 1) {

                image.at<uchar>(y, x) = cv::saturate_cast<uchar>(image.at<uchar>(y, x) + echantillon);
            } else if (image.channels() == 3) {

                for (int c = 0; c < image.channels(); ++c) {
                    image.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(image.at<cv::Vec3b>(y, x)[c] + echantillon);
                }
            }
        }
    }
}

void ajouterBruitGaussian(cv::Mat& imageRef, cv::Mat& image, double intensiteBruit)
{
	image = imageRef.clone();
    std::default_random_engine generateur;
    std::normal_distribution<double> distributionGaussienne(0.0, intensiteBruit);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {

            double echantillon = distributionGaussienne(generateur);

            if (image.channels() == 1) {

                image.at<uchar>(y, x) = cv::saturate_cast<uchar>(image.at<uchar>(y, x) + echantillon);
            } else if (image.channels() == 3) {

                for (int c = 0; c < image.channels(); ++c) {
                    image.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(image.at<cv::Vec3b>(y, x)[c] + echantillon);
                }
            }
        }
    }
}


void ajouterBruitSelEtPoivre(cv::Mat& imageRef, cv::Mat& image, double pourcentageBruit)
{
	    image = imageRef.clone();
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int r = std::rand() % 100;
            if (r < pourcentageBruit / 2) {

                if (image.channels() == 1) {
                    image.at<uchar>(y, x) = 0;
                } else if (image.channels() == 3) {
                    for (int c = 0; c < image.channels(); ++c) {
                        image.at<cv::Vec3b>(y, x)[c] = 0;
                    }
                }
            } else if (r < pourcentageBruit) {

                if (image.channels() == 1) {
                    image.at<uchar>(y, x) = 255;
                } else if (image.channels() == 3) {
                    for (int c = 0; c < image.channels(); ++c) {
                        image.at<cv::Vec3b>(y, x)[c] = 255;
                    }
                }
            }
        }
    }
}

// Partie PSNR 

double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cerr << "Les dimensions ou les types d'images ne correspondent pas." << std::endl;
        return -1.0;
    }

    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);  


    diff = diff.mul(diff);

    double mse = cv::mean(diff)[0];  
    double psnr = 0.0;

    if (mse > 1e-10) {
        psnr = 10.0 * log10(255 * 255 / mse);
    } else {
        std::cerr << "Les images sont identiques, la PSNR est infinie." << std::endl;
    }

    return psnr;
}

// Partie SSIM 

double calculateMean(const cv::Mat& image, const cv::Rect& region) {
    cv::Scalar mean = cv::mean(image(region));
    return mean[0];
}


double calculateVariance(const cv::Mat& image, const cv::Rect& region, double mean) {
    cv::Mat squaredDiff;
    cv::pow(image(region) - mean, 2, squaredDiff);
    cv::Scalar variance = cv::mean(squaredDiff);
    return variance[0];
}


double calculateCovariance(const cv::Mat& image1, const cv::Rect& region1,
                           const cv::Mat& image2, const cv::Rect& region2,
                           double mean1, double mean2) {
    cv::Mat product;
    cv::multiply(image1(region1) - mean1, image2(region2) - mean2, product);
    cv::Scalar covariance = cv::mean(product);
    return covariance[0];
}


double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2, int windowSize, double C1, double C2) {
    const int stride = windowSize / 2;

    double ssimSum = 0.0;
    int numWindows = 0;

    for (int y = 0; y < img1.rows - windowSize + 1; y += stride) {
        for (int x = 0; x < img1.cols - windowSize + 1; x += stride) {
            cv::Rect windowRect(x, y, windowSize, windowSize);

            double mean1 = calculateMean(img1, windowRect);
            double mean2 = calculateMean(img2, windowRect);

            double variance1 = calculateVariance(img1, windowRect, mean1);
            double variance2 = calculateVariance(img2, windowRect, mean2);

            double covariance = calculateCovariance(img1, windowRect, img2, windowRect, mean1, mean2);

            double l = (2 * mean1 * mean2 + C1) / (mean1 * mean1 + mean2 * mean2 + C1);
            double c = (2 * sqrt(variance1) * sqrt(variance2) + C2) / (variance1 + variance2 + C2);
            double s = (covariance + C2 / 2) / (sqrt(variance1) * sqrt(variance2) + C2 / 2);

            double ssim = l * c * s;
            ssimSum += ssim;
            numWindows++;
        }
    }
    double ssim = ssimSum / numWindows;
    return ssim;
}

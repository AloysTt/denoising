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
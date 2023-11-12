#include <lib.h>


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
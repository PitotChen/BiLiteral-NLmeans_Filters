#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

using namespace cv;
using namespace std;

mutex coutMutex;

float calculatePatchDistance(const Mat& patch1, const Mat& patch2) {
    Scalar diffSum = sum(patch1 - patch2);
    return diffSum[0] * diffSum[0] + diffSum[1] * diffSum[1] + diffSum[2] * diffSum[2];
}

void processBlock(const Mat& src, Mat& dst, int startRow, int endRow, int templateWindowSize, int searchWindowSize, float h, int threadId, vector<int>& progress) {
    const int height = src.rows;
    const int width = src.cols;
    const int halfTemplateWindow = templateWindowSize / 2;
    const int halfSearchWindow = searchWindowSize / 2;

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < width; ++j) {
            if ((i + templateWindowSize > height) || (j + templateWindowSize > width)) {
                continue;
            }
            

            Mat weightedSumB = Mat::zeros(templateWindowSize, templateWindowSize, src.type());
            Mat weightedSumG = Mat::zeros(templateWindowSize, templateWindowSize, src.type());
            Mat weightedSumR = Mat::zeros(templateWindowSize, templateWindowSize, src.type());

            float totalWeight = 0.0;
            

            for (int y = max(i - halfSearchWindow, 0); y <= min(i + halfSearchWindow, height - templateWindowSize); ++y) {
                for (int x = max(j - halfSearchWindow, 0); x <= min(j + halfSearchWindow, width - templateWindowSize); ++x) {
                    Mat searchPatch = src(Rect(x, y, templateWindowSize, templateWindowSize));
                    float distance = calculatePatchDistance(src(Rect(j, i, templateWindowSize, templateWindowSize)), searchPatch);
                    float weight = exp(-distance / (h * h));
                    Vec3f centerPixel = src.at<Vec3f>(y + halfTemplateWindow, x + halfTemplateWindow);

                    float centerPixelB = centerPixel[0]; 
                    float centerPixelG = centerPixel[1]; 
                    float centerPixelR = centerPixel[2];
                    // weightedSumB += searchPatch * weight;
                    // weightedSumG += searchPatch * weight;
                    // weightedSumR += searchPatch * weight;
                    weightedSumB += centerPixelB * weight;
                    weightedSumG += centerPixelG * weight;
                    weightedSumR += centerPixelR * weight;
                    totalWeight += weight;
                }
            }


            if (totalWeight > 0) {
                Vec3f weightedAverage = Vec3f(
                    sum(weightedSumB)[0] / totalWeight, 
                    sum(weightedSumG)[0] / totalWeight, 
                    sum(weightedSumR)[0] / totalWeight);
                dst.at<Vec3f>(i, j) = weightedAverage;
            }
        }
        // 更新进度
        progress[threadId] = ((i - startRow) + 1) * 100 / (endRow - startRow);
    }
}


// void processBlock(const Mat& src, Mat& dst, int startRow, int endRow, int templateWindowSize, int searchWindowSize, float h, int threadId, vector<int>& progress) {
//     const int height = src.rows;
//     const int width = src.cols;
//     const int halfTemplateWindow = templateWindowSize / 2;
//     const int halfSearchWindow = searchWindowSize / 2;

//     for (int i = startRow; i < endRow; ++i) {
//         for (int j = 0; j < width; ++j) {
//          
//             if ((i + templateWindowSize > height) || (j + templateWindowSize > width)) {
//                 continue; 
//             }
//             Mat patchCenter = src(Rect(j, i, templateWindowSize, templateWindowSize));
//             if (patchCenter.rows != templateWindowSize || patchCenter.cols != templateWindowSize) {
//                 continue; 
//             }

//             Mat weightedSum = Mat::zeros(templateWindowSize, templateWindowSize, src.type());

//             float totalWeight = 0.0;
//             for (int y = max(i - halfSearchWindow, 0); y <= min(i + halfSearchWindow, height - templateWindowSize); ++y) {
//                 for (int x = max(j - halfSearchWindow, 0); x <= min(j + halfSearchWindow, width - templateWindowSize); ++x) {
//                     Mat searchPatch = src(Rect(x, y, templateWindowSize, templateWindowSize));
//                     float distance = calculatePatchDistance(patchCenter, searchPatch);
//                     float weight = exp(-distance / (h * h));
//                     weightedSum += searchPatch * weight;
//                     totalWeight += weight;
//                 }
//             }

//             if (totalWeight > 0) {
//                 Vec3f weightedAverage = sum(weightedSum)[0] / totalWeight;
//                 dst.at<Vec3f>(i, j) = weightedAverage;
//             }

//             //dst.at<Vec3f>(i, j) = weightedAverage;
//         }
//         progress[threadId] = ((i - startRow) + 1) * 100 / (endRow - startRow);
//     }
// }

void nlMeansDenoisingCustom(const Mat& src, Mat& dst, int templateWindowSize, int searchWindowSize, float h, int num_threads) {
    CV_Assert(src.depth() == CV_32F);
    dst = Mat::zeros(src.size(), src.type());

    vector<thread> threads;
    vector<int> progress(num_threads, 0); // Progress for each thread
    int rowsPerThread = src.rows / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i + 1) * rowsPerThread;
        if (i == num_threads - 1) {
            endRow = src.rows;
        }
        threads.emplace_back(processBlock, cref(src), ref(dst), startRow, endRow, templateWindowSize, searchWindowSize, h, i, ref(progress));
    }

    // Monitor progress
    bool processing = true;
    while (processing) {
        lock_guard<mutex> lock(coutMutex);
        processing = false;
        for (int i = 0; i < num_threads; ++i) {
            cout << "Thread " << i << ": " << progress[i] << "% ";
            if (progress[i] < 100) {
                processing = true;
            }
        }
        cout << "\r" << flush;
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    cout << endl;

    for (auto& t : threads) {
        t.join();
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <num_threads> <image_path>" << endl;
        return 1;
    }

    int num_threads = stoi(argv[1]);
    string image_path = argv[2];

    Mat src = imread(image_path, IMREAD_COLOR);
    if (src.empty()) {
        cerr << "Could not open or find the image" << endl;
        return 1;
    }

    Mat srcFloat;
    src.convertTo(srcFloat, CV_32FC3, 1.0 / 255.0);

    Mat dst;
    nlMeansDenoisingCustom(srcFloat, dst, 7, 21, 0.2f, num_threads);

    Mat dst8U;
    dst.convertTo(dst8U, CV_8UC3, 255.0);

    string output_path = "denoised_image.jpg";
    imwrite(output_path, dst8U);
    cout << "Denoised image saved to " << output_path << endl;

    return 0;
}

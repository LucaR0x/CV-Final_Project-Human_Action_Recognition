/**
AUTHOR: ROSSETTO LUCA
*/

#include "tracker.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/video.hpp>
#include <cmath>

Tracker::Tracker() {
    bg_subtractor = cv::createBackgroundSubtractorKNN(200, 400.0, false);
    first_frame = true;
    bg_initialized = false;
}

void Tracker::init(const std::vector<cv::Mat>& sequence_frames) {
    if (sequence_frames.empty()) return;

    int rows = sequence_frames[0].rows;
    int cols = sequence_frames[0].cols;
    int num_frames = (int)sequence_frames.size();

    bg_median = cv::Mat::zeros(rows, cols, CV_8UC1);
    std::vector<uchar> pixel_values(num_frames);

    // Compute pixel-wise median across sequence frames for background estimation
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            for (int i = 0; i < num_frames; ++i) {
                cv::Mat gray_f;
                if (sequence_frames[i].channels() == 3) {
                    cv::cvtColor(sequence_frames[i], gray_f, cv::COLOR_BGR2GRAY);
                } else {
                    gray_f = sequence_frames[i];
                }
                pixel_values[i] = gray_f.at<uchar>(r, c);
            }
            
            for (int i = 0; i < num_frames - 1; ++i) {
                int min_idx = i;
                for (int j = i + 1; j < num_frames; ++j) {
                    if (pixel_values[j] < pixel_values[min_idx]) {
                        min_idx = j;
                    }
                }
                uchar temp = pixel_values[i];
                pixel_values[i] = pixel_values[min_idx];
                pixel_values[min_idx] = temp;
            }
            
            bg_median.at<uchar>(r, c) = pixel_values[num_frames / 2];
        }
    }

    bg_initialized = true;
}

cv::Rect Tracker::processFrame(const cv::Mat& frame, cv::Mat& out_mask) {
    cv::Mat gray_img;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray_img, cv::COLOR_BGR2GRAY);
    } else {
        gray_img = frame.clone();
    }

    int img_w = gray_img.cols;
    int img_h = gray_img.rows;

    cv::Mat fg_mask;
    if (bg_initialized && !bg_median.empty()) {
        cv::Mat diff_img;
        cv::absdiff(gray_img, bg_median, diff_img);
        cv::GaussianBlur(diff_img, diff_img, cv::Size(5, 5), 1.0);
        cv::threshold(diff_img, fg_mask, 18, 255, cv::THRESH_BINARY);

        // Threshold fallback for low-contrast frames
        if (cv::countNonZero(fg_mask) < 35) {
            cv::threshold(diff_img, fg_mask, 10, 255, cv::THRESH_BINARY);
        }
    } else {
        cv::Mat fg_knn;
        double lr;
        if (first_frame) {
            lr = 0.5;
        } else {
            lr = 0.0002;
        }
        bg_subtractor->apply(gray_img, fg_knn, lr);
        cv::threshold(fg_knn, fg_mask, 150, 255, cv::THRESH_BINARY);
    }

    // Zero out noise along image border
    cv::rectangle(fg_mask, cv::Rect(0, 0, img_w, img_h), cv::Scalar(0), 6);

    // Vertical morphological closing kernel to join body parts
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 25));
    cv::morphologyEx(fg_mask, out_mask, cv::MORPH_CLOSE, close_kernel);
    cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 9));
    cv::dilate(out_mask, out_mask, dilate_kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(out_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect main_box(0, 0, 0, 0);
    double max_area = 0.0;

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        cv::Rect b = cv::boundingRect(contours[i]);
        if (b.width > img_w * 0.85 || b.height > img_h * 0.85) {
            continue;
        }

        if (area > max_area) {
            max_area = area;
            main_box = b;
        }
    }

    cv::Rect final_box = main_box;
    if (final_box.area() > 0) {
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area < 25.0) {
                continue;
            }
            cv::Rect b = cv::boundingRect(contours[i]);
            if (b.x == main_box.x && b.y == main_box.y && b.width == main_box.width && b.height == main_box.height) {
                continue;
            }
            if (b.width > img_w * 0.85 || b.height > img_h * 0.85) {
                continue;
            }

            int diff_x1 = main_box.x - (b.x + b.width);
            int diff_x2 = b.x - (main_box.x + main_box.width);
            int dx = 0;
            if (diff_x1 > dx) dx = diff_x1;
            if (diff_x2 > dx) dx = diff_x2;

            int diff_y1 = main_box.y - (b.y + b.height);
            int diff_y2 = b.y - (main_box.y + main_box.height);
            int dy = 0;
            if (diff_y1 > dy) dy = diff_y1;
            if (diff_y2 > dy) dy = diff_y2;

            if (dx < 20 && dy < 20) {
                int x1 = (final_box.x < b.x) ? final_box.x : b.x;
                int y1 = (final_box.y < b.y) ? final_box.y : b.y;
                int x2 = (final_box.x + final_box.width > b.x + b.width) ? final_box.x + final_box.width : b.x + b.width;
                int y2 = (final_box.y + final_box.height > b.y + b.height) ? final_box.y + final_box.height : b.y + b.height;
                final_box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            }
        }
    }

    // Adjust vertical aspect ratio for human physical proportions
    if (final_box.area() > 0) {
        float aspect_ratio = (float)final_box.height / ((float)final_box.width + 1e-4f);
        if (aspect_ratio < 1.4f) {
            int target_h = (int)(final_box.width * 2.0f);
            int max_allowed_h = (int)(img_h * 0.65f);
            if (target_h > max_allowed_h) {
                target_h = max_allowed_h;
            }

            if (target_h > final_box.height) {
                int pad_y = (target_h - final_box.height) / 2;
                final_box.y = final_box.y - pad_y;
                if (final_box.y < 0) {
                    final_box.y = 0;
                }

                int remaining_h = img_h - final_box.y;
                if (target_h < remaining_h) {
                    final_box.height = target_h;
                } else {
                    final_box.height = remaining_h;
                }
            }
        }
    }

    first_frame = false;
    return final_box;
}
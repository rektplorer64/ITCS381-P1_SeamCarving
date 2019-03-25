#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"

#define SEAM_COLOR 255
#define WINDOW_NAME "EXAMPLE"
#define WINDOW_NAME_M_MATRIX "Calculated M Matrix"
#define WINDOW_NAME_SEAM "Calculated Best Seam"

#define MODE_VERTICAL 0
#define MODE_HORIZONTAL 1

#define VERBOSE true
#define VERBOSE_SEAM false
#define VERBOSE_VALUE_TEST false

using namespace cv;
using namespace std;

/**
 * Determines the Type of a OpenCV Matrix
 * @param mat input Matrix
 */
void matrixType(Mat mat){
    int type = mat.type();                      // Get Type Integer of the Matrix
    string r;                                   // Main String

    uchar depth = type & CV_MAT_DEPTH_MASK;     //
    uchar channel = 1 + (type >> CV_CN_SHIFT);

    switch(depth){
        case CV_8U:
            r = "8U";
            break;
        case CV_8S:
            r = "8S";
            break;
        case CV_16U:
            r = "16U";
            break;
        case CV_16S:
            r = "16S";
            break;
        case CV_32S:
            r = "32S";
            break;
        case CV_32F:
            r = "32F";
            break;
        case CV_64F:
            r = "64F";
            break;
        default:
            r = "User";
            break;
    }

    r += "C";
    r += (channel + '0');

    printf("Matrix: %s %dx%d \n", r.c_str(), mat.cols, mat.rows);
}

/**
 * Manipulates the Matrix to make it padded with duplicates.
 * @param rawInput Input Image
 * @return Duplication padded Image
 */
Mat duplicateWithPadding(Mat rawInput){
    Mat input = rawInput;
    cvtColor(rawInput, input, COLOR_BGR2GRAY);      // Input Image is converted to GREYSCALE (CV_8UC1)

    Size sz = input.size();                         // Determines the Size of the Image
    int height = sz.height;                         // Get Height
    int width = sz.width;                           // Get Width

    Mat output(height + 2, width + 2, CV_8UC1);     // Create the output Matrix with 2 extra pixels in both axis

    Size outSize = output.size();                   // Determines the Size of the New Matrix
    const int outHeight = outSize.height;           // Get Height
    const int outWidth = outSize.width;             // Get Width

    // Print Verbose
    cout << height << " " << width << "\n";
    cout << outHeight << " " << outWidth << "\n";

    int PADDING_SIZE = 1;
    copyMakeBorder(input, output, PADDING_SIZE, PADDING_SIZE, PADDING_SIZE,
                   PADDING_SIZE,
                   BORDER_REPLICATE);               // Use OpenCV Function to create Image with Padding Size = 1
    /*
    for(int i = 0; i < outHeight; i++){
        int k;
        if(i == 0 || i == 1){
            k = 0;
        }else if(i == outHeight - 1 || i == outHeight - 2){
            k = height - 1;
        }else{
            k = i - 1;
        }
        // Vec3b* inputPixelRow = input.ptr<Vec3b>(k);
        // Vec3b* outputPixelRow = output.ptr<Vec3b>(i);

        for(int j = 0; j < outWidth; j++){
            if(j == 0 || j == 1){  // If now it is either the first or second pixel
                // of the output
                output.at<uchar>(i, j) = input.at<uchar>(k, 0);  // Use the first pixel
            }else if(j == outWidth - 1 ||
                     j == outWidth - 2){  // If now it is either the second last or
                // the last pixel of the output
                // outputPixelRow[j] = inputPixelRow[width - 1];	// Use the last
                // pixel
                output.at<uchar>(i, j) = input.at<uchar>(k, width - 1);
            }else{
                // outputPixelRow[j] = inputPixelRow[j - 1];
                output.at<uchar>(i, j) = input.at<uchar>(k, j - 1);
            }

            // cout << setfill('0') << setw(3) << (int)output.at<uchar>(i, j) << " ";
            // int sum = R + G + B;
            // cout << setfill('0') << setw(3) << sum << " ";
            // cout << "Color(" << i << "," << j << "): " << R << " " << G << " " << B
            // << "\n";
        }
        // cout << "\n";
    }
    */
    // imwrite("outpadding.jpg", output);
    return output;
}

/**
 * Finds the Minimum Value in Matrix M
 * @param input Integer Array of Cu, Cl and Cr
 * @param outputIndexOfMin The index of Minimum Value
 * @param outputMin The Minimum Value
 */
void findMinMatrixM(int *input, int *outputIndexOfMin, int *outputMin){
    int minValue = 999;                 // Minimum Value
    int minIndex = -1;                  // Index of the Minimum Value
    for(int i = 0; i < 3; i++){         // Iterates through the Array
        if(*(input + i) < minValue){    // If the Value in the array is lower than current Min Value
            minValue = *(input + i);    // Make Min Value Stores the array value
            minIndex = i;               // Store the index
        }
    }
    *outputIndexOfMin = minIndex + 1;   // Add offset to the index
    *outputMin = minValue;              // Assign the Value to the Input Param
}

/**
 * Finds the Best Seam from the calculated M and K matrices
 * @param MatrixM Calculated Energy M Matrix
 * @param MatrixK Position for Selected Energy position
 * @param input_MatrixSeam Target Output Seam Matrix
 */
void findBestSeam(Mat MatrixM, Mat MatrixK, Mat &input_MatrixSeam){
    Size sz = MatrixM.size();           // Get Matrix M Size
    int height = sz.height;
    int width = sz.width;

    Mat MatrixSeam(height, width, CV_8UC1, Scalar(0));      // Create a New Matrix for Seam filled with Black color

    // Find Minimum M value in The Last Row in Padded Duplicate Image
    int minimumM = 99999999999;
    int minimumM_index = -1;
    for(int j = 0; j < width; j++){
        int current = (int) MatrixM.at<ushort>(height - 1, j);       // Get Minimum M Value in the last row column by column
        if(current < minimumM){             // If the current pixel has lower M val
            minimumM = current;             // Keep it
            minimumM_index = j;             // Remember its Position
        }
    }

    // We mark that Pixel in the Last Row with Predefined Seam Color (White; 255)
    MatrixSeam.at<uchar>(height - 1, minimumM_index) = (uchar) SEAM_COLOR;

    int previousCol = minimumM_index;                           // Initialize the Previous Column with minimumM_Index
    int currentCol = 0;                                         // Init Current Col at the first col
    int direction = 0;                                          // Init the direction as 0
    for(int i = height - 1; i > 0; i--){                        // Iterates from the last to the first row
        direction = (int) MatrixK.at<uchar>(i, previousCol);    // Get the Direction from K Matrix
        switch(direction){                                      // Test the K value
            case 1:
                currentCol = previousCol - 1;                   // If the direction is 1, column offset = -1
                break;
            case 2:
                currentCol = previousCol;                       // If the direction is 2, column offset = 0
                break;
            case 3:
                currentCol = previousCol + 1;                   // If the direction is 3, column offset = 1
                break;
            default:
                break;
        }

        MatrixSeam.at<uchar>(i, currentCol) = (uchar) SEAM_COLOR;   // Make that Pixel the Seam
        if(i == 1){                                                 // If it is the first row, make it a seam
            MatrixSeam.at<uchar>(i - 1, currentCol) = (uchar) SEAM_COLOR;
        }
        previousCol = currentCol;                                   // Change to the previous column
    }

    input_MatrixSeam = MatrixSeam;                                  // Assign the result to input_MatrixSeam
}

/**
 * Calculates Matrix M and Matrix K for Seam Carving
 * @param paddedImage Duplication padded Input Image for Seam Carving
 * @param MatrixM Calculated Energy M Matrix
 * @param MatrixK Position for Selected Energy position
 */
void calculateMatrixM(Mat paddedImage, Mat &MatrixM, Mat &MatrixK){
    Size sz = paddedImage.size();                       // Get the size of Padded Input Image
    int height = sz.height;
    int width = sz.width;

    Mat M_MATRIX(height - 2, width - 2, CV_16UC1);       // Create Empty Matrix for M values
    Mat K_MATRIX(height - 2, width - 2, CV_8UC1);       // Create Empty Matrix for K values

    // ofstream cout("output.txt");

    for(int i = 1; i < height - 1; i++){                                        // Iterates through rows
        for(int j = 1; j < width - 1; j++){                                     // Iterates through cols
            int C_UP = abs((int) paddedImage.at<uchar>(i, j + 1) -
                           (int) paddedImage.at<uchar>(i, j - 1));              // Calculate the Cu Value

            // cout << "R: " << i << ", C: " << j << " | Cl = " << C_LT << " , Cu = "
            // << C_UP << ", Cr = " << C_RT << "\n"; M_MATRIX.at<float>(i, j) =
            // *min_element(cValues, cValues + 4);

            if(i == 1){                                                         // If it is the first ROW
                M_MATRIX.at<ushort>(i - 1, j - 1) = (ushort) C_UP;                // Only care about C UPPER only
                K_MATRIX.at<uchar>(i - 1, j - 1) = (uchar) 0;                   // All K values in the Top Row are ZERO
            }else{
                int C_LT = C_UP + abs((int) paddedImage.at<uchar>(i - 1, j) -
                                      (int) paddedImage.at<uchar>(i, j - 1));   // Calculate the Cl Value
                int C_RT = C_UP + abs((int) paddedImage.at<uchar>(i - 1, j) -
                                      (int) paddedImage.at<uchar>(i, j + 1));   // Calculate the Cr Value

                int min, index;                                                 // Integers for Min and Index Value
                int left, middle, right;                                        // Integers for Left (1), Middle (2) and Right (3) Position
                int cValues[3];
                if(j == 1){                 // First Row
                    left = 9999999;                                                         // We don't have the Left.
                    right = ((int) M_MATRIX.at<ushort>((i - 1) - 1, (j + 1) - 1)) + C_RT;    // We calculate the Right.
                }else if(j == width - 2){   // Last Row
                    left = ((int) M_MATRIX.at<ushort>((i - 1) - 1, (j - 1) - 1)) + C_LT;     // We calculate the Left.
                    right = 9999999;                                                        // We don't have the Right.
                }else{
                    left = ((int) M_MATRIX.at<ushort>((i - 1) - 1, (j - 1) - 1)) + C_LT;     // We calculate the Left.
                    right = ((int) M_MATRIX.at<ushort>((i - 1) - 1, (j + 1) - 1)) + C_RT;    // We calculate the Right.
                }
                middle = ((int) M_MATRIX.at<ushort>((i - 1) - 1, j - 1)) + C_UP;             // We always calculate the Middle.

                // We Assign these value to find Minimum.
                cValues[0] = left;
                cValues[1] = middle;
                cValues[2] = right;

                //cout << "l: " << left << ", m: " << middle << ", r: " << right;
                findMinMatrixM(cValues, &index, &min);                                      // Find the Matrix M
                //cout << "; min: " << min << ", dir: " << index << "\n";

                M_MATRIX.at<ushort>(i - 1, j - 1) = (ushort) min;                             // Store M value to M Matrix
                K_MATRIX.at<uchar>(i - 1, j - 1) = (uchar) index;                           // Store K value to K Matrix

                // cout << setfill(' ') << setw(3) << (int)K_MATRIX.at<uchar>(i - 1, j - 1) << " ";
            }
        }
    }

    //cout << "M = " << endl << " " << M_MATRIX << endl << endl;

    // Assigns the M and K value to given M and K Parameters
    MatrixM = M_MATRIX;
    MatrixK = K_MATRIX;

    if(VERBOSE_VALUE_TEST){
        cout << "M matrix = " << endl << " " << M_MATRIX << endl << endl;
        cout << "K matrix = " << endl << " " << K_MATRIX << endl << endl;
    }

}

/**
 * Inserts or Deletes Seam depends on the boolean parameter
 * @param img Input image Matrix
 * @param seam Seam Matrix
 * @param outputImg Output Matrix
 * @param isInserting Boolean to determine whether to insert or delete
 */
void insertOrDeleteSeam(Mat img, Mat seam, Mat &outputImg, bool isInserting){
    Mat DEBUG_SEAM_MAT, DEBUG_SEAM_COLOR;
    if(VERBOSE_SEAM){
        DEBUG_SEAM_MAT = Mat(outputImg.rows, outputImg.cols, CV_8UC3, Scalar(0, 0, 0));
        DEBUG_SEAM_COLOR = Mat(1, 1, CV_8UC3, Scalar(0, 0, 255));
    }

    int seamPixel;
    for(int i = 0; i < outputImg.rows; i++){                                // Iterates through Rows
        int offset = 0;                                                     // For every row, Reset the Offset
        for(int j = 0; j < outputImg.cols; j++){                            // Iterates through Columns
            seamPixel = (int) seam.at<uchar>(i, j + offset);                // Get Seam Pixel from Seam Matrix
            /*outputImgPixel = (int) outputImg.at<uchar>(i, j);*/
            if(seamPixel == SEAM_COLOR){                                    // If the Pixel has the Seam Color
                if(isInserting){                                            // If we are INSERTING SEAMS
                    if(VERBOSE_SEAM){
                        DEBUG_SEAM_MAT.at<Vec3b>(i, j) = DEBUG_SEAM_COLOR.at<Vec3b>(0, 0);
                        DEBUG_SEAM_MAT.at<Vec3b>(i, j + 1) = img.at<Vec3b>(i, j);
                    }
                    outputImg.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);        // Copy the Img Pixel (at i and j) to Output (at i and j)
                    outputImg.at<Vec3b>(i, j + 1) = img.at<Vec3b>(i, j);    // We also copy the Img Pixel (at i and j) to Output (at i and j + 1).
                    j++;                                                    // We increase j by 1.
                    offset = -1;                                            // Set the offset to -1
                }else{                                                      // If we are DELETING SEAMS
                    if(VERBOSE_SEAM){
                        DEBUG_SEAM_MAT.at<Vec3b>(i, j) = img.at<Vec3b>(i, j + 1);
                    }
                    outputImg.at<Vec3b>(i, j) = img.at<Vec3b>(i, j + 1);    // Copy the Img Pixel (at i and j) to Output (at i and j).
                    offset = 1;                                             // Set the offset to 1
                }
            }else{                                                          // If it doesn't have SEAM color
                if(VERBOSE_SEAM){
                    DEBUG_SEAM_MAT.at<Vec3b>(i, j) = img.at<Vec3b>(i, j + offset);
                }
                outputImg.at<Vec3b>(i, j) = img.at<Vec3b>(i, j + offset);   // Copy the Pixel from img with offset
            }
        }
    }

    if(VERBOSE_SEAM){
        imshow("IMG with Seam", DEBUG_SEAM_MAT);
    }
}

// Forward Seam Carving
int main(){
    setBreakOnError(true);

    Mat img = imread("test2.jpg", IMREAD_COLOR);

    // DEBUG: Check for Matrix Type
    matrixType(img);

    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
    imshow(WINDOW_NAME, img);

    int c = cvWaitKey(0);

    // std::cin.ignore();

    Size sz = img.size();
    int height = sz.height;
    int width = sz.width;

    // ESC = 27, a = 97, d = 100, s = 115, w = 119
    while(c != 27){
        // Looping till get the command 'a', 'b', 'c', 'w', 's'
        while(c != 97 && c != 100 && c != 115 && c != 119 && c != 27){
            c = cvWaitKey(0);
        }

        // cout << "Height: " << height << ", Width " << width << "\n";

        // Keyboard Command :: 'a', 'd' --> Vertical, 'w', 's' --> Horizontal
        // 'a' --> Reduce width, 'd' --> Increase width
        Mat mMatrix, kMatrix, seamMatrix;
        int MODE;
        if(c == 97 || c == 100){    // If the Key is 'a' or 'd'
            MODE = MODE_VERTICAL;
            // Construct M matrix and K Matrix in the VERTICAL direction
            cout << "Calculating M Matrix\n";
            calculateMatrixM(duplicateWithPadding(img), mMatrix, kMatrix);

            // Find the Best Seam in VERTICAL direction
            cout << "Finding the Best Seam\n";
            findBestSeam(mMatrix, kMatrix, seamMatrix);

            Mat img_new;
            if(c == 97){            // Key = 'a'
                if(width - 1 > 0){
                    // Reduce width or delete seam VERTICALLY
                    // Copy the pixels into this 
                    img_new = Mat(height, --width, CV_8UC3, Scalar(0, 0, 0));
                    insertOrDeleteSeam(img, seamMatrix, img_new, false);
                }
            }else if(c == 100){      // Key = 'd'
                // Increase width or Insert seam VERTICALLY
                // Copy the pixels into this image
                img_new = Mat(height, ++width, CV_8UC3, Scalar(0, 0, 0));

                cout << "Inserting Seam Vertically\n";
                insertOrDeleteSeam(img, seamMatrix, img_new, true);
            }

            // Show the resized image
            imshow(WINDOW_NAME, img_new);

            if(VERBOSE == 1){
                imshow("rawM_mat ", mMatrix);
                imshow(WINDOW_NAME_SEAM, seamMatrix);
                // cvWaitKey(0);
            }

            // Clone img_new into img for next loop processing
            img.release();
            img = img_new.clone();
            img_new.release();
        }else if(c == 115 || c == 119){    // If the Key is 's' or 'w'
            MODE = MODE_HORIZONTAL;

            Mat rotatedImg;
            transpose(img, rotatedImg);

            // Construct M matrix and K Matrix in the HORIZONTAL direction
            cout << "Calculating M Matrix\n";
            calculateMatrixM(duplicateWithPadding(rotatedImg), mMatrix, kMatrix);

            // Find the Best Seam in HORIZONTAL direction
            cout << "Finding the Best Seam\n";
            findBestSeam(mMatrix, kMatrix, seamMatrix);

            Mat img_new;
            if(c == 115){   // Key = 's' 
                if(height - 1 > 0){
                    // Reduce width or delete seam HORIZONTALLY
                    // Copy the pixels into this image
                    img_new = Mat(width, --height, CV_8UC3, Scalar(0, 0, 0));

                    insertOrDeleteSeam(rotatedImg, seamMatrix, img_new, false);

                    Mat originalMat = img_new.clone();
                    transpose(originalMat, img_new);
                }
            }else if(c == 119){  // Key = 'w'
                // Increase width or Insert seam HORIZONTALLY
                // Copy the pixels into this image
                img_new = Mat(width, ++height, CV_8UC3, Scalar(0, 0, 0));

                // Insert The Seam by using the 90 degree Rotated Image
                insertOrDeleteSeam(rotatedImg, seamMatrix, img_new, true);

                Mat originalMat = img_new.clone();
                transpose(originalMat, img_new);
            }

            // Show the resized image
            imshow(WINDOW_NAME, img_new);

            if(VERBOSE == 1){
                transpose(mMatrix, mMatrix);
                imshow("mMat", mMatrix);

                transpose(seamMatrix, seamMatrix);
                imshow(WINDOW_NAME_SEAM, seamMatrix);
                // cvWaitKey(0);
            }

            // Clone img_new into img for next loop processing
            img.release();
            img = img_new.clone();
            img_new.release();
        }

        if(c == 27){
            break;
        }
        c = cvWaitKey(0);
    }
    return 0;

}
#include <iostream>
#include <string>  
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "edge_detect.hh"
#include "non_local_means_cpu.hh"
#include "knn.hh"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("usage: main <Image_Path> <Func_name> <Func_Params...>\n");
        return 1;
    }
    string func_name = argv[2];
    if (func_name == "conv" and argc < 4)
    {
        printf("usage: main <Image_Path> conv <Conv_size>\n");
        return 1;
    }
    if (func_name == "knn" and argc < 5)
    {
        printf("usage: main <Image_Path> <Func_name> <Conv_size> <Weight_Decay_Param>\n");
        return 1;
    }
    if (func_name == "nlm" and argc < 6)
    {
        printf("usage: main <Image_Path> nlm <Conv_size> <Block_radius> <Weight_Decay_Param>\n");
        return 1;
    }
    Mat image;
    image;
    if (func_name == "edge_detect")
        image = imread(argv[1], 0);
    else
        image = imread(argv[1], cv::IMREAD_UNCHANGED);
    if (!image.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return 1;
    }
    Mat res;
    
    if (func_name == "nlm")
        res = non_local_means_cpu(image, stoi(argv[3]), stoi(argv[4]), stod(argv[5]));
    else if (func_name == "knn")
        res = knn(image, stoi(argv[3]), stof(argv[4]));
    else if (func_name == "conv")
        res = convolution(image, stoi(argv[3]));
    else if (func_name == "edge_detect")
    {
        res = knn_grey(image, 2, 150.0);
        res = conv_with_mask(res, 1);
    }
    else if (func_name == "pixelize")
    {
        res = pixelize(image, stoi(argv[3]));
    }
    // namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    // imshow("Display Window", res);
    // waitKey(0);
    
    // Construct output filename in output/ directory
    string output_filename = "output/cpu_";
    output_filename += func_name;
    output_filename += "_result.jpg";
    
    imwrite(output_filename, res);
    cout << "Saved CPU output to " << output_filename << endl;
    return 0;
}

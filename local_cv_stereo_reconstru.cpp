//cv_stereo_reconstru.cpp
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <stdio.h>
#include<iostream>//标准输入输出
//命名空间
using namespace cv;
using namespace std;
//将点云数据保存到本地
static void saveXYZ(const char* file_name, const Mat& mat_xyz)
{
    const double max_z_value = 1.0e4;
    FILE* fp_xyz = fopen(file_name, "wt");
    for(int xyz_rows = 0; xyz_rows < mat_xyz.rows; xyz_rows++)//可以传入roi参数　减少遍历的像素点
    {
        for(int xyz_cols = 0; xyz_cols < mat_xyz.cols; xyz_cols++)
        {
            Vec3f xyz_point = mat_xyz.at<Vec3f>(xyz_rows, xyz_cols);
            if(fabs(xyz_point[2] - max_z_value) < FLT_EPSILON || fabs(xyz_point[2]) > max_z_value) continue;
            fprintf(fp_xyz, "%f %f %f\n", xyz_point[0], xyz_point[1], xyz_point[2]);
        }
    }
    fclose(fp_xyz);
}
int main(int argc, char** argv)
{
    std::string filename_img_l = "";
    std::string filename_img_r = "";
    std::string name_intrin_file = "";
    std::string name_extrin_file = "";
    std::string File_name_disparity = "";
    std::string File_name_point_cloud = "";
    std::string File_name_point_cloud_ch = "point_cloud_file_ch.txt";
    std::string File_name_disparity_ch="disparity_image_ch.png";
    std::string File_name_point_cloud_local="poin_cloud_xyz_local.txt";//经过形态学处理的点云文本
    enum { method_BM=0, method_SGBM=1, method_HH=2, method_VAR=3, method_3WAY=4 };
    int method_alg = method_SGBM;//选择了SGBM算法
    int stereo_sad_window_size, stereo_num_of_disparities;
    bool display_no;
    float method_stereo_scal;
    int i_1=16,i_2=9,j_3=0,j_4=16,j_5=3;
    Ptr<StereoBM> stereo_bm = StereoBM::create(i_1,i_2);
    Ptr<StereoSGBM> stereo_sgbm = StereoSGBM::create(j_3,j_4,j_5);
    cv::CommandLineParser parser(argc, argv,
        "{@arg1|left7.jpg|}{@arg2|right7.jpg|}{help h||}{method_algorithm|stereo_sgbm|}{method_max_disparity|80|}{method_block_size|7|}{method_no_display||}{method_stereo_scal|1|}{i|intrinsics_file.yml|}{e|extrinsics_file.yml|}{o|method_disparity_image.png|}{p|file_cloud_point.txt|}");
    if(parser.has("help"))
    {
        return 0;
    }
    filename_img_l = parser.get<std::string>(0);
    //打印左图文件名称
    cout<< filename_img_l << endl;
    filename_img_r = parser.get<std::string>(1);
    //打印右图文件名称
    cout<< filename_img_r << endl;
    if (parser.has("method_algorithm"))
    {
        std::string _method_alg = parser.get<std::string>("method_algorithm");
        method_alg = _method_alg == "stereo_sgbm" ? method_SGBM :-1;
        cout<<_method_alg<<endl;//打印调试　选取的方法
    }
    stereo_num_of_disparities = parser.get<int>("method_max_disparity");
    stereo_sad_window_size = parser.get<int>("method_block_size");
    method_stereo_scal = parser.get<float>("method_stereo_scal");
    display_no = parser.has("method_no_display");
    if( parser.has("i") )
        name_intrin_file = parser.get<std::string>("i");
    if( parser.has("e") )
        name_extrin_file = parser.get<std::string>("e");
    if( parser.has("o") )
        File_name_disparity = parser.get<std::string>("o");
    if( parser.has("p") )
        File_name_point_cloud = parser.get<std::string>("p");
    int mode_color_method = method_alg == method_BM ? 0 : -1;
    Mat left_img = imread(filename_img_l, mode_color_method);
    Mat right_img = imread(filename_img_r, mode_color_method);
    Mat left_local ,right_local;//定义变量，局部图像
    if (left_img.empty())
    {
        printf("command_line param_error: could not load left_img\n");
        return -1;
    }
    if (right_img.empty())
    {
        printf("command_line param_error: could not load right_img\n");
        return -1;
    }
    if (method_stereo_scal != 1.f)
    {
        Mat scal_temp_1, scal_temp_2;
        int scal_method = method_stereo_scal < 1 ? INTER_AREA : INTER_CUBIC;
        resize(left_img, scal_temp_1, Size(), method_stereo_scal, method_stereo_scal, scal_method);
        left_img = scal_temp_1;
        resize(right_img, scal_temp_2, Size(), method_stereo_scal, method_stereo_scal, scal_method);
        right_img = scal_temp_2;
    }
    Size size_image = left_img.size();
    Rect method_roi_1, method_roi_2;
    Mat remp_Q;
    if( !name_intrin_file.empty() )
    {
        // reading intrinsic parameters
        FileStorage fs(name_intrin_file, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("can't open file %s\n", name_intrin_file.c_str());
            return -1;
        }
        Mat intrin_M1, intrin_D1, intrin_M2, intrin_D2;
        fs["intrin_M1"] >> intrin_M1;
        fs["intrin_D1"] >> intrin_D1;
        fs["intrin_M2"] >> intrin_M2;
        fs["intrin_D2"] >> intrin_D2;
        intrin_M1 *= method_stereo_scal;
        intrin_M2 *= method_stereo_scal;
        fs.open(name_extrin_file, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("can't open file %s\n", name_extrin_file.c_str());
            return -1;
        }
        Mat Rec,extrin_R, extrin_T, output_R_1, output_P_1, output_R_2, output_P_2;
        fs["extrin_R"] >> extrin_R; //似乎后面会进行Rodrigues变换
        //cout<<Rec<<endl;
        //Rodrigues(Rec, extrin_R); //Rodrigues变换
        //cout<<extrin_R<<endl;
        fs["extrin_T"] >> extrin_T;
        stereoRectify( intrin_M1, intrin_D1, intrin_M2, intrin_D2, size_image, extrin_R, extrin_T, output_R_1, output_R_2, output_P_1, output_P_2, remp_Q, CALIB_ZERO_DISPARITY, -1, size_image, &method_roi_1, &method_roi_2 );
        Mat URMmap11, URMmap12, URMmap21, URMmap22;
        initUndistortRectifyMap(intrin_M1, intrin_D1, output_R_1, output_P_1, size_image, CV_16SC2, URMmap11, URMmap12);
        initUndistortRectifyMap(intrin_M2, intrin_D2, output_R_2, output_P_2, size_image, CV_16SC2, URMmap21, URMmap22);
        Mat left_img_rectify, right_img_rectify;
        remap(left_img, left_img_rectify, URMmap11, URMmap12, INTER_LINEAR);
        remap(right_img, right_img_rectify, URMmap21, URMmap22, INTER_LINEAR);
        left_img = left_img_rectify;
        right_img = right_img_rectify;
    }
    //stereo_num_of_disparities = method_max_disparity=80
    stereo_num_of_disparities = stereo_num_of_disparities > 0 ? stereo_num_of_disparities : ((size_image.width/8) + 15) & -16;// & -16 被１６整除   
    stereo_sgbm->setPreFilterCap(63);
    int sgbm_winsize = stereo_sad_window_size > 0 ? stereo_sad_window_size : 3;
    stereo_sgbm->setBlockSize(sgbm_winsize);
    int _channels = left_img.channels();
    stereo_sgbm->setP1(8*_channels*sgbm_winsize*sgbm_winsize);
    stereo_sgbm->setP2(32*_channels*sgbm_winsize*sgbm_winsize);
    stereo_sgbm->setMinDisparity(0);
    // stereo_num_of_disparities = method_max_disparity=80 在参数设置中就设置了 这几个地方要一致
    stereo_sgbm->setNumDisparities(stereo_num_of_disparities);//80
    stereo_sgbm->setUniquenessRatio(10);
    stereo_sgbm->setSpeckleWindowSize(100);
    stereo_sgbm->setSpeckleRange(32);
    stereo_sgbm->setDisp12MaxDiff(-1);
    if(method_alg==method_SGBM)
        stereo_sgbm->setMode(StereoSGBM::MODE_SGBM);


    Mat disparites, disparites_u8, disparites_u8_ch;//定义变量，视差图
    Mat disparites_local,disparites_local_u8;//定义变量，local 视差
    Mat disparites_morph_loc_u8;//定义变量，形态学处理结果
    Mat disparites_local_morph;//
    Mat poin_cloud_xyz_local_image;//经过形态学处理的深度图


    int64 t_local_0 = getTickCount();
    //CV_16SC1 创建空图
    Mat disp_ch_2=Mat::zeros(left_img.rows,left_img.cols,CV_16SC1);
    //创建roi_rect
    Point pt1=Point(478-stereo_num_of_disparities,178);//478 stereo_num_of_disparities=80
    Point pt2=Point(783,578);
    Rect rect_pikachu=Rect(pt1.x,pt1.y,pt2.x-pt1.x,pt2.y-pt1.y);//定义roi的范围 rect_pikachu  
    //later will create rect_morph_loc_roi by rect_morph_loc 
    Rect rect_morph_loc=Rect(pt1.x+stereo_num_of_disparities,pt1.y,pt2.x-(pt1.x+stereo_num_of_disparities),pt2.y-pt1.y);
    //create rect_morph_loc_roi by rect_morph_loc
    Mat rect_morph_loc_roi=disp_ch_2(rect_morph_loc);//roi
    //根据框的位置，创建左右图的local_roi
    Mat left_local_roi=left_img(rect_pikachu);//roi
    Mat right_local_roi=right_img(rect_pikachu);//roi
    //深拷贝原图local_roi
    left_local=left_local_roi.clone();
    right_local=right_local_roi.clone();
    //计算local视差图，并计时
    int64 t_local = getTickCount();
    if (method_alg == method_SGBM)
        stereo_sgbm->compute(left_local,right_local,disparites_local);
    t_local = getTickCount()-t_local;
    printf("disparites_local time of used: %fms\n", t_local*1000/getTickFrequency());
    // namedWindow("local_disparites"); 
    // imshow("local_disparites",disparites_local);//show the local disparites
    //cut the extra columns
    disparites_local = disparites_local.colRange(80, disparites_local.cols);
    // imshow("local_disparites_cut",disparites_local);
    
    
    //convert the disparites to 8bit visible image
    disparites_local.convertTo(disparites_local_u8, CV_8U, 255/(stereo_num_of_disparities*16.));
    // imshow("local_disparites_u8",disparites_local_u8);
    //haveOpenCL()
    //copy disparites_local_u8
//disparites_local_u8.copyTo(disparites_morph_loc_u8);//将disparites_local_u8 copy to a new instance: disparites_morph_loc_u8
    disparites_morph_loc_u8=disparites_local_u8;//浅拷贝
    //thersh _binary
    threshold(disparites_morph_loc_u8,disparites_morph_loc_u8,200,255,THRESH_BINARY);
    //imshow("disparites_morph_loc_u8_thersh",disparites_morph_loc_u8);
    //define the element
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));  //size 7*7 ?
	//进行形态学操作Morphological manipulation ,Morphological filtering to get the clean disparites_mask
	morphologyEx(disparites_morph_loc_u8, disparites_morph_loc_u8, MORPH_OPEN, element);//open first erode sencond dilate,先腐蚀再膨胀
    //show the result
    // imshow("disparites_morph_loc_u8",disparites_morph_loc_u8);
    //mask to get the new image for show
//disparites_local_u8.copyTo(disparites_morph_loc_u8,disparites_morph_loc_u8);
    // imshow ("disparites_local_u8_smoothing",disparites_morph_loc_u8);

    //disparites_morph_loc_u8 AND operation ,but isn't suit,because that operation will change the origin image,it get the binary image 
    //disparites_morph_loc_u8=(disparites_morph_loc_u8 | disparites_local_u8);
    // bitwise_and(disparites_morph_loc_u8,disparites_local_u8,disparites_morph_loc_u8);
   
    // cout a pixel value of disparites_morph_loc_u8 for checking whether the image is a grey-scale map ,and the results is yes
    // Point xyz_local_u8;
    // xyz_local_u8.x=233;
    // xyz_local_u8.y=233;
    // int gray_value = (int)disparites_morph_loc_u8.at<uchar>(xyz_local_u8);
    // cout << "gray_value = " << gray_value <<endl;


    //mask to get the new image of real disparites
    disparites_local.copyTo(disparites_local_morph,disparites_morph_loc_u8);
    // imshow("disparites_local_morph",disparites_local_morph);

    disparites_local_morph.copyTo(rect_morph_loc_roi);
    // imshow("disp_ch_morph",disp_ch_2);

    reprojectImageTo3D(disp_ch_2, poin_cloud_xyz_local_image, remp_Q, true);
    //saveXYZ(File_name_point_cloud_local.c_str(), poin_cloud_xyz_local_image);
    t_local_0 = getTickCount()-t_local_0;
    printf("all time of used: %fms\n", t_local_0*1000/getTickFrequency());
    









    //计算视差图，并计时
    int64 t = getTickCount();
    if( method_alg == method_SGBM)
        stereo_sgbm->compute(left_img, right_img, disparites);
    t = getTickCount() - t;
    printf("Time of used: %fms\n", t*1000/getTickFrequency());

    
    Mat disp_ch=Mat::zeros(disparites.rows,disparites.cols,disparites.type());//CV_16SC1 创建空图
    //创建某图的roi
    Mat disp_roi=disparites(rect_pikachu);
    Mat disp_roi_zero=disp_ch(rect_pikachu);
    //深拷贝原图roi
    Mat disp_roi_threshold=disp_roi.clone();
    Mat roi_threshold_result;//阈值结果
    threshold(disp_roi_threshold,roi_threshold_result,1000,2000,THRESH_TOZERO);//去除较小值小于1000去除
    imshow("disp_roi_threshold_first",roi_threshold_result);//显示阈值结果
    disp_roi_threshold = roi_threshold_result.clone();//深拷贝
	threshold(disp_roi_threshold, roi_threshold_result, 2000, 255, THRESH_TOZERO_INV);//去除数值较高的，大于2000去除
	imshow("disp_roi_threshold_second", roi_threshold_result);//显示阈值后的效果 */
    imshow("disp_roi",disp_roi);//展示原图的感兴趣区域
    roi_threshold_result.copyTo(disp_roi_zero);//将16位阈值后视差原图的roi拷贝到16位空的视差图相应的Ｒoi中
    namedWindow("目标视差图"); 
    imshow("目标视差图",disp_ch);//显示新的是视差图 ,clear disparites
    imshow("disparites",disparites);//5.26 add
    if( method_alg != method_VAR )
        {
        disparites.convertTo(disparites_u8, CV_8U, 255/(stereo_num_of_disparities*16.));//将原图视差图转换为８位无符号（灰度可见）的形式
        cout<<"stereo_num_of_disparities="<<stereo_num_of_disparities<<endl;//打印最大视差窗口
        disp_ch.convertTo(disparites_u8_ch, CV_8U, 255/(stereo_num_of_disparities*16.));//目标视差图转换为８位无符号（灰度可见）的形式
        
        }
    // cheack match between  disparites_u8_ch and disparites_morph_loc_u8
    // int gray_value_1 = (int)disparites_u8_ch.at<uchar>(233+178,233+478);//478-80,178 233+80+478-80,233+178 // at(rows,cols)
    // cout << "gray_value_1 = "<<gray_value_1 <<endl;

    // cout a pixel value of disparites_u8_ch for checking whether corresponding to  disparites_local_u8 ,and the results is yes
    // Point xyz_local_u8;
    // xyz_local_u8.x=233;//disparites_local_u8.cols/2
    // xyz_local_u8.y=233;//disparites_local_u8.rows/2
    // int gray_value = (int)disparites_local_u8.at<uchar>(xyz_local_u8);
    // cout << gray_value <<endl;
    // cout << disparites_u8_ch.type()<<endl;
    // int gray_value_1 = (int)disparites_u8_ch.at<uchar>(233+178,233+80+478-80);//478-80,178 233+80+478-80,233+178 // at(rows,cols)
    // cout << gray_value_1 <<endl;

    if( !display_no )
    {
        namedWindow("left_image", 1);
        imshow("left_image", left_img);
        imwrite("Rectify_left_pic.png",left_img);
        namedWindow("right_image", 1);
        imshow("right_image", right_img);
        imwrite("Rectify_right_pic.png",right_img);
        namedWindow("disparity", 0);
        namedWindow("disparity_ch", 0);
        imshow("disparity", disparites_u8);
        imshow("disparity_ch", disparites_u8_ch);
        printf("press any key to continue...");
        fflush(stdout);
        waitKey();
        printf("\n");
    }
    if(!File_name_disparity.empty())//如果存在这个文件名　继续
        imwrite(File_name_disparity, disparites_u8);
    if(!File_name_disparity_ch.empty())//如果存在这个文件名　继续
        imwrite(File_name_disparity_ch, disparites_u8_ch);//去除干扰后的可见视差图
    if(!File_name_point_cloud.empty())//如果存在这个文件名　继续
    {
        printf("point cloud is storing...");
        fflush(stdout);
        Mat poin_cloud_xyz;
        Mat poin_cloud_xyz_ch;
        reprojectImageTo3D(disparites, poin_cloud_xyz, remp_Q, true);//poin_cloud_xyz.depth=CV_32F poin_cloud_xyz.type=CV_32FC3
        reprojectImageTo3D(disp_ch, poin_cloud_xyz_ch, remp_Q, true);
        //debug
        cout<<"poin_cloud_xyz.type()="<<poin_cloud_xyz.type()<<endl<<"poin_cloud_xyz.channels()="<<poin_cloud_xyz.channels()<<endl<<"poin_cloud_xyz.depth="<<poin_cloud_xyz.depth()<<endl;//CV_16SC1 16位有符号数单通道
        cout<<"poin_cloud_xyz_ch.type()="<<poin_cloud_xyz_ch.type()<<endl<<"poin_cloud_xyz_ch.channels()="<<poin_cloud_xyz_ch.channels()<<endl<<"poin_cloud_xyz_ch.depth="<<poin_cloud_xyz_ch.depth()<<endl;//CV_16SC1 16位有符号数单通道       
        rectangle(poin_cloud_xyz,rect_pikachu,Scalar(255,0,0));
        imshow("poin_cloud_xyz",poin_cloud_xyz);
        imshow("poin_cloud_xyz_ch",poin_cloud_xyz_ch);
        imwrite("poin_cloud_xyz.png",poin_cloud_xyz);
        imwrite("poin_cloud_xyz_ch.png",poin_cloud_xyz_ch);
        Point xyz_p_1;
        xyz_p_1.x=669;//691(点１) 524(点２) 544(棋盘格) 669(picha_07)
        xyz_p_1.y=366;//334(点１) 140(点２) 206(棋盘格) 366(picha_07)
        cout<<xyz_p_1<<"in disp_ch="<<poin_cloud_xyz_ch.at<Vec3f>(xyz_p_1)*16<<endl;
        Point xyz_p_2;
        xyz_p_2.x=721;// 596(棋盘格) 721(picha_07)
        xyz_p_2.y=363;// 206(棋盘格) 363(picha_07)
        cout<<xyz_p_2<<"in disp_ch="<<poin_cloud_xyz_ch.at<Vec3f>(xyz_p_2)*16<<endl;
        waitKey();
        saveXYZ(File_name_point_cloud.c_str(), poin_cloud_xyz);
        saveXYZ(File_name_point_cloud_ch.c_str(), poin_cloud_xyz_ch);
        printf("\n");
    }

    return 0;
}

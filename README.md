# cv_stereo_constru
#6.2号建
这个工程是用cmake编译的，CMakeLists.txt是配置文件,其中：
`set(OpenCV_DIR /usr/opencv34/share/OpenCV)`　定义了opencv3.4的安装位置，
`add_executable(stereo_piont_cloud new_cv_stereo_reconstru.cpp)``stereo_piont_cloud`：可执行文件的名程　`new_cv_stereo_reconstru.cpp`：源文件的名称

#这个工程是在官方的例程上改的
`new_cv_stereo_reconstru.cpp`　是源文件，`extrinsics_file.yml`和`intrinsics_file.yml`　分别是摄像头的内外参数文件
大概的思路就是，对左右图像进行矫正，双目匹配，获得视差图，根据目标物体在左图像(矫正后)的边框(由目标检测获取，最好是用语义分割的方式来获取物体的准确轮廓，不用后期的滤波），对视差图进行分割，阈值滤波(说实话本文这个方式是死调的，不具备一般性），得到干净的，只含目标物体的视差图，然后计算三维点云．


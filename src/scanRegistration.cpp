// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;  // 10HZ，转一圈的时间是0.1s

const int systemDelay = 0; 
int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;
float cloudCurvature[400000];
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;

double MINIMUM_RANGE = 0.1; 

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    if (&cloud_in != &cloud_out)
    {
        // 两个点云一模一样
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    /**
     * notes: 这里的cloud_in和cloud_out实际上指向的是同一个点云，任何一个的改变都会作用于另一个
     *      这里使用索引来改变点云，达到原地修改的目的；对于大的数据，应该使用原地修改，而不是copy一份，修改好后，然后赋值回去，这样对电脑性能有较高要求
     */
    size_t j = 0;
    // 把点云距离小于给定阈值的去除掉
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        // 这样就将后面的无用数据删除了
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;  // 每列只有一个点云
    cloud_out.width = static_cast<uint32_t>(j);  // 点云数量
    cloud_out.is_dense = true;  // 稠密点云
}

// 订阅lidar消息
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    // 如果系统没有初始化的话，就等几帧
    if (!systemInited)  // 初始化为false
    { 
        systemInitCount++;  // 初始化为0
        if (systemInitCount >= systemDelay)  // systemDelay为常整型，初始化为0
        {
            systemInited = true;
        }
        else  // 此时还不到初始化，直接结束
            return;
    }

    TicToc t_whole;
    TicToc t_prepare;
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    // 把点云从ros格式转到pcl的格式
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;
    // 去除掉点云中的nan点：天空点，发射后没有接收到
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    // 去除距离小于阈值的点：删除近点
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

    // notes：激光坐标系的表达为右-前-上

    // 计算起始点和结束点的角度，由于激光雷达是顺时针旋转，这里取反就相当于转成了逆时针
    int cloudSize = laserCloudIn.points.size();
    // 此时发射的所有scan计算的角度都相同，联系图像理解
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    // atan2范围是[-Pi,PI]，这里加上2PI是为了保证起始到结束相差2PI符合实际
    // 此时发射的所有scan计算的角度也都相同，联系图像理解
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

#if 0
    std::cout << "startOri = " << startOri << std::endl;
    std::cout << "endOri = " << endOri << std::endl;
    getchar();
#endif

    // 总有一些例外，比如这里大于3PI，和小于PI，就需要做一些调整到合理范围
    if (endOri - startOri > 3 * M_PI)
    {
        // 这种情况表示结果接近4pi
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        // 这种结果表示结果接近0
        endOri += 2 * M_PI;
    }
    // 经过以上处理后，endOri - startOri接近2pi
    //printf("end Ori %f\n", endOri);

    /**
     * 对于激光线束的理解：线束是从下往上排列的，这可以从+15看出
     *      1. 16线激光：相邻线束相隔2度，排列分别为-15、-13、...-1、1、...、13、15
     */

    bool halfPassed = false;
    int count = cloudSize;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    // 遍历每一个点
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
        // 计算他的俯仰角：注意联系激光坐标系理解，这个角度正好就是当前线束发射的俯仰角
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;
        // 计算是第几根scan
        if (N_SCANS == 16)
        {
            // 线束ID从下往上
            scanID = int((angle + 15) / 2 + 0.5);  // 加0.5是为了四舍五入
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);
        // 计算水平角
        float ori = -atan2(point.y, point.x);
        if (!halfPassed)  // 初始值为false，表示距离起始是否超过了一半
        { 
            // 确保-PI / 2 < ori - startOri < 3 / 2 * PI
            if (ori < startOri - M_PI / 2)
            {
                // ori接近-pi，startOri接近pi
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                // ori接近pi，startOri接近-pi；此种情况一旦发生，ori < startOri，后面的计算就会出现负数
                ori -= 2 * M_PI;
            }
            // 如果超过180度，就说明过了一半了
            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            // 此时halfPassed为true，相比于startOri，ori应该更接近endOri
            // 确保-PI * 3 / 2 < ori - endOri < PI / 2
            ori += 2 * M_PI;    // 先补偿2PI
            if (ori < endOri - M_PI * 3 / 2)
            {
                // 此种情况不可能发生，因为ori已经累加了2pi
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        // 角度的计算是为了计算相对的起始时刻的时间
        float relTime = (ori - startOri) / (endOri - startOri);
        // 整数部分是scan的索引，小数部分是相对起始时刻的时间
        point.intensity = scanID + scanPeriod * relTime;
        // 根据scan的idx送入各自数组
        laserCloudScans[scanID].push_back(point); 
    }
    // cloudSize是有效的点云的数目
    cloudSize = count;
    printf("points size %d \n", cloudSize);
#if 0
    for (int i = 0; i < N_SCANS; i++) {
        pcl::PointCloud<PointType> points = laserCloudScans[i];
        for (auto &point: points) {
            std::cout << point.intensity << std::endl;
        }
    }
#endif

    // notes: laserCloudScans中存储的点云是按照顺序排列的

    // 得到的laserCloud是按照scan进行排序的，从矩阵的角度来说，就是ROW-MAJOR
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    // 全部集合到一个点云里面去，但是使用两个数组标记其实和结果，这里分别+5和-6是为了计算曲率方便
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = laserCloud->size() + 5;
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;

#if 0
        std::cout << "laserCloud->size() + 5  = " << scanStartInd[i] << std::endl;
        std::cout << "laserCloud->size() - 6 = " << scanEndInd[i] << std::endl;
#endif
    }

    printf("prepare time %f \n", t_prepare.toc());
    // 开始计算曲率
    for (int i = 5; i < cloudSize - 5; i++)
    {
        /**
         * 对三个轴，使用左右5个点与当前点进行比较，计算结果可以认为是当前点附近在这个轴上的波动
         *
         * 需要注意的是，如果左边的点距离LIDAR较近，右边的点距离LIDAR较远，那么求得的diffX可能接近0，但是实际场景并不是左右两边的点都接近当前点
         * 建议使用∑|xi - x0|来计算diffX，对diffY和diffZ按照同理进行计算
         */
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        // 存储曲率，索引
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
    }


    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;
    // 遍历每个scan
    for (int i = 0; i < N_SCANS; i++)
    {
        // 有效的点云太少，只有不超过6个，这种时候不提取特征点
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        // 用来存储不太平整的点
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        // 将每个scan等分成6等分
        for (int j = 0; j < 6; j++)
        {
            // 每个等分的起始和结束点
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;
            // 对点云按照曲率进行排序，小的在前，大的在后
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            t_q_sort += t_tmp.toc();

            /**
             * 以0.1为阈值，大于阈值的为角点，小于阈值的为面点；
             * 每提取一个特征点，都会将其同一个scan的左右的5个点，如果距离较近，就不会将他们提取为特征点
             */

            int largestPickedNum = 0;
            // 挑选曲率比较大的部分
            for (int k = ep; k >= sp; k--)
            {
                // 排序后顺序就乱了，这个时候索引的作用就体现出来了
                int ind = cloudSortInd[k]; 

                // 看看这个点是否是被设置为可以提取，同时曲率是否大于阈值
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {

                    largestPickedNum++;
                    // 每段选2个曲率大的点
                    if (largestPickedNum <= 2)
                    {                        
                        // label为2是曲率大的标记
                        cloudLabel[ind] = 2;
                        // cornerPointsSharp存放大曲率的点
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);  // 比较大的曲率对应的特征点，数量为20个，包含了cornerPointsSharp
                    }
                    // 以及20个曲率稍微大一些的点
                    else if (largestPickedNum <= 20)
                    {                        
                        // label置1表示曲率稍微大
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    // 超过20个就算了
                    else
                    {
                        break;
                    }
                    // 这个点被选中后 pick标志位置1
                    cloudNeighborPicked[ind] = 1; 
                    // 为了保证特征点不过度集中，将选中的点周围5个点都置1,避免后续会选到
                    // 在周围的点中，如果距离比较近，就不提取特征点了
                    for (int l = 1; l <= 5; l++)
                    {
                        // 查看相邻点距离是否差异过大，如果差异过大说明点云在此不连续，是特征边缘，就会是新的特征，因此就不置位了
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    // 下面同理
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
            // 下面开始挑选面点
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];
                // 确保这个点没有被pick且曲率小于阈值
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {
                    // -1认为是平坦的点
                    cloudLabel[ind] = -1; 
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    // notes: 这里并没有像角点一样提取较为平坦的点，而是直接仅仅知识提取了面点

                    smallestPickedNum++;
                    // 这里不区分平坦和比较平坦，因为剩下的点label默认是0,就是比较平坦
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }
                    // 下面同理
                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // notes：较为平坦的特征点在这里被提取，剩余的未被提取的点都在这里
            for (int k = sp; k <= ep; k++)
            {
                // 这里可以看到，剩下来的点都是一般平坦，这个也符合实际
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        // 对较为平坦的点的集合执行体素滤波
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        // 一般平坦的点比较多，所以这里做一个体素滤波
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    // 分别将当前点云、四种特征的点云发布出去
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    // laserCloud包含了当前的所有的点云，并且其按照scan有序排列
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    // 发布角点
    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    // 发布一般的角点
    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    // 发布面点
    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    // 发布一般的面点
    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // pub each scam
    // 可以按照每个scan发出去，不过这里是false
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            // 获取每一个scan的点云数据并发布出去
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;
    // 从配置文件中获取多少线的激光雷达
    nh.param<int>("scan_line", N_SCANS, 16);
    // 最小有效距离：用于去除近点，比如载体，载体是动态物体
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    printf("scan line number %d \n", N_SCANS);
    // 只有线束是16 32 64的才可以继续
    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    // 订阅激光雷达消息
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    // 发布，由雷达里程计接收
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);
    // 发布，由雷达里程计接收
    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);
    // 发布，由雷达里程计接收
    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);
    // 发布，由雷达里程计接收
    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);
    // 发布，由雷达里程计接收
    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);
    // 没有节点接收，用于rviz显示
    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    if(PUB_EACH_LINE)  // 这里是false
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            // 没有接收者
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();  // 循环执行

    return 0;
}


#include "odom.hpp"
#include <Eigen/Core>
#include <tuple>
#include <vector>
#include <pcl/common/concatenate.h>
#include <fstream>
#include <chrono>
using namespace std;
namespace {
using Vector3i = Eigen::Matrix<int, 3, 1>;
using Voxel = Vector3i;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3d = Eigen::Matrix<double, 3, 3>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
inline double square(double x) { return x * x; }
struct VoxelHash {
    size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
    }
};
struct ResultTuple {
    ResultTuple() {
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTuple operator+(const ResultTuple &other) {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }

    Matrix6d JTJ;
    Vector6d JTr;
};
} 
namespace Odom {
constexpr int MAX_NUM_ITERATIONS_ = 200;
double ESTIMATION_THRESHOLD_=0.0005;
/**
 * [功能描述]：帧注册函数，将当前彩色帧与地图帧进行配准，并更新局部地图
 * @param color_frame：当前彩色点云帧数据，包含[x,y,z,intensity,time,color]信息
 * @param map_frame：地图点云帧数据，用于配准的目标帧
 * @param factor：距离因子，用于控制配准精度和搜索范围
 * @param initial_guess：初始位姿估计，SE3变换矩阵
 * @param new_pose：输出参数，配准后的新位姿（引用传递）
 * @return Vector6dVector：返回输入的彩色帧数据
 */
Vector6dVector Onion_Odom::RegisterFrame(const Vector6dVector color_frame, const Vector6dVector map_frame, double factor, const Sophus::SE3d &initial_guess, Sophus::SE3d &new_pose) {
	// 调用彩色配准函数，计算新的位姿估计
	new_pose = Color_RegisterFrame(color_frame,
                                          initial_guess,
                                          factor);
    
    // 设置局部地图的搜索距离为因子的一半
    color_local_map_.dis_ = factor/2.0;
    
    // 根据搜索距离与体素大小的关系，自适应设置搜索范围
    // 1.732是sqrt(3)，用于计算体素对角线长度
    // 当搜索距离较小时，使用较小的搜索范围（num_range_ = 0）
    if (3*color_local_map_.dis_<0.5*1.732*color_local_map_.voxel_size_)
    	color_local_map_.num_range_ = 0; // 最小搜索范围，适用于高精度配准
    // 当搜索距离中等时，使用中等搜索范围（num_range_ = 1）
    else if (3*color_local_map_.dis_<1.5*1.732*color_local_map_.voxel_size_)
    	color_local_map_.num_range_ = 1; // 中等搜索范围，平衡精度和效率
    // 当搜索距离较大时，使用最大搜索范围（num_range_ = 2）
    else
    	color_local_map_.num_range_ = 2; // 最大搜索范围，适用于粗糙配准
    
    // 打印调试信息：搜索距离
    cout<<"color_local_map_.dis_-----------"<<color_local_map_.dis_<<endl;
    // 打印调试信息：搜索范围级别
    cout<<"color_local_map_.num_range_-----------"<<color_local_map_.num_range_<<endl;
    
    // 使用新位姿更新局部彩色地图
    color_local_map_.Color_Update(map_frame, new_pose);
    
    // 增加扫描计数，用于统计处理的帧数
	scan_num++;
	
	// 返回输入的彩色帧数据
	return color_frame;
}
//-------------------------------------------------------------------------------------------------------

void Color_TransformPoints(const Sophus::SE3d &T, Vector6dVector &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { 
                   Eigen::Vector4d homogeneous_point(point[0], point[1], point[2], 1.0);
				   Eigen::Vector4d transformed_homogeneous_point = T.matrix() * homogeneous_point;
				   Vector6d transformed_point;
				   transformed_point << transformed_homogeneous_point[0], transformed_homogeneous_point[1], transformed_homogeneous_point[2], point[3], point[4], point[5],point[6];
  				   return transformed_point; });
}

/**
 * [功能描述]：点云对齐函数，使用加权最小二乘法求解两个点云之间的最优SE(3)变换
 * @param source：源点云数据，需要进行变换的点云
 * @param target：目标点云数据，变换后的目标位置
 * @param th：鲁棒估计阈值，用于计算权重函数，剔除异常值
 * @return Sophus::SE3d：返回求解得到的最优SE(3)变换矩阵
 */
Sophus::SE3d AlignClouds(const Vector6dVector &source,
                         const Vector6dVector &target,
                         double th) {
    // 定义雅可比矩阵和残差计算函数（Lambda表达式）
    auto compute_jacobian_and_residual = [&](auto i) {
        // 计算位置残差：源点与目标点的位置差
        const Eigen::Vector3d pos_residual = (source[i].template head<3>() - target[i].template head<3>()).template cast<double>();
        
        // 初始化3x6雅可比矩阵
        Matrix3_6d J_r;
        // 雅可比矩阵的前3x3块：对平移的偏导数（单位矩阵）
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        // 雅可比矩阵的后3x3块：对旋转的偏导数（反对称矩阵）
        // SO3d::hat()将向量转换为反对称矩阵，用于旋转的线性化
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i].template head<3>().template cast<double>());
        
        // 返回雅可比矩阵和位置残差的元组
        return std::make_tuple(J_r, pos_residual);
    };

    // 使用TBB并行归约计算高斯-牛顿法的正规方程组
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        tbb::blocked_range<size_t>{0, source.size()}, // 并行处理范围：所有点云数据
        ResultTuple(), // 初始累加器：零矩阵和零向量
        // Lambda表达式：每个线程处理一个数据块
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
            // 定义鲁棒权重函数：th/(th + residual)，用于降低异常值影响
            auto Weight = [&](double residual, double th_) { return th_ / (th_ + residual); };
            // 解构累加器，获取线程本地的JTJ和JTr
            auto &[JTJ_private, JTr_private] = J;
            
            // 遍历当前线程分配的点云范围
            for (auto i = r.begin(); i < r.end(); ++i) {
                // 计算第i个点的雅可比矩阵和残差
                const auto &[J_r, pos_residual] = compute_jacobian_and_residual(i);
                // 根据残差的平方范数计算权重（鲁棒估计）
                double w = Weight(pos_residual.squaredNorm(), th);
                
                // 累加JTJ矩阵：J_r^T * w * J_r
                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                // 累加JTr向量：J_r^T * w * pos_residual
                JTr_private.noalias() += J_r.transpose() * w * pos_residual;
            }
            return J; // 返回线程本地的累加结果
        },
        // 归约函数：合并不同线程的结果
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });
    
    // 求解正规方程组：JTJ * x = -JTr
    // 使用LDLT分解求解线性方程组（高效且数值稳定）
    const Vector6d x = JTJ.ldlt().solve(-JTr);
    
    // 将李代数向量x转换为SE(3)变换矩阵
    // SE3d::exp()将se(3)李代数映射到SE(3)李群
    return Sophus::SE3d::exp(x);
}



/**
 * [功能描述]：彩色帧配准函数，使用ICP算法将关键帧与局部地图进行精确配准
 * @param key_frame：输入的关键帧点云数据，包含[x,y,z,intensity,time,color]信息
 * @param initial_guess：初始位姿估计，SE3变换矩阵，作为配准的起始点
 * @param factor：距离因子，用于控制配准精度和收敛条件
 * @return Sophus::SE3d：返回配准后的位姿变换矩阵
 */
Sophus::SE3d Onion_Odom::Color_RegisterFrame(
						   const Vector6dVector &key_frame,
                           const Sophus::SE3d &initial_guess,
                           double factor) {
    // 复制关键帧数据作为源点云
    Vector6dVector key_source = key_frame; 
    // 使用初始位姿估计变换源点云，将其转换到估计的坐标系
    Color_TransformPoints(initial_guess, key_source); 
    // 初始化ICP累积变换矩阵为单位矩阵
    Sophus::SE3d T_icp = Sophus::SE3d();
    
    // ICP迭代配准循环，最大迭代次数为MAX_NUM_ITERATIONS_
	for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {
		// 在局部地图中寻找与当前源点云对应的点对
		// 搜索半径为dis_的3倍，扩大搜索范围以找到更多对应点
		const auto &[src, tgt] = color_local_map_.GetCorrespondences(key_source, color_local_map_.dis_ * 3.0);
		
		// 使用找到的对应点对进行点云配准
		// 阈值设为dis_的1/3，用于鲁棒估计和异常值剔除
		auto estimation = AlignClouds(src, tgt, color_local_map_.dis_/3.0);
		
		// 检查估计的变换是否接近单位矩阵（即无明显变化）
		// 如果变换很小（小于1e-5），跳过此次迭代继续下次
		if (estimation.matrix().isApprox(Eigen::Matrix4d::Identity(), 1e-5)) {
    		continue;
		}
		
		// 使用当前估计的变换更新源点云位置
		Color_TransformPoints(estimation, key_source);
		// 累积ICP变换：将当前变换与之前的累积变换组合
		T_icp = estimation * T_icp;
		
		// 检查收敛条件：如果变换的对数范数小于阈值，认为已收敛
		// estimation.log()返回se(3)李代数，.norm()计算其范数
		if (estimation.log().norm() < ESTIMATION_THRESHOLD_) {
		    break; // 收敛，退出迭代循环
		}
	}
    // 返回最终配准结果：ICP累积变换与初始估计的复合
    return T_icp*initial_guess;
}
}  // namespace

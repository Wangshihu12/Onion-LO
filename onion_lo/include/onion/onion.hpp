#include<iostream>
#include <cmath>
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 
#include <tbb/parallel_for.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/tbb.h>
#include <mutex>
#include <vector>
#include <functional> 
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <tsl/robin_map.h>

std::mutex print_mutex;
using namespace std;
class Onion {
public:
    double layer_thickness = 5;
    double onion_angle = 3.0;
private:
    //---------------------------------------
    int seg_min_num = 5;
    int exp_key_num = 1000;
    double min_range = 0.0;
    double max_range = 200;
    double angle_v = 0.4;
    double angle_h = 0.01;
    double seg_angle = 0.4;
    //------------------------------Layer------------------------------------
	struct Layer {
		int face;
		int R;
		bool operator==(const Layer& other) const {
		    return face == other.face && R == other.R;
		}
	};
	struct LayerHash {
		std::size_t hash(const Layer& layer) const {
		    std::size_t h1 = std::hash<int>()(layer.face);
		    std::size_t h2 = std::hash<int>()(layer.R);
		    return h1 ^ (h2 << 1);
		}
		std::size_t operator()(const Layer& layer) const {
		    return hash(layer);
		}
		bool equal(const Layer& lhs, const Layer& rhs) const {
		    return lhs.face == rhs.face && lhs.R == rhs.R;
		}
	};
	struct LayerBlock {
		double volume;
		double plane_num = 0;
		int seg_num;
		Vector6dVector seg_points;
		Vector6dVector p_points;
		Vector6dVector np_points;
		Vector6dVector points;
		LayerBlock() = default;
		LayerBlock(const Vector6d &point) {
		    AddPoint(point);
		}
		inline void AddPoint(const Vector6d &point) {
		    points.push_back(point);
		}
		void Merge(const LayerBlock& other) {
		    for (const auto& point : other.points) {
		        this->points.push_back(point);
		    }
		}
		void Merge(LayerBlock&& other) {
		    this->points.insert(this->points.end(), std::make_move_iterator(other.points.begin()), std::make_move_iterator(other.points.end()));
		}
	};
	//----------------------------Cell---------------------------------
	struct Cell {
		int x, y, z;
		bool operator==(const Cell& other) const {
		    return x == other.x && y == other.y && z == other.z;
		}
	};
	struct CellHash {
		std::size_t hash(const Cell& cell) const {
		    std::size_t hx = std::hash<int>()(cell.x);
		    std::size_t hy = std::hash<int>()(cell.y);
		    std::size_t hz = std::hash<int>()(cell.z);
		    return hx ^ (hy << 1) ^ (hz << 2);
		}
		bool equal(const Cell& a, const Cell& b) const {
		    return a == b;
		}
	};
	struct CellBlock {
		int num_points_ = 0;
		bool intensity_fea = 0;
		bool line_fea = 0;
		bool planar_fea = 0;
		Vector6dVector points;
		void AddPoint(const Vector6d &point) {
		    points.push_back(point);
		    num_points_++;
		}
		void Merge(const CellBlock& other) {
		    points.insert(points.end(), other.points.begin(), other.points.end());
		    num_points_ += other.num_points_;
		}
	};
	void thread_safe_print(const std::string& message1, const std::string& message2, const std::string& message3, const std::string& message4, const std::string& message5, const std::string& message6) {
		std::lock_guard<std::mutex> lock(print_mutex);
		std::cout <<message1<< message2<<message3<<message4<<message5<<message6<<std::endl;
	}
public:
	double Total_V;
	double Plane_rotio;
	int Total_NUM;
	int Raw_points_num;
	double Onion_Factor;
	Vector6dVector Seg_points;
	tbb::concurrent_hash_map<Layer, LayerBlock, LayerHash> onion_ball;
	/**
	 * [功能描述]：创建Onion结构体，将3D点云按球面分层组织，实现多线程并行处理
	 * @param PointCloud：输入点云数据，每个点包含[x,y,z,intensity,time,color]信息
	 * @param Resolution_v：垂直方向角度分辨率（弧度）
	 * @param Resolution_h：水平方向角度分辨率（弧度）
	 * @param key_num：期望的关键点数量
	 * @return 无返回值，结果存储在onion_ball成员变量中
	 */
	void Create_Onion(Vector6dVector PointCloud, double Resolution_v,double Resolution_h, double key_num) {
		exp_key_num = key_num; // 设置期望关键点数量
		angle_v = Resolution_v; // 设置垂直角度分辨率
		angle_h = Resolution_h; // 设置水平角度分辨率
		seg_angle = std::max(angle_v, angle_h); // 取最大角度作为分割角度
		Raw_points_num = PointCloud.size(); // 记录原始点云数量
		
		// 创建线程本地的Onion球面结构，用于并行处理
		tbb::combinable<std::unordered_map<Layer, LayerBlock, LayerHash>> local_onion_balls;
		
		// 使用TBB并行处理点云数据
		tbb::parallel_for(tbb::blocked_range<size_t>(0, PointCloud.size()), [&](const tbb::blocked_range<size_t>& r) {
		    auto& local_map = local_onion_balls.local(); // 获取当前线程的本地映射表
		    for (size_t i = r.begin(); i != r.end(); ++i) { // 遍历当前线程分配的点云范围
		        auto& point = PointCloud[i]; // 获取当前点
		        Layer layer; // 创建层结构体
		        double x = point[0]; // 提取x坐标
		        double y = point[1]; // 提取y坐标
		        double z = point[2]; // 提取z坐标
		        double dist = std::sqrt(x * x + y * y + z * z); // 计算点到原点的距离
		        
		        // 距离过滤：只处理在有效范围内的点
		        if (dist>min_range || dist < max_range){
				    double theta = std::acos(z / dist); // 计算极角θ（与z轴夹角）
				    double phi = atan2(y, x); // 计算方位角φ（在xy平面内的角度）
					layer.R = std::ceil(dist / layer_thickness); // 计算径向层号（向上取整）
					
					// 根据角度将点分配到6个面之一（类似立方体投影）
					if (theta <= M_PI / 4) {
						layer.face = 1; // 上表面（z>0且接近z轴）
					} else if (theta >= 3 * M_PI / 4) {
						layer.face = -1; // 下表面（z<0且接近-z轴）
					} else if (phi >= -M_PI / 4 && phi <= M_PI / 4) {
						layer.face = 2; // 前表面（x>0且接近x轴）
					} else if (phi >= 3 * M_PI / 4 && phi <= 5 * M_PI / 4) {
						layer.face = -2; // 后表面（x<0且接近-x轴）
					} else if (phi >= M_PI / 4 && phi <= 3 * M_PI / 4) {
						layer.face = 3; // 右表面（y>0且接近y轴）
					} else {
						layer.face = -3; // 左表面（y<0且接近-y轴）
					}
					
					// 将点添加到对应的层块中
					auto it = local_map.find(layer); // 查找是否已存在该层
					if (it == local_map.end()) {
						local_map[layer] = LayerBlock(point); // 创建新的层块
					} else {
						it->second.AddPoint(point); // 向现有层块添加点
					}
				}
		    }
		});
		
		// 合并所有线程的本地结果到全局onion_ball
		local_onion_balls.combine_each([&](std::unordered_map<Layer, LayerBlock, LayerHash>& local_map) {
		    for (auto& pair : local_map) { // 遍历每个线程的本地映射
		        tbb::concurrent_hash_map<Layer, LayerBlock, LayerHash>::accessor accessor; // 创建访问器
		        if (onion_ball.insert(accessor, pair.first)) { // 尝试插入新层
		            accessor->second = std::move(pair.second); // 移动构造新层块
		        } else {
		            accessor->second.Merge(std::move(pair.second)); // 合并到现有层块
		        }
		    }
		});
	}
void IG_Classifier(const Vector6dVector& input_point, double voxelSize, double v_size,Vector6dVector& p_point, Vector6dVector& np_point, double &plane_rate, double &V, Vector6dVector& all_points) {
	double plane_num = 0;
	double no_plane_num = 0;
    p_point.reserve(input_point.size());
    all_points.reserve(input_point.size());
    np_point.reserve(input_point.size());
    tbb::concurrent_hash_map<Cell, CellBlock, CellHash> segVoxels;
    tbb::concurrent_hash_map<Cell, CellBlock, CellHash> fixedVoxels;
    for (const auto& p : input_point) {
        Cell cell_key{
            static_cast<int>(std::floor(p[0] / voxelSize)),
            static_cast<int>(std::floor(p[1] / voxelSize)),
            static_cast<int>(std::floor(p[2] / voxelSize))
        };
        tbb::concurrent_hash_map<Cell, CellBlock, CellHash>::accessor accessor;
        if (segVoxels.insert(accessor, cell_key)) {
            accessor->second = CellBlock();
        }
        accessor->second.AddPoint(p);
        Cell fixed_cell_key{
		    static_cast<int>(std::floor(p[0] / v_size)),
		    static_cast<int>(std::floor(p[1] / v_size)),
		    static_cast<int>(std::floor(p[2] / v_size))
		};
		tbb::concurrent_hash_map<Cell, CellBlock, CellHash>::accessor fixed_accessor;
		if (fixedVoxels.insert(fixed_accessor, fixed_cell_key)) {
		    fixed_accessor->second = CellBlock();
		}
		fixed_accessor->second.AddPoint(p);
    }
    int occupiedVoxelCount = 0;
	for (auto it = fixedVoxels.begin(); it != fixedVoxels.end(); ++it) {
		if (it->second.num_points_ > 3) {
		    occupiedVoxelCount++;
		}
	}
	double computedVolume = occupiedVoxelCount * std::pow(v_size, 3);
	double NUM = 0;
	double p_NUM = 0;
	double np_NUM = 0;
	double use_points = 0;
    for (auto& [cell_key, cell_block] : segVoxels) {
        double point_count = cell_block.num_points_;
        double con_w = point_count/std::pow(v_size, 3);
        if (point_count >= seg_min_num) {
        	NUM++;
        	use_points += point_count;
        	std::vector<double> intensities;
            Eigen::Matrix3f covariance_matrix = Eigen::Matrix3f::Zero();
            Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
            Eigen::Matrix<float, 1, 1> covariance_matrix_i = Eigen::Matrix<float, 1, 1>::Zero();
			float intensity_mean = 0.0f;
            for (const auto& point : cell_block.points) {
                centroid[0] += point[0];
                centroid[1] += point[1];
                centroid[2] += point[2];
                intensity_mean += point[3];
                intensities.push_back(point[3]);
            }
            centroid /= static_cast<float>(point_count);
			intensity_mean /= static_cast<float>(point_count);
            //---------------------------------------------------------------------
			std::sort(intensities.begin(), intensities.end());
			double intensity_median;
			size_t size = intensities.size();
			if (size % 2 == 0) {
				intensity_median = (intensities[size / 2 - 1] + intensities[size / 2]) / 2.0;
			} else {
				intensity_median = intensities[size / 2];
			}
			double variance_sum = 0.0;
			for (const auto& intensity : intensities) {
				variance_sum += std::pow(intensity - intensity_mean, 2);
			}
			double variance = variance_sum / intensities.size();
			double standard_deviation = std::sqrt(variance);
			//---------------------------------------------------------------------
            for (const auto& point : cell_block.points) {
                Eigen::Vector3f p(point[0], point[1], point[2]);
                covariance_matrix += (p - centroid) * (p - centroid).transpose();
                covariance_matrix_i += Eigen::Matrix<float, 1, 1>::Constant((point[3] - intensity_mean) * (point[3] - intensity_mean));
            }
            covariance_matrix /= static_cast<float>(point_count);
            covariance_matrix_i /= static_cast<float>(point_count);
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 1, 1>> eigen_solver_i(covariance_matrix_i);
			float eigenvalue_i = eigen_solver_i.eigenvalues()[0];
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix);
            Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues();
            float sum = eigenvalues.sum();
            float e0 = eigenvalues[0] / sum;
			float e1 = eigenvalues[1] / sum;
			float e2 = eigenvalues[2] / sum;
			double score = e1/(e0+e2);
            bool isintensity = (eigenvalue_i >50);
            bool isPlane = e2/e0>10 && e1/e0>10;
            if (isPlane) {
            	p_NUM++;
		        for (auto& p : cell_block.points) {
	        		p[3] = 0;
					p[4] = 255;
					p[5] = 0;
					if (isintensity && (p[3]<0.3*eigenvalue_i || p[3]>0.7*eigenvalue_i)){
						p[3] = 255;
						p[4] = 255;
					}
					plane_num++;
					p_point.push_back(p);
					all_points.push_back(p);
		        }
            }
            else{
            	np_NUM++;
            	for (auto& p : cell_block.points) {
					p[3] = 255;
					p[4] = 0;
					p[5] = 0;
					no_plane_num++;
					np_point.push_back(p);
					all_points.push_back(p);
		        }
            }
    }
    }
    if (use_points>0){
		plane_rate = p_NUM/(use_points);
		V = computedVolume;
    }
}

//-----------------------------------Classifier--------------------------------
/**
 * [功能描述]：点云分类器，对Onion结构体中的每个层进行平面/非平面特征分类
 * @return Vector6dVector：返回分类后的点云数据，包含颜色信息
 */
Vector6dVector Classifier() {
    Seg_points.reserve(Raw_points_num); // 预分配内存空间，存储分割后的点云
    double total_volume = 0.0; // 总体积统计
    double total_num = 0.0; // 总点数统计
    double plane_num = 0.0; // 平面点数量统计
	double Layer_size = 0.0; // 层数量统计
	Layer_size=onion_ball.size(); // 获取Onion球面结构的总层数
    std::mutex mutex; // 创建互斥锁，用于线程安全的结果合并
    // 使用TBB并行处理每个层
    tbb::parallel_for(tbb::blocked_range<size_t>(0, onion_ball.size()), [&](const tbb::blocked_range<size_t>& range) {
        Vector6dVector local_points_thread; // 线程本地的点云数据
        double local_total_num = 0.0; // 线程本地的总点数
        double local_plane_num = 0.0; // 线程本地的平面点数
        double local_total_volume = 0.0; // 线程本地的总体积
        // 遍历当前线程分配到的层范围
        for (size_t i = range.begin(); i < range.end(); ++i) {
            auto it = onion_ball.begin(); // 获取Onion球面结构的迭代器
            std::advance(it, i); // 移动到第i个元素
            const auto& layer = it->first; // 获取层信息（面号和径向层号）
            auto& block = it->second; // 获取层块数据
            
            // 层过滤条件：径向层号在1-50之间且点数大于30
            if (layer.R >= 1.0 && layer.R < 50.0 && block.points.size() > 30) {
                double R = layer.R * layer_thickness; // 计算实际径向距离
                
                // 计算分类体素大小：基于径向距离和分割角度
                double class_size = std::round(R * std::sin(3.0 *seg_angle * M_PI / 180.0) * 10.0) / 10.0;
                
                // 计算固定体素大小：基于径向距离和Onion角度
                double v_size = std::round(R * std::sin(onion_angle * M_PI / 180.0) * 10.0) / 10.0;
                Vector6dVector plane_points_local; // 本地平面点集合
                Vector6dVector no_plane_points_local; // 本地非平面点集合
                Vector6dVector total_points; // 本地总点集合
                double volume = 0.0; // 体积统计
                double plane_rate_local = 0.0; // 平面率统计
                
                // 调用IG分类器进行特征分类
                IG_Classifier(block.points, class_size, v_size, plane_points_local, 
                				no_plane_points_local, plane_rate_local, volume, total_points);
                // 更新层块数据
                block.seg_points = total_points; // 存储分割后的点云
                block.p_points = plane_points_local; // 存储平面点
                block.np_points = no_plane_points_local; // 存储非平面点
                block.seg_num = total_points.size(); // 记录分割点数
                block.plane_num = plane_points_local.size(); // 记录平面点数
                block.volume = volume; // 记录体积
                // 累加线程本地统计
                local_points_thread.insert(local_points_thread.end(), total_points.begin(), total_points.end());
                local_total_num += block.seg_num; // 累加总点数
                local_plane_num += block.plane_num; // 累加平面点数
                local_total_volume += block.volume; // 累加总体积
                // 线程安全的调试信息打印
                thread_safe_print(
                    "layer.Face " + std::to_string(layer.face), // 打印面号
                    "  layer.R " + std::to_string(layer.R), // 打印径向层号
                    "  v_size " + std::to_string(v_size), // 打印体素大小
                    "  class_size " + std::to_string(class_size), // 打印分类大小
                    "  block.volume " + std::to_string(block.volume), // 打印块体积
                    "  block.seg_num " + std::to_string(block.seg_num) // 打印分割点数
                );
            }
        }  
        // 线程安全地合并结果到全局变量
        std::lock_guard<std::mutex> lock(mutex); // 获取互斥锁
        Seg_points.insert(Seg_points.end(), local_points_thread.begin(), local_points_thread.end()); // 合并点云数据
        total_num += local_total_num; // 合并总点数
        plane_num += local_plane_num; // 合并平面点数
        total_volume += local_total_volume; // 合并总体积
    });
    // 计算全局统计信息
    Total_NUM = Seg_points.size(); // 记录总点数
    Plane_rotio = Total_NUM > 0 ? static_cast<double>(plane_num) / Total_NUM : 0.0; // 计算平面点比例
    Total_V = total_volume; // 记录总体积
    Onion_Factor = std::pow(total_volume/exp_key_num, 1.0/3.0); // 计算Onion因子（立方根）
    
    return Seg_points; // 返回分类后的点云数据
}
using Vector3i = Eigen::Matrix<int, 3, 1>;
using Voxel = Vector3i;
struct VoxelHash {
    size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
    }
};
Vector6dVector D_PointCloud(double target_total_points, Vector6dVector inputs) {
        Vector6dVector downsampled_points;
    	downsampled_points.reserve(target_total_points);
        int step = std::round(inputs.size() / target_total_points);
        if (step < 1) step=1;
        for (int i = 0; i < inputs.size(); i += step) {
            downsampled_points.push_back(inputs[i]);
        }
        return downsampled_points;
}
Vector6dVector Voxel_Downsample(double target_total_points, const Vector6dVector& inputs, double V) {
    Vector6dVector frame_downsampled;
    if (inputs.size() > target_total_points) {
        double voxel_size = std::pow(V / target_total_points, 1.0 / 3.0);
        double inv_voxel_size = 1.0 / voxel_size;
        tsl::robin_map<Voxel, Vector6d, VoxelHash> grid;
        grid.reserve(inputs.size());
        for (const auto &point : inputs) {
            int voxel_x = static_cast<int>(point[0] * inv_voxel_size);
            int voxel_y = static_cast<int>(point[1] * inv_voxel_size);
            int voxel_z = static_cast<int>(point[2] * inv_voxel_size);
            Voxel voxel;
            voxel << voxel_x, voxel_y, voxel_z;
            grid.try_emplace(voxel, point);
        }
        frame_downsampled.reserve(grid.size());
        for (const auto &pair : grid) {
            frame_downsampled.push_back(pair.second);
        }
    } else {
        frame_downsampled = inputs;
    }
    return frame_downsampled;
}
/**
 * [功能描述]：基于Onion球面结构的智能点云下采样函数，根据体积权重自适应分配采样点数
 * @param target_total_points：目标总点数，控制最终输出的点云规模
 * @param Key_PointCloud：输出参数，存储关键点云数据（引用传递）
 * @return Vector6dVector：返回下采样后的点云数据
 */
Vector6dVector NEW_Downsample_PointCloud(int target_total_points, Vector6dVector &Key_PointCloud) {
    Vector6dVector Downsample_PointCloud; // 存储下采样后的点云数据
    Downsample_PointCloud.reserve(target_total_points); // 预分配内存空间，提高性能
    Key_PointCloud.reserve(target_total_points); // 预分配关键点云内存空间
    std::mutex mutex; // 创建互斥锁，用于线程安全的结果合并
    
    // 使用TBB并行处理Onion球面结构中的每个层
    tbb::parallel_for(tbb::blocked_range<size_t>(0, onion_ball.size()), [&](const tbb::blocked_range<size_t>& range) {
        Vector6dVector local_points_thread; // 线程本地的下采样点云数据
        Vector6dVector local_key_points_thread; // 线程本地的关键点云数据
        // 预分配内存，限制最大容量避免内存溢出
        local_points_thread.reserve(std::min(5 * target_total_points, 10000));
        local_key_points_thread.reserve(std::min(target_total_points, 10000));
        
        // 遍历当前线程分配到的层范围
        for (size_t i = range.begin(); i < range.end(); ++i) {
            auto it = onion_ball.begin(); // 获取Onion球面结构的迭代器
            std::advance(it, i); // 移动到第i个元素
            const auto& layer = it->first; // 获取层信息（面号和径向层号）
            auto& block = it->second; // 获取层块数据
			Vector6dVector filter_points; // 过滤后的点云数据（未使用）
            
            // 层过滤条件：径向层号在1-100之间且分割点数大于10
            if (layer.R >= 1.0 && layer.R < 100.0 && block.seg_points.size() > 10) {
                // 计算数量权重：当前层分割点数占总分割点数的比例
                double num_weight = static_cast<double>( block.seg_num) / static_cast<double>(Total_NUM);
                // 计算体积权重：当前层体积占总体积的比例
                double v_weight = block.volume / Total_V;
                // 根据体积权重计算当前层应分配的下采样点数，最少10个点
                int down_num = std::max(10, static_cast<int>(std::ceil((v_weight) * target_total_points)));
                Vector6dVector down_points; // 下采样点云数据
                Vector6dVector key_points; // 关键点云数据
                // 预分配内存，避免重复分配
                down_points.reserve(std::min(down_num, static_cast<int>(block.seg_points.size())));
                key_points.reserve(std::min(down_num, static_cast<int>(block.seg_points.size())));
                
                // 第一次体素下采样：使用5倍目标点数进行粗采样
            	down_points = Voxel_Downsample(5.0*down_num, block.seg_points, block.volume);
                // 第二次体素下采样：从粗采样结果中提取关键点
            	key_points = Voxel_Downsample(down_num, down_points, block.volume);
                
                // 将当前层的下采样点云添加到线程本地集合
                local_points_thread.insert(local_points_thread.end(), down_points.begin(), down_points.end());
                // 将当前层的关键点云添加到线程本地集合
                local_key_points_thread.insert(local_key_points_thread.end(), key_points.begin(), key_points.end());
            }
        }
        
        // 线程安全地合并结果到全局变量
        std::lock_guard<std::mutex> lock(mutex); // 获取互斥锁
        // 将线程本地的下采样点云合并到全局结果
        Downsample_PointCloud.insert(Downsample_PointCloud.end(), local_points_thread.begin(), local_points_thread.end());
        // 将线程本地的关键点云合并到全局结果
		Key_PointCloud.insert(Key_PointCloud.end(), local_key_points_thread.begin(), local_key_points_thread.end());
    });
    return Downsample_PointCloud; // 返回下采样后的点云数据
}

void clear(){
	Seg_points.clear();
}
};

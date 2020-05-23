#pragma once
#if !defined(__CUDACC__)
#include <eigen3/Eigen/Core>
#endif
#include <vector_types.h>
struct mat33
{
	mat33()
	{
	}

#if !defined(__CUDACC__)
	mat33(Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& e)
	{
		memcpy(data, e.data(), sizeof(mat33));
	}
#endif

	float3 data[3];
};

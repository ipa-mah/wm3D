/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef KERNEL_CONTAINERS_HPP_
#define KERNEL_CONTAINERS_HPP_

#include <cstddef>
#include <cuda/common/common.hpp>
/*
#if defined(__CUDACC__)
	#define GPU_HOST_DEVICE__ __host__ __device__ __forceinline__
##else
	#define GPU_HOST_DEVICE__
#endif
*/
template <typename T>
struct DevPtr
{
	typedef T elem_type;
	const static size_t elem_size = sizeof(elem_type);

	T* data;

	__HOSTDEVICE__ DevPtr() : data(0)
	{
	}
	__HOSTDEVICE__ DevPtr(T* data_arg) : data(data_arg)
	{
	}

	__HOSTDEVICE__ size_t elemSize() const
	{
		return elem_size;
	}
	__HOSTDEVICE__ operator T*()
	{
		return data;
	}
	__HOSTDEVICE__ operator const T*() const
	{
		return data;
	}
};

template <typename T>
struct PtrSz : public DevPtr<T>
{
	__HOSTDEVICE__ PtrSz() : size(0)
	{
	}
	__HOSTDEVICE__ PtrSz(T* data_arg, size_t size_arg) : DevPtr<T>(data_arg), size(size_arg)
	{
	}

	size_t size;
};

template <typename T>
struct PtrStep : public DevPtr<T>
{
	__HOSTDEVICE__ PtrStep() : step(0)
	{
	}
	__HOSTDEVICE__ PtrStep(T* data_arg, size_t step_arg) : DevPtr<T>(data_arg), step(step_arg)
	{
	}

	/** \brief stride between two consecutive rows in bytes. Step is stored always and everywhere in bytes!!! */
	size_t step;

	__HOSTDEVICE__ T* ptr(int y = 0)
	{
		return (T*)((char*)DevPtr<T>::data + y * step);
	}
	__HOSTDEVICE__ const T* ptr(int y = 0) const
	{
		return (const T*)((const char*)DevPtr<T>::data + y * step);
	}

	__HOSTDEVICE__ T& operator()(int y, int x)
	{
		return ptr(y)[x];
	}
	__HOSTDEVICE__ const T& operator()(int y, int x) const
	{
		return ptr(y)[x];
	}
};

template <typename T>
struct PtrStepSz : public PtrStep<T>
{
	__HOSTDEVICE__ PtrStepSz() : cols(0), rows(0)
	{
	}
	__HOSTDEVICE__ PtrStepSz(int rows_arg, int cols_arg, T* data_arg, size_t step_arg) : PtrStep<T>(data_arg, step_arg), cols(cols_arg), rows(rows_arg)
	{
	}

	int cols;
	int rows;
};

#endif /* KERNEL_CONTAINERS_HPP_ */

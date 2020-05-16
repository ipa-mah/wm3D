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

 #pragma once 

#include "wm3D/cuda/data_types.cuh"

#define DIVISOR 32767

#define COMPOUND_VEC3_OP(type, scalar, op)                                          \
	__device__ __host__ __forceinline__ type& operator op(type& v1, const type& v2) \
	{                                                                               \
		v1.x op v2.x;                                                               \
		v1.y op v2.y;                                                               \
		v1.z op v2.z;                                                               \
		return v1;                                                                  \
	}                                                                               \
	__device__ __host__ __forceinline__ type& operator op(type& v, scalar val)      \
	{                                                                               \
		v.x op val;                                                                 \
		v.y op val;                                                                 \
		v.z op val;                                                                 \
		return v;                                                                   \
	}

COMPOUND_VEC3_OP(float3, float, -=)
COMPOUND_VEC3_OP(float3, float, +=)
COMPOUND_VEC3_OP(float3, float, *=)
COMPOUND_VEC3_OP(float3, float, /=)

COMPOUND_VEC3_OP(short3, short, -=)

COMPOUND_VEC3_OP(int3, int, +=)

#undef COMPOUND_VEC3_OP

static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }

__device__ __host__ __forceinline__ float dot(const float3& v1, const float3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __host__ __forceinline__ float3 cross(const float3& v1, const float3& v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

#define VEC_BINOP(type, scalar, op, cop)                                                 \
	__device__ __host__ __forceinline__ type operator op(const type& v1, const type& v2) \
	{                                                                                    \
		type r = v1;                                                                     \
		r cop v2;                                                                        \
		return r;                                                                        \
	}                                                                                    \
	__device__ __host__ __forceinline__ type operator op(const type& v1, scalar c)       \
	{                                                                                    \
		type r = v1;                                                                     \
		r cop c;                                                                         \
		return r;                                                                        \
	}

VEC_BINOP(float3, float, -, -=)
VEC_BINOP(float3, float, +, +=)
VEC_BINOP(float3, float, *, *=)
VEC_BINOP(float3, float, /, /=)

VEC_BINOP(short3, short, -, -=)

VEC_BINOP(int3, int, +, +=)

#undef VEC_BINOP

template <typename T>
__device__ __host__ __forceinline__ float norm(const T& val)
{
	return sqrtf(dot(val, val));
}

template <typename T>
__host__ __device__ __forceinline__ float inverse_norm(const T& v)
{
	return rsqrtf(dot(v, v));
}

template <typename T>
__host__ __device__ __forceinline__ T normalized(const T& v)
{
	return v * inverse_norm(v);
}

template <typename T>
__host__ __device__ __forceinline__ T normalized_safe(const T& v)
{
	return (dot(v, v) > 0) ? (v * rsqrtf(dot(v, v))) : v;
}

__device__ __forceinline__ float3 operator*(const mat33& m, const float3& vec)
{
	return make_float3(dot(m.data[0], vec), dot(m.data[1], vec), dot(m.data[2], vec));
}

// tsdf computation
__device__ __forceinline__ short2* TsdfVolume::operator()(int x, int y, int z)
{
	return data + x + y * dims.x + z * dims.y * dims.x;
}
__device__ __forceinline__ const short2* TsdfVolume::operator()(int x, int y, int z) const 
{
	return data + x + y * dims.x + z * dims.y * dims.x;
}
__device__ __forceinline__ short2* TsdfVolume::beg(int x, int y) const
{
	return data + x + dims.x * y;
}
__device__ __forceinline__ short2* TsdfVolume::zstep(short2 *const ptr) const
{
	return ptr + dims.x * dims.y;
}

__device__ __forceinline__ void clear_voxel(uchar4& value)
{
	value = make_uchar4(0, 0, 0, 0);
}
__device__ __forceinline__ void clear_voxel(float& value)
{
	value = 0.0f;
}
__device__ __forceinline__ void clear_voxel(short& value)
{
	value = max(-DIVISOR, min(DIVISOR, __float2int_rz(0 * DIVISOR)));
}
////



__device__ __forceinline__ short2 Gpu::pack_tsdf(float tsdf, int weight)
{
	int fixedp = max (-DIVISOR, min (DIVISOR, __float2int_rz (tsdf * DIVISOR)));
	return make_short2 (fixedp, weight);
}


__device__ __forceinline__ float Gpu::unpack_tsdf(short2 value, int& weight)
{
	weight = value.y;
  	return __int2float_rn (value.x) / DIVISOR;
}

__device__ __forceinline__ float Gpu::unpack_tsdf(short2 value)
{
	return static_cast<float>(value.x) / DIVISOR;
}

template <class T>
__device__ __host__ __forceinline__ void swap(T& a, T& b)
{
	T c(a);
	a = b;
	b = c;
}


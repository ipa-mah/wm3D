#pragma once
#include <cmath>
#if defined(__CUDACC__)
#define __ALIGN__(n) __align__(n)
/* Use these to avoid redundant conditional macro code ONLY in headers */
#define __HOST__ __host__
#define __DEVICE__ __device__
#define __HOSTDEVICE__ __host__ __device__
#define __GLOBAL__ __global__
#else
#define __HOST__
#define __DEVICE__
#define __HOSTDEVICE__
#define __GLOBAL__
#define __ALIGN__(n) alignas(n)
#define __int_as_float(n) float(int(n))
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif
#ifndef ushort
typedef unsigned short ushort;
#endif
#ifndef uint
typedef unsigned int uinit;
#endif

#define WM3D_MIN(a, b) (a < b ? a : b)
#define WM3D_MAX(a, b) (b < a ? a : b)

#define THREAD_3D_UNIT 8
#define THREAD_2D_UNIT 16
#define THREAD_1D_UNIT 256
#define DIV_CEILING(a, b) ((a + b - 1) / b)

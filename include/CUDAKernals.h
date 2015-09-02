#ifndef HELLOCUDA_H
#define HELLOCUDA_H
//----------------------------------------------------------------------------------------------------------------------
/// @file CudaSPHKernals.h
/// @author Declan Russell
/// @date 28/08/2015
/// @version 1.0
/// @brief Used for prototyping our CUDA kernals to be used in our C++ application.
/// @brief This file is linked with CUDAKernals.cu by gcc after CUDAKernals.cu has been
/// @brief compiled by nvcc.
//----------------------------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

//----------------------------------------------------------------------------------------------------------------------
/// @brief structure for some particle data
//----------------------------------------------------------------------------------------------------------------------
struct particles
{
    int idx;
    int numN;
    int nIdx[4];
    float3 acc;
    float3 oldP;
};
//----------------------------------------------------------------------------------------------------------------------
/// @brief Our cloth solver function.
/// @param _stream - the cuda stream to run our kernal
/// @param _posBuffer - buffer of our verticies positions
/// @param _particleBuffer - buffer of our particle information
/// @param _numParticles - the number of particles to solve for
/// @param _numNeighbours - the number of neighbours our particles have
/// @param _restLength - the rest length between particles
/// @param _mass - the mass of our particles
/// @param _timeStep - the timestep to increment our simulation
/// @param _maxNumThreads - the maximum number of threads our device has per block
//----------------------------------------------------------------------------------------------------------------------
void clothSolver(cudaStream_t _stream, float3 * _posBuffer, particles* _particleBuffer, int _numParticles, int _numNeighbours, float _restLength, float _mass, float _timeStep, int _maxNumThreads);
//----------------------------------------------------------------------------------------------------------------------
/// @brief resets the position of our constrained particles
/// @param _posBuffer - buffer of our verticies positions
/// @param _particleBuffer - buffer of our particle information
/// @param _numParticles - the number of particles to solve for
/// @param _maxNumThreads - the maximum number of threads our device has per block
//----------------------------------------------------------------------------------------------------------------------
void resetConstParticles(float3 * _posBuffer, particles* _particleBuffer,int _numParticles, int _maxNumThreads);
//----------------------------------------------------------------------------------------------------------------------




#endif // HELLOCUDA_H

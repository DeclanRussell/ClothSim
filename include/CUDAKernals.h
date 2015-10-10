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
#include <eigen-nvcc/Eigen>

//----------------------------------------------------------------------------------------------------------------------
/// @brief structure for the contraints data
//----------------------------------------------------------------------------------------------------------------------
struct constraints
{
    int particleA,particleB;
};
//----------------------------------------------------------------------------------------------------------------------
/// @brief Our cloth verlet intergration solver function.
/// @param _stream - the cuda stream to run our kernal
/// @param _posBuffer - buffer of our verticies positions
/// @param _oldPosBuffer - buffer to store old positions of particles
/// @param _numParticles - the number of particles to solve for
/// @param _mass - the mass of our particles
/// @param _timeStep - the timestep to increment our simulation
/// @param _maxNumThreads - the maximum number of threads our device has per block
//----------------------------------------------------------------------------------------------------------------------
void clothVerletIntegration(cudaStream_t _stream, float3 * _posBuffer, float3 *_oldPosBuffer, int _numParticles, float _mass, float _timeStep, int _maxNumThreads);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Our cloth constraint solver function.
/// @param _stream - the cuda stream to run our kernal
/// @param _posBuffer - buffer of our verticies positions
/// @param _constraintsBuffer - buffer of our contraints information
/// @param _numConstraints - the number of constraints to solve for
/// @param _restLength - the rest length between particles
//----------------------------------------------------------------------------------------------------------------------
void clothConstraintSolver(cudaStream_t _stream, float3 * _posBuffer, constraints* _constraintsBuffer, int _numConstraints, float _restLength, int _maxNumThreads);
//----------------------------------------------------------------------------------------------------------------------
/// @brief resets the position of our constrained particles
/// @param _stream - the cuda stream to run our kernal
/// @param _posBuffer - buffer of our verticies positions
/// @param _oldPosBuffer - buffer of our old particle positions
/// @param _fixedParticles - buffer of the fixed particle indicies
/// @param _numParticles - the number of particles to solve for
/// @param _maxNumThreads - the maximum number of threads our device has per block
//----------------------------------------------------------------------------------------------------------------------
void resetFixedParticles(cudaStream_t _stream, float3 * _posBuffer, float3* _oldPosBuffer, int* _fixedParticles,int _numParticles, int _maxNumThreads);
//----------------------------------------------------------------------------------------------------------------------
/// @brief tests intersection with ray and points in our mesh
/// @param _stream - the cuda stream to run our kernal
/// @param _posBuffer - buffer of our verticies positions
/// @param _idBuffer - buffer to store id of intesected vertex
/// @param _from - position our ray is from
/// @param _ray - direction of the ray
/// @param _raidus - radius of points to intersect
/// @param _modelMatrix - model matrix of scene
/// @param _numParticles - the number of particles to solve for
/// @param _maxNumThreads - the maximum number of threads our device has per block
//----------------------------------------------------------------------------------------------------------------------
void testIntersect(cudaStream_t _stream, float3 * _posBuffer, int *_idBuffer, Eigen::Vector3f _from, Eigen::Vector3f _ray, float _radius, Eigen::Matrix4f _modelMatrix, int _numParticles, int _maxNumThreads);
//----------------------------------------------------------------------------------------------------------------------




#endif // HELLOCUDA_H

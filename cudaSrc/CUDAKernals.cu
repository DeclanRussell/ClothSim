//----------------------------------------------------------------------------------------------------------------------
/// @file CudaSPHKernals.cu
/// @author Declan Russell
/// @date 28/08/2015
/// @version 1.0
//----------------------------------------------------------------------------------------------------------------------
#include "CUDAKernals.h"
#include "helper_math.h"  //< some math operations with cuda types
#include <iostream>

#define gravity make_float3(0.f,-9.8f,0.f)

//----------------------------------------------------------------------------------------------------------------------
/// @brief verlet intergration solver kernal
//----------------------------------------------------------------------------------------------------------------------
__global__ void verletIntSolverKernal(float3 *_posBuffer, float3 *_oldPosBuffer, int _numParticles, float _mass, float _timeStep)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        float3 pos = _posBuffer[idx];
        float3 temp = pos;
        float3 oldPos = _oldPosBuffer[idx];
        //printf("Verlet Pos %f,%f,%f \n",pos.x,pos.y,pos.z);
        pos +=  pos-oldPos + gravity*_timeStep*_timeStep;
        _oldPosBuffer[idx] = temp;
        _posBuffer[idx] = pos;
    }
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief our cloth constraint solver kernal
//----------------------------------------------------------------------------------------------------------------------
__global__ void constraintSolverKernal(float3* _posBuffer, constraints* _constraintsBuffer, float _restLength, int _numConstraints)
{
    // Compute our thread idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numConstraints)
    {
        // Get particles positions.
        // This usees a lot of accesses to global memory which is slow. It would be nice if I could think of a better way
        // to deal with the banking conflicts :(
        constraints c = _constraintsBuffer[idx];
        float3 partA = _posBuffer[c.particleA];
        float3 partB = _posBuffer[c.particleB];

        // Satisfy the constrainsts
        float3 delta;
        float deltaLength,diff;

        delta = partB - partA;
        deltaLength = length(delta);
        diff = (deltaLength-_restLength)/deltaLength;


        delta*=0.5f*diff;
        //delta *= _restLength*_restLength/(delta*delta+_restLength*_restLength)-.5f;


        _posBuffer[c.particleA] = partA+delta;
        _posBuffer[c.particleB] = partB-delta;

    }

}
//----------------------------------------------------------------------------------------------------------------------
/// @brief kernal to move the fixed particles back to their old locations
//----------------------------------------------------------------------------------------------------------------------
__global__ void resetConstPartKernal(float3 *_posBuffer, float3 *_oldPosBuffer, int *_fixedParticlesBuffer,int _numFixedPoints)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numFixedPoints)
    {
        // Get particle and its position.
        int partIdx = _fixedParticlesBuffer[threadIdx.x + blockIdx.x * blockDim.x];
        _posBuffer[partIdx] = _oldPosBuffer[partIdx];
    }
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief kernal to test intersection of ray and particles
//----------------------------------------------------------------------------------------------------------------------
__global__ void testIntersectKernal(float3 *_posBuffer, int *_idBuffer, Eigen::Vector3f _from, Eigen::Vector3f _ray, float _radius, Eigen::Matrix4f _modelMatrix, int _numParticles)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        float3 partPos = _posBuffer[idx];

        Eigen::Vector4f pPos4 = Eigen::Vector4f(partPos.x,partPos.y,partPos.z,1.0);
        pPos4 = _modelMatrix*pPos4;
        Eigen::Vector3f pPos3 = Eigen::Vector3f(pPos4[0],pPos4[1],pPos4[2]);

        float b = _ray.dot(_from-pPos3);
        float c = (_from-pPos3).dot(_from-pPos3) - (_radius*_radius);
        if(b*b-c>=0)
        {
            printf("hit vert %d",idx);
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
void clothVerletIntegration(cudaStream_t _stream, float3 *_posBuffer, float3 *_oldPosBuffer, int _numParticles, float _mass, float _timeStep, int _maxNumThreads)
{
    if(_numParticles>_maxNumThreads)
    {
        //calculate how many blocks we want
        int blocks = ceil(_numParticles/_maxNumThreads)+1;
        verletIntSolverKernal<<<blocks,_maxNumThreads,0,_stream>>>(_posBuffer, _oldPosBuffer, _numParticles, _mass, _timeStep);
    }
    else
    {
        verletIntSolverKernal<<<1,_numParticles,0,_stream>>>(_posBuffer, _oldPosBuffer, _numParticles, _mass, _timeStep);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void clothConstraintSolver(cudaStream_t _stream, float3 *_posBuffer, constraints *_constraintsBuffer, int _numConstraints, float _restLength, int _maxNumThreads)
{
    if(_numConstraints>_maxNumThreads)
    {
        //calculate how many blocks we want
        int blocks = ceil(_numConstraints/_maxNumThreads)+1;
        constraintSolverKernal<<<blocks,_maxNumThreads,0,_stream>>>(_posBuffer,_constraintsBuffer, _restLength, _numConstraints);
    }
    else
    {
        constraintSolverKernal<<<1,_numConstraints,0,_stream>>>(_posBuffer,_constraintsBuffer, _restLength, _numConstraints);
    }

}
//----------------------------------------------------------------------------------------------------------------------
void resetFixedParticles(cudaStream_t _stream, float3 *_posBuffer, float3 *_oldPosBuffer, int *_fixedParticles, int _numParticles, int _maxNumThreads)
{
    if(_numParticles>_maxNumThreads)
    {
        //calculate how many blocks we want
        int blocks = ceil(_numParticles/_maxNumThreads)+1;
        resetConstPartKernal<<<blocks,_maxNumThreads,0,_stream>>>(_posBuffer,_oldPosBuffer,_fixedParticles,_numParticles);
    }
    else
    {
        resetConstPartKernal<<<1,_numParticles,0,_stream>>>(_posBuffer,_oldPosBuffer,_fixedParticles,_numParticles);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void testIntersect(cudaStream_t _stream, float3 *_posBuffer, int *_idBuffer, Eigen::Vector3f _from, Eigen::Vector3f _ray, float _radius, Eigen::Matrix4f _modelMatrix, int _numParticles, int _maxNumThreads)
{
    if(_numParticles>_maxNumThreads)
    {
        //calculate how many blocks we want
        int blocks = ceil(_numParticles/_maxNumThreads)+1;
        testIntersectKernal<<<blocks,_maxNumThreads,0,_stream>>>(_posBuffer, _idBuffer, _from, _ray, _radius, _modelMatrix, _numParticles);
    }
    else
    {
        testIntersectKernal<<<1,_numParticles,0,_stream>>>(_posBuffer, _idBuffer, _from, _ray, _radius, _modelMatrix, _numParticles);
    }
}
//----------------------------------------------------------------------------------------------------------------------

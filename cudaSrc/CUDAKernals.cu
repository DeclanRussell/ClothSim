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
        constraints c = _constraintsBuffer[idx];
        float3 partA = _posBuffer[c.particleA];
        float3 partB = _posBuffer[c.particleB];

        // Satisfy the constrainsts
        float3 delta;
        float diff;

        delta = partB - partA;
        //deltaLength = length(delta);
        //diff = (deltaLength-_restLength)/deltaLength;

        float d = (delta.x*delta.x)+(delta.y*delta.y)+(delta.z*delta.z);
        diff = (d - (_restLength*_restLength))/((_restLength*_restLength) + d);

        delta*=0.5f*diff;
        //delta *= _restLength*_restLength/(delta*delta+_restLength*_restLength)-.5f;
        //printf("delta length %f delta %f,%f,%f A %d B %d\n",deltaLength,delta.x,delta.y,delta.z,c.particleA,c.particleB);

        _posBuffer[c.particleA] = partA+delta;
        _posBuffer[c.particleB] = partB-delta;
    }

}
//----------------------------------------------------------------------------------------------------------------------
/// @brief kernal to move the fixed particles back to their old locations
//----------------------------------------------------------------------------------------------------------------------
__global__ void resetConstPartKernal(float3 *_posBuffer, float3 *_oldPosBuffer, int *_fixedParticlesBuffer,int _numParticles)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        // Get particle and its position.
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(_fixedParticlesBuffer[idx]==1)
        {
            _posBuffer[idx] = _oldPosBuffer[idx];
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief device function to test intesection of ray and particles
/// @brief returns true or false if particle has been selected
//----------------------------------------------------------------------------------------------------------------------
__device__ bool testIntersection(int _idx,float3 *_posBuffer, Eigen::Vector3f _from, Eigen::Vector3f _ray, float _radius, Eigen::Matrix4f _V)
{
    float3 partPos = _posBuffer[_idx];

    Eigen::Vector4f pPos4 = Eigen::Vector4f(partPos.x,partPos.y,partPos.z,1.0);
    pPos4 = _V*pPos4;
    Eigen::Vector3f pPos3 = Eigen::Vector3f(pPos4[0],pPos4[1],pPos4[2]);

    float b = _ray.dot(_from-pPos3);
    float c = (_from-pPos3).dot(_from-pPos3) - (_radius*_radius);
    float t = b*b-c;
    if(t>=0)
    {
        printf("idx %d pos %f,%f,%f\n",_idx,pPos3[0],pPos3[1],pPos3[2]);
        return true;
    }
    else
    {
        return false;
    }
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief kernal to fix new selected particles
//----------------------------------------------------------------------------------------------------------------------
__global__ void fixParticlesKernal(float3 *_posBuffer, int *_fixVertsBuffer, Eigen::Vector3f _from, Eigen::Vector3f _ray, float _radius, Eigen::Matrix4f _V, int _numParticles)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        bool selected = testIntersection(idx,_posBuffer,_from,_ray,_radius,_V);
        if(selected)
        {
//            printf("idx %d\n",idx);
            _fixVertsBuffer[idx] = 1;
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief kernal to fix new selected particles
//----------------------------------------------------------------------------------------------------------------------
__global__ void unFixParticlesKernal(float3 *_posBuffer, int *_fixVertsBuffer, Eigen::Vector3f _from, Eigen::Vector3f _ray, float _radius, Eigen::Matrix4f _V, int _numParticles)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        bool selected = testIntersection(idx,_posBuffer,_from,_ray,_radius,_V);
        if(selected)
        {
//            printf("idx %d\n",idx);
            _fixVertsBuffer[idx] = 0;
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief kernal to fix and set the currently selected move vertex
//----------------------------------------------------------------------------------------------------------------------
__global__ void setMoveVertexKernal(float3 *_posBuffer, int *_fixVertsBuffer, int *_moveVertex, Eigen::Vector3f _from, Eigen::Vector3f _ray, float _radius, Eigen::Matrix4f _V, int _numParticles)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        bool selected = testIntersection(idx,_posBuffer,_from,_ray,_radius,_V);
        if(selected)
        {
//            printf("idx %d\n",idx);
            _fixVertsBuffer[idx] = 1;
            _moveVertex[0] = idx;
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief unselectes the currently selected move vertex
//----------------------------------------------------------------------------------------------------------------------
__global__ void unselectMoveVertexKernal( int *_fixVertsBuffer, int *_moveVertex)
{
    int idx = _moveVertex[0];
    if(idx!=-1)
    {
        _fixVertsBuffer[idx] = 0;
        _moveVertex[0] = -1;
    }
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief moves the currently selected move vertex
//----------------------------------------------------------------------------------------------------------------------
__global__ void moveVertexKernal(float3 *_posBuffer, int *_moveVertex, float3 _dir)
{
    int idx = _moveVertex[0];
    if(idx!=-1)
    {
        _posBuffer[idx]+=_dir;
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
void fixParticles(cudaStream_t _stream, float3 *_posBuffer, int *_fixedBuffer, Eigen::Vector3f _from, Eigen::Vector3f _ray, float _radius, Eigen::Matrix4f _V, int _numParticles, int _maxNumThreads)
{
    if(_numParticles>_maxNumThreads)
    {
        //calculate how many blocks we want
        int blocks = ceil(_numParticles/_maxNumThreads)+1;
        fixParticlesKernal<<<blocks,_maxNumThreads,0,_stream>>>(_posBuffer, _fixedBuffer, _from, _ray, _radius, _V, _numParticles);
    }
    else
    {
        fixParticlesKernal<<<1,_numParticles,0,_stream>>>(_posBuffer, _fixedBuffer, _from, _ray, _radius, _V, _numParticles);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void unFixParticles(cudaStream_t _stream, float3 * _posBuffer, int *_fixedBuffer, Eigen::Vector3f _from, Eigen::Vector3f _ray, float _radius, Eigen::Matrix4f _V, int _numParticles, int _maxNumThreads)
{
    if(_numParticles>_maxNumThreads)
    {
        //calculate how many blocks we want
        int blocks = ceil(_numParticles/_maxNumThreads)+1;
        unFixParticlesKernal<<<blocks,_maxNumThreads,0,_stream>>>(_posBuffer, _fixedBuffer, _from, _ray, _radius, _V, _numParticles);
    }
    else
    {
        unFixParticlesKernal<<<1,_numParticles,0,_stream>>>(_posBuffer, _fixedBuffer, _from, _ray, _radius, _V, _numParticles);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void setMoveParticles(cudaStream_t _stream, float3 * _posBuffer, int *_fixedBuffer,int *_moveVertex ,Eigen::Vector3f _from, Eigen::Vector3f _ray, float _radius, Eigen::Matrix4f _V, int _numParticles, int _maxNumThreads)
{
    if(_numParticles>_maxNumThreads)
    {
        //calculate how many blocks we want
        int blocks = ceil(_numParticles/_maxNumThreads)+1;
        setMoveVertexKernal<<<blocks,_maxNumThreads,0,_stream>>>(_posBuffer, _fixedBuffer, _moveVertex, _from, _ray, _radius, _V, _numParticles);
    }
    else
    {
        setMoveVertexKernal<<<1,_numParticles,0,_stream>>>(_posBuffer, _fixedBuffer, _moveVertex, _from, _ray, _radius, _V, _numParticles);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void unSelectMoveParticle(cudaStream_t _stream, int *_fixedBuffer,int *_moveVertex)
{
    unselectMoveVertexKernal<<<1,1,0,_stream>>>(_fixedBuffer,_moveVertex);
}
//----------------------------------------------------------------------------------------------------------------------
void moveSelectedParticle(cudaStream_t _stream,float3 *_posBuffer, int *_moveVertex, float3 _dir)
{
    moveVertexKernal<<<1,1,0,_stream>>>(_posBuffer,_moveVertex,_dir);
}
//----------------------------------------------------------------------------------------------------------------------

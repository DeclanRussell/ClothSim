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
/// @brief our cloth solver kernal
//----------------------------------------------------------------------------------------------------------------------
__global__ void clothSolverKernal(float3* _posBuffer, particles* _partBuffer, float _restLength, float _mass, int _numN, float _timeStep)
{
    // Get particle and its position.
    particles part = _partBuffer[threadIdx.x];
    float3 pos = _posBuffer[part.idx];

    // Verlet integeration
    part.oldP = pos;
    pos +=  pos-part.oldP + gravity*_timeStep*_timeStep;


    // Satisfy the constrainsts
    // This usees a lot of accesses to global memory which is slow. It would be nice if I could think of a better way
    // to deal with the banking conflicts :(
    float3 nPos,delta;
    float deltaLength,diff;
    for(int i=0; i<_numN; i++)
    {
        nPos = _posBuffer[part.nIdx[i]];
        //printf("nPos %f,%f,%f\n",nPos.x,nPos.y,nPos.z);
        delta = nPos - pos;
        deltaLength = length(delta*delta);
        if(deltaLength!=deltaLength)
        {
            printf("fuck fuck fuck\n");
            //printf("idx %d nIdx[%d] %d numN %d\n",part.idx,i,part.nIdx[i],_numN);
            printf("Pos %f,%f,%f nPos %f,%f,%f idx %d\n",pos.x,pos.y,pos.z,nPos.x,nPos.y,nPos.z,part.nIdx[i]);
        }
        diff = (deltaLength-_restLength)/deltaLength;
        delta*=.5f*diff;
        //delta *= _restLength*_restLength/(delta*delta+_restLength*_restLength)-.5f;
        pos+=delta;
        //_posBuffer[part.nIdx[i]] = nPos-delta;
        //printf("Delta %f,%f,%f \n",delta.x,delta.y,delta.z);
    }
    //printf("Pos %f,%f,%f \n",pos.x,pos.y,pos.z);
   _posBuffer[part.idx] = pos;

}
//----------------------------------------------------------------------------------------------------------------------
__global__ void resetConstPartKernal(float3 *_posBuffer, particles *_partBuffer)
{
    // Get particle and its position.
    particles part = _partBuffer[threadIdx.x];
    _posBuffer[part.idx] = part.oldP;
}

//----------------------------------------------------------------------------------------------------------------------
void clothSolver(cudaStream_t _stream, float3 *_posBuffer, particles *_particleBuffer, int _numParticles, int _numNeighbours, float _restLength, float _mass, float _timeStep, int _maxNumThreads)
{
    if(_numParticles>_maxNumThreads)
    {
        //calculate how many blocks we want
        int blocks = ceil(_numParticles/_maxNumThreads)+1;
        clothSolverKernal<<<blocks,_maxNumThreads,0,_stream>>>(_posBuffer,_particleBuffer, _restLength, _mass,_numNeighbours,_timeStep);
    }
    else
    {
        clothSolverKernal<<<1,_numParticles,0,_stream>>>(_posBuffer,_particleBuffer, _restLength, _mass,_numNeighbours,_timeStep);
    }

}
//----------------------------------------------------------------------------------------------------------------------
void resetConstParticles(float3 *_posBuffer, particles *_particleBuffer, int _numParticles, int _maxNumThreads)
{
    if(_numParticles>_maxNumThreads)
    {
        //calculate how many blocks we want
        int blocks = ceil(_numParticles/_maxNumThreads)+1;
        resetConstPartKernal<<<blocks,_maxNumThreads>>>(_posBuffer,_particleBuffer);
    }
    else
    {
        resetConstPartKernal<<<1,_numParticles>>>(_posBuffer,_particleBuffer);
    }
}

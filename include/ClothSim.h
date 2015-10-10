#ifndef CLOTHSIM_H
#define CLOTHSIM_H

//----------------------------------------------------------------------------------------------------------------------
/// @file ClothSim.h
/// @class ClothSim
/// @author Declan Russell
/// @date 28/08/2015
/// @version 1.0
/// @brief Class for managing our CUDA accelerated cloth simulation
//----------------------------------------------------------------------------------------------------------------------

#ifdef DARWIN
    #include <GLFW/glfw3.h>
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #include <GL/gl.h>
#endif

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include <QString>
#include "ShaderProgram.h"
#include "CUDAKernals.h"


class ClothSim
{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief constructor for a basic plane of cloth
    /// @param _width - width of our plane
    /// @param _height - height of our plane
    //----------------------------------------------------------------------------------------------------------------------
    ClothSim(int _width, int _height);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~ClothSim();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief draw function for some standard phong drawing
    //----------------------------------------------------------------------------------------------------------------------
    void draw(glm::mat4 _MV, glm::mat4 _MVP, glm::mat3 _normalMat);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief updates our simulation a given time step
    /// @param _timeStep - timeStep to update our simulation
    //----------------------------------------------------------------------------------------------------------------------
    void update(float _timeStep);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to our VAO incase we want to draw it with something else
    //----------------------------------------------------------------------------------------------------------------------
    inline GLuint getVAO(){return m_VAO;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set the texture on our geometry
    /// @param _loc - location of desired texture
    //----------------------------------------------------------------------------------------------------------------------
    void setTexture(QString _loc);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator for our rest length between our particles
    /// @param _len - desired rest length
    //----------------------------------------------------------------------------------------------------------------------
    inline void setRestLength(float _len){m_restLength = _len;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief resets our simulation
    //----------------------------------------------------------------------------------------------------------------------
    void reset();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief sets phong to be the active shader
    //----------------------------------------------------------------------------------------------------------------------
    void usePhongShader();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief sets our cloth shader to be the active shader
    //----------------------------------------------------------------------------------------------------------------------
    void useClothShader();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief fix new points of the cloth
    //----------------------------------------------------------------------------------------------------------------------
    void fixNewPoints(glm::vec3 _from, glm::vec3 _ray, glm::mat4 _modelMatrix);
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief function to create a plane of specified size
    /// @param _width - width of the plane
    /// @param _height - height of the plane
    //----------------------------------------------------------------------------------------------------------------------
    void createPlane(int _width, int _height);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief creates a basic phong shader for our geometry
    //----------------------------------------------------------------------------------------------------------------------
    void createPhongShader();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief creates our more detailed cloth shader
    //----------------------------------------------------------------------------------------------------------------------
    void createClothShader();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our basic constructor that we dont want anyone to use
    //----------------------------------------------------------------------------------------------------------------------
    ClothSim(){}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief bool to indicate if we want to fix more points
    //----------------------------------------------------------------------------------------------------------------------
    bool m_fixNewPoints;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief intersection ray origin
    //----------------------------------------------------------------------------------------------------------------------
    glm::vec3 m_rayOrigin;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief intersection ray direction
    //----------------------------------------------------------------------------------------------------------------------
    glm::vec3 m_rayDir;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief model matrix of scene
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_modelMatrix;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief verticies radius;
    //----------------------------------------------------------------------------------------------------------------------
    float m_vertRadius;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief handle to our VAO
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VAO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief handle to our verts VBO
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VBOverts;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief handle to our normals VBO
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VBOnorms;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief handle to our indicies VBO
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VBOidc;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief handle to our texture coords VBO
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VBOtexCoords;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief handle to the VBO that tells us if the vertex is fixed or not
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VBOFixedVerts;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the number of threads our device has per block
    //----------------------------------------------------------------------------------------------------------------------
    int m_threadsPerBlock;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our vertex cuda graphics resource. Used for OpenGL interop
    //----------------------------------------------------------------------------------------------------------------------
    cudaGraphicsResource_t m_resourceVerts;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our normals cuda graphics resource. Used for OpenGL interop
    //----------------------------------------------------------------------------------------------------------------------
    cudaGraphicsResource_t m_resourceNorms;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief number of particles in cloth
    //----------------------------------------------------------------------------------------------------------------------
    int m_numParticles;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief buffer for intersected particle id's
    //----------------------------------------------------------------------------------------------------------------------
    int* d_intersectIds;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our old particle positions
    //----------------------------------------------------------------------------------------------------------------------
    float3* d_oldParticlePos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief structure to hold our constraint buffer properties
    //----------------------------------------------------------------------------------------------------------------------
    struct constProps{
        constraints* ptr;
        int numConsts;
    };
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointers to our device buffers that contain our constraints information
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<constProps> d_constraintBuffers;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our device buffer of fixed particle indexes
    //----------------------------------------------------------------------------------------------------------------------
    int* d_fixedPartBuffer;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief number of fixed particles in our simulation
    //----------------------------------------------------------------------------------------------------------------------
    int m_numFixedParticles;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our CUDA stream to help run kernals concurrently
    //----------------------------------------------------------------------------------------------------------------------
    cudaStream_t m_cudaStream;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our basic phong shader program
    //----------------------------------------------------------------------------------------------------------------------
    ShaderProgram *m_phongShaderProgram;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our more detailed cloth shader program
    //----------------------------------------------------------------------------------------------------------------------
    ShaderProgram *m_clothShaderProgram;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our active shader program
    //----------------------------------------------------------------------------------------------------------------------
    ShaderProgram *m_activeShaderProgram;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief location of our MV matrix in our shader program
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_MVLoc;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief location of our MVP matrix in our shader program
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_MVPLoc;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief location of our normal matrix in our shader program
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_normMatLoc;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the number of verticies in our geometry
    //----------------------------------------------------------------------------------------------------------------------
    int m_numVerts;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the width of our grid
    //----------------------------------------------------------------------------------------------------------------------
    int m_width;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the height of our grid
    //----------------------------------------------------------------------------------------------------------------------
    int m_height;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the rest between our particles
    //----------------------------------------------------------------------------------------------------------------------
    float m_restLength;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the mass of our particles
    //----------------------------------------------------------------------------------------------------------------------
    float m_mass;
    //----------------------------------------------------------------------------------------------------------------------
};

#endif // CLOTHSIM_H

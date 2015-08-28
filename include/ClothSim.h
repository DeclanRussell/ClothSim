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
    void draw();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to our VAO incase we want to draw it with something else
    //----------------------------------------------------------------------------------------------------------------------
    inline GLuint getVAO(){return m_VAO;}
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief function to create a plane of specified size
    /// @param _width - width of the plane
    /// @param _height - height of the plane
    //----------------------------------------------------------------------------------------------------------------------
    void createPlane(int _width, int _height);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our basic constructor that we dont want anyone to use
    //----------------------------------------------------------------------------------------------------------------------
    ClothSim(){}
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
    /// @brief our vertex cuda graphics resource. Used for OpenGL interop
    //----------------------------------------------------------------------------------------------------------------------
    cudaGraphicsResource_t m_resourceVerts;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our normals cuda graphics resource. Used for OpenGL interop
    //----------------------------------------------------------------------------------------------------------------------
    cudaGraphicsResource_t m_resourceNorms;
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
};

#endif // CLOTHSIM_H

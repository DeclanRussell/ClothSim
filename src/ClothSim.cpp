#include "ClothSim.h"

#include <vector>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

//----------------------------------------------------------------------------------------------------------------------
ClothSim::ClothSim(int _width, int _height)
{
    //create our plane
    createPlane(_width,_height);
}
//----------------------------------------------------------------------------------------------------------------------
ClothSim::~ClothSim()
{
    // Make sure we remember to unregister our cuda resource
    cudaGraphicsUnregisterResource(m_resourceVerts);
    cudaGraphicsUnregisterResource(m_resourceNorms);
    // Delete our OpenGL buffers and arrays
    glDeleteBuffers(1,&m_VBOidc);
    glDeleteBuffers(1,&m_VBOverts);
    glDeleteBuffers(1,&m_VBOnorms);
    glDeleteVertexArrays(1,&m_VAO);
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::draw()
{

}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::createPlane(int _width, int _height)
{
    std::vector<glm::vec3> vertices, normals;
    std::vector<glm::vec2> texCoords;

    m_width = _width;
    m_height = _height;

    // calculate the deltas for the x,z values of our point
    float wStep=1.f/(float)m_width;
    float hStep=1.f/(float)m_height;
    // now we assume that the grid is centered at 0,0,0 so we make
    // it flow from -w/2 -d/2
    float xPos=-((float)m_width/2.0);
    float yPos=-((float)m_height/2.0);
    //texture coords
    float xTexCoord,yTexCoord;
    xTexCoord=yTexCoord=0;

    // now loop from top left to bottom right and generate points
    // Sourced form Jon Macey's NGL library
    for(int y=0; y<m_height; y++){
       for(int x=0; x<m_width; x++){

           // grab the colour and use for the Y (height) only use the red channel
          vertices.push_back(glm::vec3(xPos,0.0, yPos));
          normals.push_back(glm::vec3(0.0, 0.0, 0.0));
          texCoords.push_back(glm::vec2(xTexCoord,yTexCoord));

          // calculate the new position
          yPos+=hStep;
          //calculate next texture coord
          yTexCoord+=hStep;
       }

       // now increment to next x row
       xPos+=wStep;
       xTexCoord+=wStep;
       // we need to re-set the ypos for new row
       yPos=-((float)m_height/2.f);
       yTexCoord = 0;
    }

    GLuint numTris = (m_height-1)*(m_width-1)*2;
    GLuint *tris = new GLuint[numTris*3];
    int i, j, fidx = 0;
    for (i=0; i < m_height - 1; ++i) {
        for (j=0; j < m_width - 1; ++j) {
            tris[fidx*3+0] = (i+1)*m_height+j;
            tris[fidx*3+1] = i*m_height+j+1;
            tris[fidx*3+2] = i*m_height+j;
            fidx++;
            tris[fidx*3+0] = (i+1)*m_height+j;
            tris[fidx*3+1] = (i+1)*m_height+j+1;
            tris[fidx*3+2] = i*m_height+j+1;
            fidx++;
        }
    }

    m_numVerts = 3*numTris;

    // Create our VAO and vertex buffers
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    // Put our vertices into an OpenGL buffer
    glGenBuffers(1, &m_VBOverts);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOverts);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // create our cuda graphics resource for our vertexs used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceVerts, m_VBOverts, cudaGraphicsRegisterFlagsNone));

    // create some pretty standard normals
    std::vector<float3> norms;
    for (int i=0; i<m_height*m_width; i++){
        norms.push_back(make_float3(0.0, 0.0, 1.0));
    }

    // Put our normals into an OpenGL buffer
    glGenBuffers(1, &m_VBOnorms);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOnorms);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*norms.size(), &norms[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // create our cuda graphics resource for our normals used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceNorms, m_VBOnorms, cudaGraphicsRegisterFlagsWriteDiscard));

    // Set our indecies for our plane
    glGenBuffers(1, &m_VBOidc);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_VBOidc);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3*numTris*sizeof(GLuint),tris, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
//----------------------------------------------------------------------------------------------------------------------

#include "ClothSim.h"

#include <vector>
#define GLM_FORCE_RADIANS
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <QImage>
#include <QGLWidget>
#include "GLTextureLib.h"
#include <helper_cuda.h>


//----------------------------------------------------------------------------------------------------------------------
ClothSim::ClothSim(int _width, int _height) : m_restLength(1), m_mass(1)
{
    //Lets test some cuda stuff
    int count;
    if (cudaGetDeviceCount(&count))
        return;
    std::cout << "Found" << count << "CUDA device(s)" << std::endl;
    if(count == 0){
        std::cerr<<"Install an Nvidia chip!"<<std::endl;
        return;
    }
    for (int i=0; i < count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        QString deviceString = QString("* %1, Compute capability: %2.%3").arg(prop.name).arg(prop.major).arg(prop.minor);
        QString propString1 = QString("  Global mem: %1M, Shared mem per block: %2k, Registers per block: %3").arg(prop.totalGlobalMem / 1024 / 1024).arg(prop.sharedMemPerBlock / 1024).arg(prop.regsPerBlock);
        QString propString2 = QString("  Warp size: %1 threads, Max threads per block: %2, Multiprocessor count: %3 MaxBlocks: %4").arg(prop.warpSize).arg(prop.maxThreadsPerBlock).arg(prop.multiProcessorCount).arg(prop.maxGridSize[0]);
        std::cout << deviceString.toStdString() << std::endl;
        std::cout << propString1.toStdString() << std::endl;
        std::cout << propString2.toStdString() << std::endl;
        m_threadsPerBlock = prop.maxThreadsPerBlock;
    }
    //create our plane
    createPlane(_width,_height);
    createPhongShader();
    createClothShader();
    usePhongShader();
    //useClothShader();
}
//----------------------------------------------------------------------------------------------------------------------
ClothSim::~ClothSim()
{
    // Make sure we remember to unregister our cuda resource
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourceVerts));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourceNorms));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourceFixedParts));
    // Delete our OpenGL buffers and arrays
    glDeleteBuffers(1,&m_VBOidc);
    glDeleteBuffers(1,&m_VBOverts);
    glDeleteBuffers(1,&m_VBOnorms);
    glDeleteBuffers(1,&m_VBOtexCoords);
    glDeleteBuffers(1,&m_VBOFixedVerts);
    glDeleteVertexArrays(1,&m_VAO);
    // Delete our device pointers
    for(unsigned int i=0; i<d_constraintBuffers.size(); i++)
        checkCudaErrors(cudaFree(d_constraintBuffers[i].ptr));
    checkCudaErrors(cudaFree(d_oldParticlePos));
    // Delete our CUDA streams as well
    checkCudaErrors(cudaStreamDestroy(m_cudaStream));
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::draw(glm::mat4 _MV, glm::mat4 _MVP, glm::mat3 _normalMat)
{
    //use or phong shader program
    m_activeShaderProgram->use();
    //load in our matricies
    glUniformMatrix4fv(m_MVLoc, 1, GL_FALSE, glm::value_ptr(_MV));
    glUniformMatrix4fv(m_MVPLoc, 1, GL_FALSE, glm::value_ptr(_MVP));
    glUniformMatrix3fv(m_normMatLoc, 1, GL_FALSE, glm::value_ptr(_normalMat));

    //std::cout<<m_numVerts<<std::endl;
    //draw our VAO
    glBindVertexArray(m_VAO);
    (*GLTextureLib::getInstance())["clothTexture"]->bind(0);
    glDrawElements(GL_TRIANGLES, m_numVerts,GL_UNSIGNED_INT, (void*)0);
    glBindVertexArray(0);
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::update(float _timeStep)
{
    //map our buffer pointers
    float3* d_posPtr;
    size_t d_posSize;
    cudaGraphicsMapResources(1,&m_resourceVerts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_resourceVerts);

    // Verlet integration
    clothVerletIntegration(m_cudaStream,d_posPtr,d_oldParticlePos,m_numParticles,m_mass,_timeStep,m_threadsPerBlock);

    //make sure all our threads are done
    cudaThreadSynchronize();


    for(unsigned int i=0; i<d_constraintBuffers.size(); i++)
    {
        // Satisfy our constraints
        clothConstraintSolver(m_cudaStream,d_posPtr,d_constraintBuffers[i].ptr,d_constraintBuffers[i].numConsts,m_restLength,m_threadsPerBlock);

        //make sure all our threads are done
        cudaThreadSynchronize();
        //std::cout<<std::endl;
    }

    int* d_fixPartPtr;
    size_t d_fixPartSize;
    cudaGraphicsMapResources(1,&m_resourceFixedParts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_fixPartPtr,&d_fixPartSize,m_resourceFixedParts);

    //Reset our constrained particles
    resetFixedParticles(m_cudaStream,d_posPtr,d_oldParticlePos,d_fixPartPtr,m_numParticles,m_threadsPerBlock);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //unmap our buffer pointer and set it free into the wild
    cudaGraphicsUnmapResources(1,&m_resourceVerts);
    cudaGraphicsUnmapResources(1,&m_resourceFixedParts);
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::setTexture(QString _loc)
{
    QImage tex = QImage(_loc);
    //some error checking
    if(tex.isNull()){
        std::cerr<<"Error: Texture could not be loaded."<<std::endl;
        return;
    }
    tex = QGLWidget::convertToGLFormat(tex);
    GLTextureLib *texLib = GLTextureLib::getInstance();
    GLTexture *GLtex = texLib->addTexture("clothTexture",GL_TEXTURE_2D,0,GL_RGBA,tex.width(),tex.height(),0,GL_RGBA,GL_UNSIGNED_BYTE,tex.bits());
    GLtex->setTexParamiteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    GLtex->setTexParamiteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    m_phongShaderProgram->use();
    GLtex->bind(0);
    glUniform1i(m_phongShaderProgram->getUniformLoc("tex"),0);
    m_clothShaderProgram->use();
    GLtex->bind(0);
    glUniform1i(m_clothShaderProgram->getUniformLoc("tex"),0);

}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::reset()
{
    // now loop from bottom left to top right and generate points
    // Sourced form Jon Macey's NGL library
    std::vector<float3> vertices;
    // calculate the deltas for the x,z values of our point
    float wStep=1.f/(float)m_width;
    float hStep=1.f/(float)m_height;
    // now we assume that the grid is centered at 0,0,0 so we make
    // it flow from -w/2 -d/2
    float xPos=-0.5;
    float yPos=0.5;
    for(int y=0; y<m_height; y++)
    {
       for(int x=0; x<m_width; x++)
       {
          // grab the colour and use for the Y (height) only use the red channel
          vertices.push_back(make_float3(xPos*10,yPos*10,0));
          // calculate the new position
          yPos-=hStep;
       }

       // now increment to next x row
       xPos+=wStep;
       // we need to re-set the ypos for new row
       yPos=0.5;
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_VBOverts);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaMemcpy(d_oldParticlePos,&vertices[0],vertices.size()*sizeof(float3),cudaMemcpyHostToDevice);

}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::usePhongShader()
{
    m_phongShaderProgram->use();
    //get the locations of our uniforms that we frequently use in our shader
    m_MVLoc = m_phongShaderProgram->getUniformLoc("modelViewMatrix");
    m_MVPLoc = m_phongShaderProgram->getUniformLoc("modelViewProjectionMatrix");
    m_normMatLoc = m_phongShaderProgram->getUniformLoc("normalMatrix");
    m_activeShaderProgram = m_phongShaderProgram;
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::useClothShader()
{
    m_clothShaderProgram->use();
    //get the locations of our uniforms that we frequently use in our shader
    m_MVLoc = m_clothShaderProgram->getUniformLoc("MV");
    m_MVPLoc = m_clothShaderProgram->getUniformLoc("MVP");
    m_normMatLoc = m_clothShaderProgram->getUniformLoc("normalMatrix");
    m_activeShaderProgram = m_clothShaderProgram;
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::fixNewPoints(glm::vec3 _from, glm::vec3 _ray, glm::mat4 _viewMatrix)
{

    Eigen::Vector3f from(_from.x,_from.y,_from.z);
    Eigen::Vector3f ray(_ray.x,_ray.y,_ray.z);
    Eigen::Matrix4f egnM;
    egnM << _viewMatrix[0][0],_viewMatrix[1][0],_viewMatrix[2][0],_viewMatrix[3][0],
            _viewMatrix[0][1],_viewMatrix[1][1],_viewMatrix[2][1],_viewMatrix[3][1],
            _viewMatrix[0][2],_viewMatrix[1][2],_viewMatrix[2][2],_viewMatrix[3][2],
            _viewMatrix[0][3],_viewMatrix[1][3],_viewMatrix[2][3],_viewMatrix[3][3];

    //map our buffer pointer
    float3* d_posPtr;
    int* d_fixPartPtr;
    size_t d_posSize,d_fixPartSize;
    cudaGraphicsMapResources(1,&m_resourceVerts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_resourceVerts);
    cudaGraphicsMapResources(1,&m_resourceFixedParts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_fixPartPtr,&d_fixPartSize,m_resourceFixedParts);

    //test our intersection
    fixParticles(m_cudaStream,d_posPtr,d_fixPartPtr,from,ray,m_vertRadius,egnM,m_numParticles,m_threadsPerBlock);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //unmap our buffer pointer and set it free into the wild
    cudaGraphicsUnmapResources(1,&m_resourceVerts);
    cudaGraphicsUnmapResources(1,&m_resourceFixedParts);
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::unFixPoints(glm::vec3 _from, glm::vec3 _ray, glm::mat4 _viewMatrix)
{
    Eigen::Vector3f from(_from.x,_from.y,_from.z);
    Eigen::Vector3f ray(_ray.x,_ray.y,_ray.z);
    Eigen::Matrix4f egnM;
    egnM << _viewMatrix[0][0],_viewMatrix[1][0],_viewMatrix[2][0],_viewMatrix[3][0],
            _viewMatrix[0][1],_viewMatrix[1][1],_viewMatrix[2][1],_viewMatrix[3][1],
            _viewMatrix[0][2],_viewMatrix[1][2],_viewMatrix[2][2],_viewMatrix[3][2],
            _viewMatrix[0][3],_viewMatrix[1][3],_viewMatrix[2][3],_viewMatrix[3][3];

    //map our buffer pointer
    float3* d_posPtr;
    int* d_fixPartPtr;
    size_t d_posSize,d_fixPartSize;
    cudaGraphicsMapResources(1,&m_resourceVerts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_resourceVerts);
    cudaGraphicsMapResources(1,&m_resourceFixedParts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_fixPartPtr,&d_fixPartSize,m_resourceFixedParts);

    //test our intersection
    unFixParticles(m_cudaStream,d_posPtr,d_fixPartPtr,from,ray,m_vertRadius,egnM,m_numParticles,m_threadsPerBlock);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //unmap our buffer pointer and set it free into the wild
    cudaGraphicsUnmapResources(1,&m_resourceVerts);
    cudaGraphicsUnmapResources(1,&m_resourceFixedParts);
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::setMoveVertex(glm::vec3 _from, glm::vec3 _ray, glm::mat4 _viewMatrix){
    Eigen::Vector3f from(_from.x,_from.y,_from.z);
    Eigen::Vector3f ray(_ray.x,_ray.y,_ray.z);
    Eigen::Matrix4f egnM;
    egnM << _viewMatrix[0][0],_viewMatrix[1][0],_viewMatrix[2][0],_viewMatrix[3][0],
            _viewMatrix[0][1],_viewMatrix[1][1],_viewMatrix[2][1],_viewMatrix[3][1],
            _viewMatrix[0][2],_viewMatrix[1][2],_viewMatrix[2][2],_viewMatrix[3][2],
            _viewMatrix[0][3],_viewMatrix[1][3],_viewMatrix[2][3],_viewMatrix[3][3];

    //map our buffer pointer
    float3* d_posPtr;
    int* d_fixPartPtr;
    size_t d_posSize,d_fixPartSize;
    cudaGraphicsMapResources(1,&m_resourceVerts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_resourceVerts);
    cudaGraphicsMapResources(1,&m_resourceFixedParts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_fixPartPtr,&d_fixPartSize,m_resourceFixedParts);

    //set the vertex we wish to move
    setMoveParticles(m_cudaStream,d_posPtr,d_fixPartPtr,d_moveVertex,from,ray,m_vertRadius,egnM,m_numParticles,m_threadsPerBlock);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //unmap our buffer pointer and set it free into the wild
    cudaGraphicsUnmapResources(1,&m_resourceVerts);
    cudaGraphicsUnmapResources(1,&m_resourceFixedParts);
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::removeMoveVertex()
{
    int* d_fixPartPtr;
    size_t d_fixPartSize;
    cudaGraphicsMapResources(1,&m_resourceFixedParts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_fixPartPtr,&d_fixPartSize,m_resourceFixedParts);

    //remove the vertex we wish to move
    unSelectMoveParticle(m_cudaStream,d_fixPartPtr,d_moveVertex);

    //make sure all our threads are done
    cudaThreadSynchronize();

    cudaGraphicsUnmapResources(1,&m_resourceFixedParts);
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::moveSelectedVertex(glm::vec3 _dir)
{
    //map our buffer pointer
    float3* d_posPtr;
    size_t d_posSize;
    cudaGraphicsMapResources(1,&m_resourceVerts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_resourceVerts);

    moveSelectedParticle(m_cudaStream,d_posPtr,d_moveVertex,make_float3(_dir.x,_dir.y,_dir.z));

    //make sure all our threads are done
    cudaThreadSynchronize();

    //unmap our buffer pointer and set it free into the wild
    cudaGraphicsUnmapResources(1,&m_resourceVerts);
}

//----------------------------------------------------------------------------------------------------------------------
void ClothSim::createPlane(int _width, int _height)
{
    std::vector<float3> vertices;
    std::vector<glm::vec2> texCoords;

    m_width = _width;
    m_height = _height;


    // calculate the deltas for the x,z values of our point
    float wStep=1.f/(float)m_width;
    float hStep=1.f/(float)m_height;
    m_restLength = 10.f/((m_width+m_height)/2);
    m_vertRadius = 5.f/((m_width+m_height)/2);
    std::cout<<"rest Length "<<m_restLength<<std::endl;
    // now we assume that the grid is centered at 0,0,0 so we make
    // it flow from -w/2 -d/2
    float xPos=-0.5;
    float yPos=0.5;
    //texture coords
    float xTexCoord,yTexCoord;
    xTexCoord=0;
    yTexCoord=1;

    // now loop from bottom left to top right and generate points
    // Sourced form Jon Macey's NGL library
    for(int y=0; y<m_height; y++)
    {
       for(int x=0; x<m_width; x++)
       {

           // grab the colour and use for the Y (height) only use the red channel
          vertices.push_back(make_float3(xPos*10,yPos*10,0));
          texCoords.push_back(glm::vec2(xTexCoord,yTexCoord));

          // calculate the new position
          yPos-=hStep;
          //calculate next texture coord
          yTexCoord-=hStep;
       }

       // now increment to next x row
       xPos+=wStep;
       xTexCoord+=wStep;
       // we need to re-set the ypos for new row
       yPos=0.5;
       yTexCoord = 1;
    }
    m_numParticles = vertices.size();

    GLuint numTris = (m_height-1)*(m_width-1)*2;
    GLuint *tris = new GLuint[numTris*3];
    int i, j, fidx = 0;
    for (i=0; i < m_height - 1; ++i)
    {
        for (j=0; j < m_width - 1; ++j)
        {
            tris[fidx*3+0] = (i+1)*m_width+j;
            tris[fidx*3+1] = i*m_width+j+1;
            tris[fidx*3+2] = i*m_width+j;
            fidx++;
            tris[fidx*3+0] = (i+1)*m_width+j;
            tris[fidx*3+1] = (i+1)*m_width+j+1;
            tris[fidx*3+2] = i*m_width+j+1;
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
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // create our cuda graphics resource for our vertexs used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceVerts, m_VBOverts, cudaGraphicsRegisterFlagsWriteDiscard));

    // create some pretty standard normals
    std::vector<float3> norms;
    for (int i=0; i<m_numVerts; i++)
    {
        norms.push_back(make_float3(0.0, 0.0, -1.0));
    }

    // Put our normals into an OpenGL buffer
    glGenBuffers(1, &m_VBOnorms);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOnorms);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*norms.size(), &norms[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // create our cuda graphics resource for our normals used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceNorms, m_VBOnorms, cudaGraphicsRegisterFlagsWriteDiscard));

    // Put our texture coords into an OpenGL buffer
    glGenBuffers(1, &m_VBOtexCoords);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOtexCoords);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)*texCoords.size(), &texCoords[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);


    // OpenGL buffer to tell us if the points are fixed or not
    std::vector<int> fixedVerts;
    for (unsigned int i=0; i<vertices.size(); i++)
    {
        fixedVerts.push_back(0);
    }
    glGenBuffers(1, &m_VBOFixedVerts);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOFixedVerts);
    glBufferData(GL_ARRAY_BUFFER, sizeof(int)*fixedVerts.size(), &fixedVerts[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_INT, GL_FALSE, 0, 0);

    // create our cuda graphics resource for our fixed verts used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceFixedParts, m_VBOFixedVerts, cudaGraphicsRegisterFlagsWriteDiscard));

    // Set our indecies for our plane
    glGenBuffers(1, &m_VBOidc);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_VBOidc);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3*numTris*sizeof(GLuint),tris, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Now lets initialise our constraints data
    // Now the issue with porting a cloth simulation to the GPU is when you relax the contraints you
    // can get race conditions when writting the new positions of the vertecies. To solve this we
    // organise the constraints into seperate buffers such that no constraint shares vertecies with
    // other constraints then solving one at a time. I do this by having 2 buffers for every other horizontal
    // constraint and 2 buffers for every other horizontal contraint. I had issues with the diagonal constraints
    // but the result still looks ok therefore I have left them out.
    std::vector<constraints> constraintData1,constraintData2,constraintData3,constraintData4,constraintData5,constraintData6;
    bool switch1,switch2;
    switch1 = switch2 = true;
    for(int y=0; y<m_height; y++)
    {
        for(int x=0; x<m_width; x++)
        {
            //create a constraint
            constraints c;
            c.particleA = y*m_width + x;
            //add the horizontal constraint --> =
            if(x!=m_width-1)
            {
                c.particleB = c.particleA+1;
                if (switch1)
                {
                    constraintData1.push_back(c);
                    switch1=!switch1;
                }
                else
                {
                    constraintData2.push_back(c);
                    switch1=!switch1;
                }

                // diagonal up right --> //
                /// @todo diagonal up right springs not working
                if(y!=0)
                {
                    c.particleB = c.particleA-m_width+1;
                    //constraintData5.push_back(c);
                    //std::cout<<"//"<<c.particleB<<std::endl;
                }

            }
            if(y!=m_height-1)
            {
                //add the vertical constraint --> ||
                c.particleB = c.particleA+m_width;
                if(switch2)
                {
                    constraintData3.push_back(c);
                }
                else
                {
                    constraintData4.push_back(c);
                }

                // diagonal down right --> \\
                /// @todo diagonal down right springs not working
                if(x!=m_width-1)
                {
                    c.particleB = c.particleA+m_width+1;
                    //constraintData6.push_back(c);
                    //std::cout<<"\\\\"<<c.particleB<<std::endl;
                }

            }
        }
        switch2=!switch2;
    }

    // the top 2 corners will be addd to our fixed point array so the cloth is attached to something
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOFixedVerts);
    void* data = glMapBuffer(GL_ARRAY_BUFFER,GL_WRITE_ONLY);
    int* ptr = (int*) data;
    ptr[0] = 1;
    ptr[m_width*(m_height-1)] = 1;
    glUnmapBuffer(GL_ARRAY_BUFFER);

    // Our old positions buffer
    cudaMalloc(&d_oldParticlePos,vertices.size()*sizeof(float3));
    cudaMemcpy(d_oldParticlePos,&vertices[0],vertices.size()*sizeof(float3),cudaMemcpyHostToDevice);

    // Our constraint data
    d_constraintBuffers.resize(6);
    cudaMalloc(&d_constraintBuffers[0].ptr,constraintData1.size()*sizeof(constraints));
    cudaMemcpy(d_constraintBuffers[0].ptr,&constraintData1[0],constraintData1.size()*sizeof(constraints),cudaMemcpyHostToDevice);
    d_constraintBuffers[0].numConsts = constraintData1.size();

    cudaMalloc(&d_constraintBuffers[1].ptr,constraintData2.size()*sizeof(constraints));
    cudaMemcpy(d_constraintBuffers[1].ptr,&constraintData2[0],constraintData2.size()*sizeof(constraints),cudaMemcpyHostToDevice);
    d_constraintBuffers[1].numConsts = constraintData2.size();

    cudaMalloc(&d_constraintBuffers[2].ptr,constraintData3.size()*sizeof(constraints));
    cudaMemcpy(d_constraintBuffers[2].ptr,&constraintData3[0],constraintData3.size()*sizeof(constraints),cudaMemcpyHostToDevice);
    d_constraintBuffers[2].numConsts = constraintData3.size();

    cudaMalloc(&d_constraintBuffers[3].ptr,constraintData4.size()*sizeof(constraints));
    cudaMemcpy(d_constraintBuffers[3].ptr,&constraintData4[0],constraintData4.size()*sizeof(constraints),cudaMemcpyHostToDevice);
    d_constraintBuffers[3].numConsts = constraintData4.size();

    cudaMalloc(&d_constraintBuffers[4].ptr,constraintData5.size()*sizeof(constraints));
    cudaMemcpy(d_constraintBuffers[4].ptr,&constraintData5[0],constraintData5.size()*sizeof(constraints),cudaMemcpyHostToDevice);
    d_constraintBuffers[4].numConsts = constraintData5.size();

    cudaMalloc(&d_constraintBuffers[5].ptr,constraintData6.size()*sizeof(constraints));
    cudaMemcpy(d_constraintBuffers[5].ptr,&constraintData6[0],constraintData6.size()*sizeof(constraints),cudaMemcpyHostToDevice);
    d_constraintBuffers[5].numConsts = constraintData6.size();

    for(unsigned int i=0;i<d_constraintBuffers.size();i++){
        std::cout<<d_constraintBuffers[i].numConsts<<std::endl;
    }

    // Create a buffer to store the index of the particle to move with mouse interaction
    cudaMalloc(&d_moveVertex,sizeof(int));
    int temp = -1;
    cudaMemcpy(d_moveVertex,&temp,sizeof(int),cudaMemcpyHostToDevice);

    // Create our CUDA stream to run our kernals on. This helps with running kernals concurrently.
    // This is something you will not get taught by richard! Check them out at http://on-demand.gputechconf.com/gtc-express/2011/presentations/StreamsAndConcurrencyWebinar.pdf
    cudaStreamCreate(&m_cudaStream);

    std::cout<<"numVerts is "<<vertices.size()<<std::endl;

}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::createPhongShader()
{
    m_phongShaderProgram = new ShaderProgram();
    Shader vertShader("shaders/phongVert.glsl", GL_VERTEX_SHADER);
    Shader fragShader("shaders/phongFrag.glsl", GL_FRAGMENT_SHADER);
    m_phongShaderProgram->attachShader(&vertShader);
    m_phongShaderProgram->attachShader(&fragShader);
    m_phongShaderProgram->bindFragDataLocation(0, "fragColour");
    m_phongShaderProgram->link();
    m_phongShaderProgram->use();

    glUniform4f(m_phongShaderProgram->getUniformLoc("light.position"),1.f,1.f,3.f,1.f);
    glUniform3f(m_phongShaderProgram->getUniformLoc("light.intensity"),.2f,.2f,.2f);
    glUniform3f(m_phongShaderProgram->getUniformLoc("Kd"),0.7f,0.7f,0.7f);
    glUniform3f(m_phongShaderProgram->getUniformLoc("Ka"),0.7f,0.7f,0.7f);
    glUniform3f(m_phongShaderProgram->getUniformLoc("Ks"),0.5f,0.5f,0.5f);
    glUniform1f(m_phongShaderProgram->getUniformLoc("shininess"),10.f);

    //get the locations of our uniforms that we frequently use in our shader
    m_MVLoc = m_phongShaderProgram->getUniformLoc("modelViewMatrix");
    m_MVPLoc = m_phongShaderProgram->getUniformLoc("modelViewProjectionMatrix");
    m_normMatLoc = m_phongShaderProgram->getUniformLoc("normalMatrix");

}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::createClothShader()
{
    m_clothShaderProgram = new ShaderProgram();
    Shader vertShader("shaders/clothVert.glsl", GL_VERTEX_SHADER);
    Shader geomShader("shaders/clothGeom.glsl", GL_GEOMETRY_SHADER);
    Shader fragShader("shaders/clothFrag.glsl", GL_FRAGMENT_SHADER);
    m_clothShaderProgram->attachShader(&vertShader);
    m_clothShaderProgram->attachShader(&geomShader);
    m_clothShaderProgram->attachShader(&fragShader);
    m_clothShaderProgram->bindFragDataLocation(0, "fragColour");
    m_clothShaderProgram->link();
    m_clothShaderProgram->use();

    glUniform4f(m_clothShaderProgram->getUniformLoc("light.position"),1.f,1.f,3.f,1.f);
    glUniform3f(m_clothShaderProgram->getUniformLoc("light.intensity"),.2f,.2f,.2f);
    glUniform3f(m_clothShaderProgram->getUniformLoc("Kd"),0.7f,0.7f,0.7f);
    glUniform3f(m_clothShaderProgram->getUniformLoc("Ka"),0.7f,0.7f,0.7f);
    glUniform3f(m_clothShaderProgram->getUniformLoc("Ks"),0.5f,0.5f,0.5f);
    glUniform1f(m_clothShaderProgram->getUniformLoc("shininess"),10.f);

    //get the locations of our uniforms that we frequently use in our shader
    m_MVLoc = m_clothShaderProgram->getUniformLoc("MV");
    m_MVPLoc = m_clothShaderProgram->getUniformLoc("MVP");
    m_normMatLoc = m_clothShaderProgram->getUniformLoc("normalMatrix");
}
//----------------------------------------------------------------------------------------------------------------------

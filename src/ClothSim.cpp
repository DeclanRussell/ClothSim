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
    glDeleteBuffers(1,&m_VBOtexCoords);
    glDeleteVertexArrays(1,&m_VAO);
    // Delete our device pointers
    for(unsigned int i=0; i<d_particlesBuffers.size();i++)
    {
       checkCudaErrors(cudaFree(d_particlesBuffers[i].ptr));
    }
    // Delete our CUDA streams as well
    for(unsigned int i=0; i<m_cudaStreams.size();i++)
    {
        cudaStreamDestroy(m_cudaStreams[i]);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::draw(glm::mat4 _MV, glm::mat4 _MVP, glm::mat3 _normalMat)
{
    //use or phong shader program
    m_shaderProgram->use();
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
    //map our buffer pointer
    float3* d_posPtr;
    size_t d_posSize;
    cudaGraphicsMapResources(1,&m_resourceVerts);
    cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_resourceVerts);

    // Launch our kernals
    for(unsigned int i=0; i<d_particlesBuffers.size();i++)
    {
        clothSolver(m_cudaStreams[i],d_posPtr,d_particlesBuffers[i].ptr,d_particlesBuffers[i].numParticles,d_particlesBuffers[i].numN,m_restLength,m_mass,_timeStep,m_threadsPerBlock);
    }

    //make sure all our threads are done
    cudaThreadSynchronize();

    //Reset our constrained particles
    resetConstParticles(d_posPtr,d_constPartBuffer.ptr,d_constPartBuffer.numParticles,m_threadsPerBlock);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //unmap our buffer pointer and set it free into the wild
    cudaGraphicsUnmapResources(1,&m_resourceVerts);
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
    m_shaderProgram->use();
    GLtex->bind(0);
    glUniform1i(m_shaderProgram->getUniformLoc("tex"),0);

}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::createPlane(int _width, int _height)
{
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> texCoords;

    m_width = _width;
    m_height = _height;

    // calculate the deltas for the x,z values of our point
    float wStep=1.f/(float)m_width;
    float hStep=1.f/(float)m_height;
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
          vertices.push_back(glm::vec3(xPos*m_width,yPos*m_height,0));
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
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // create our cuda graphics resource for our vertexs used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourceVerts, m_VBOverts, cudaGraphicsRegisterFlagsWriteDiscard));

    // create some pretty standard normals
    std::vector<float3> norms;
    for (int i=0; i<m_numVerts; i++){
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

    // Set our indecies for our plane
    glGenBuffers(1, &m_VBOidc);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_VBOidc);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3*numTris*sizeof(GLuint),tris, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Now lets initialise our particle data
    // Im going to seperate out these into 3 different arrays as you may have up to 4 neighbours per particle
    // but with our setup we wont have any with only one neighbour
    // You will see why I do this with how I have done my CUDA kernals
    std::vector<particles> constPartData,partData2,partData3,partData4;
    for(int y=0; y<m_height; y++)
    {
        for(int x=0; x<m_width; x++)
        {
            //create a particle
            particles p;
            p.acc = make_float3(0,0,0);
            p.idx = y*m_width + x;
            p.oldP = make_float3(vertices[p.idx].x,vertices[p.idx].y,vertices[p.idx].z);

            // lets constrain the top 2 corners of particles so there is something for our cloth to hang off
            if((x==0&&y==0)||(x==0&&y==m_height-1))
            {
                p.idx = y*m_width + x;
                p.numN = 0;
                constPartData.push_back(p);
                continue;
            }

            // The corners only have 2 neighbours
            if((x==m_width-1&&y==0)||(x==m_width-1&&y==m_height-1))
            {
                p.numN = 2;
                (x==0)? p.nIdx[0] = p.idx + 1 : p.nIdx[0] = p.idx - 1;
                (y==0)? p.nIdx[1] = p.idx + m_width : p.nIdx[1] = p.idx - m_width;
                //std::cout<<p.nIdx[0]<<","<<p.nIdx[1]<<std::endl;
                partData2.push_back(p);
                continue;
            }

            // if the middle section of our cloth which will all have 4 neighbours
            if((x>0 && x<m_width-1)&&(y>0 && y<m_height-1))
            {
                p.numN = 4;
                p.nIdx[0] = p.idx - 1;
                p.nIdx[1] = p.idx + 1;
                p.nIdx[2] = p.idx - m_width;
                p.nIdx[3] = p.idx + m_width;
                //std::cout<<p.nIdx[0]<<","<<p.nIdx[1]<<","<<p.nIdx[2]<<","<<p.nIdx[3]<<std::endl;
                partData4.push_back(p);
            }
            else
            {
                // not the corners there are 3 neighbours
                p.numN = 3;
                if(y==0||y==m_height-1)
                {
                    p.nIdx[0]=p.idx+1;
                    p.nIdx[1]=p.idx-1;
                    (y==0) ? p.nIdx[2] = p.idx+m_width : p.nIdx[2] = p.idx-m_width;
                }
                else
                {
                    (x==0)? p.nIdx[0] = p.idx + 1 : p.nIdx[0] = p.idx - 1;
                    p.nIdx[1] = p.idx + m_width;
                    p.nIdx[2] = p.idx - m_width;
                    partData3.push_back(p);
                }
                //std::cout<<p.nIdx[0]<<","<<p.nIdx[1]<<","<<p.nIdx[2]<<std::endl;


            }
        }
    }

    std::cout<<"Num contraints: "<<constPartData.size()<<std::endl;
    std::cout<<"Num other part: "<<partData2.size()<<" "<<partData3.size()<<" "<<partData4.size()<<" "<<partData2.size()+partData3.size()+partData4.size()<<std::endl;

    // Now lets load the particle information onto our device
    d_particlesBuffers.resize(3);
    cudaMalloc(&d_particlesBuffers[0].ptr,partData2.size()*sizeof(particles));
    cudaMemcpy(d_particlesBuffers[0].ptr,&partData2[0],partData2.size()*sizeof(particles),cudaMemcpyHostToDevice);
    d_particlesBuffers[0].numParticles = partData2.size();
    d_particlesBuffers[0].numN = 2;
    cudaMalloc(&d_particlesBuffers[1].ptr,partData3.size()*sizeof(particles));
    cudaMemcpy(d_particlesBuffers[1].ptr,&partData3[0],partData3.size()*sizeof(particles),cudaMemcpyHostToDevice);
    d_particlesBuffers[1].numParticles = partData3.size();
    d_particlesBuffers[1].numN = 3;
    cudaMalloc(&d_particlesBuffers[2].ptr,partData4.size()*sizeof(particles));
    cudaMemcpy(d_particlesBuffers[2].ptr,&partData4[0],partData4.size()*sizeof(particles),cudaMemcpyHostToDevice);
    d_particlesBuffers[2].numParticles = partData4.size();
    d_particlesBuffers[2].numN = 4;

    // New buffer for our constrained particles
    cudaMalloc(&d_constPartBuffer.ptr,constPartData.size()*sizeof(particles));
    cudaMemcpy(d_constPartBuffer.ptr,&constPartData[0],constPartData.size()*sizeof(particles),cudaMemcpyHostToDevice);
    d_constPartBuffer.numParticles = constPartData.size();
    d_constPartBuffer.numN = 0;

    // Create our CUDA streams to run our kernals on. This helps with running kernals concurrently.
    // This is something you will not get taught by richard! Check them out at http://on-demand.gputechconf.com/gtc-express/2011/presentations/StreamsAndConcurrencyWebinar.pdf
    m_cudaStreams.resize(3);
    for(int i=0;i<3;i++)
    {
        cudaStreamCreate(&m_cudaStreams[i]);
    }

    std::cout<<"numVerts is "<<vertices.size()<<std::endl;

}
//----------------------------------------------------------------------------------------------------------------------
void ClothSim::createPhongShader()
{
    m_shaderProgram = new ShaderProgram();
    Shader vertShader("shaders/phongVert.glsl", GL_VERTEX_SHADER);
    Shader fragShader("shaders/phongFrag.glsl", GL_FRAGMENT_SHADER);
    m_shaderProgram->attachShader(&vertShader);
    m_shaderProgram->attachShader(&fragShader);
    m_shaderProgram->bindFragDataLocation(0, "fragColour");
    m_shaderProgram->link();
    m_shaderProgram->use();

    glUniform4f(m_shaderProgram->getUniformLoc("light.position"),1.f,1.f,3.f,1.f);
    glUniform3f(m_shaderProgram->getUniformLoc("light.intensity"),.2f,.2f,.2f);
    glUniform3f(m_shaderProgram->getUniformLoc("Kd"),1.f,1.f,1.f);
    glUniform3f(m_shaderProgram->getUniformLoc("Ka"),1.f,1.f,1.f);
    glUniform3f(m_shaderProgram->getUniformLoc("Ks"),0.5f,0.5f,0.5f);
    glUniform1f(m_shaderProgram->getUniformLoc("shininess"),10.f);

    //get the locations of our uniforms that we frequently use in our shader
    m_MVLoc = m_shaderProgram->getUniformLoc("modelViewMatrix");
    m_MVPLoc = m_shaderProgram->getUniformLoc("modelViewProjectionMatrix");
    m_normMatLoc = m_shaderProgram->getUniformLoc("normalMatrix");

}
//----------------------------------------------------------------------------------------------------------------------

#include "OpenGLWidget.h"
#include <QGuiApplication>
#include <iostream>
#include <time.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <GLTextureLib.h>

#define DtoR 0.0174532925f
#define GLM_FORCE_CUDA

//----------------------------------------------------------------------------------------------------------------------
/// @brief the increment for x/y translation with mouse movement
//----------------------------------------------------------------------------------------------------------------------
const static float INCREMENT=0.01;
//----------------------------------------------------------------------------------------------------------------------
/// @brief the increment for the wheel zoom
//----------------------------------------------------------------------------------------------------------------------
const static float ZOOM=0.1;

OpenGLWidget::OpenGLWidget(const QGLFormat _format, QWidget *_parent) : QGLWidget(_format,_parent){
    // set this widget to have the initial keyboard focus
    setFocus();
    setFocusPolicy( Qt::StrongFocus );
    //init some members
    m_rotate=false;
    // mouse rotation values set to 0
    m_spinXFace=0;
    m_spinYFace=0;
    m_pan = false;
    m_update = false;
    m_modelPos=glm::vec3(0.0);
    // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
    this->resize(_parent->size());
}
//----------------------------------------------------------------------------------------------------------------------
OpenGLWidget::~OpenGLWidget(){
    delete m_text;
    delete m_cam;
    //Destroy our singleton classes
    GLTextureLib::getInstance()->destroy();
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::initializeGL(){
#ifdef LINUX
    glewExperimental = GL_TRUE;
    GLenum error = glewInit();
    if(error != GLEW_OK){
        std::cerr<<"GLEW IS NOT OK!!! "<<std::endl;
    }
#endif
    glClearColor(0.4f, 0.4f, 0.4f, 1.0f);
    // enable depth testing for drawing
    glEnable(GL_DEPTH_TEST);
    // enable multisampling for smoother drawing
    glEnable(GL_MULTISAMPLE);
    //enable point sprites
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // as re-size is not explicitly called we need to do this.
    glViewport(0,0,width(),height());

    //used for drawing text later
    m_text = new Text(QFont("calibri",14));
    m_text->setColour(255,0,0);
    m_text->setScreenSize(width(),height());

    // Initialise the model matrix
    m_modelMatrix = glm::mat4(1.0);

    // Initialize the camera
    // Now we will create a basic Camera from the graphics library
    // This is a static camera so it only needs to be set once
    // First create Values for the camera position
    glm::vec3 from(0,0,10);
    glm::vec3 to(0,0,0);
    glm::vec3 up(0,1,0);
    m_cam = new Camera(from,to,up);
    // set the shape using FOV 45 Aspect Ratio based on Width and Height
    // The final two are near and far clipping planes of 0.1 and 100
    m_cam->setShape(45.f,width(),height(),1.f,100.f);

    //Initialize my texture library
    GLTextureLib::getInstance();

    m_clothSim = new ClothSim(10,10);
    m_clothSim->setTexture("textures/Luke.bmp");
    m_clothSim->setRestLength(1);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

    m_currentTime = m_currentTime.currentTime();
    startTimer(0);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::resizeGL(const int _w, const int _h){
    // set the viewport for openGL
    glViewport(0,0,_w,_h);
    m_cam->setShape(45,_w,_h, m_cam->getNear(),m_cam->getFar());
    m_text->setScreenSize(_w,_h);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::timerEvent(QTimerEvent *){
    // calculate the framerate
    QTime newTime = m_currentTime.currentTime();
    int msecsPassed = m_currentTime.msecsTo(newTime);
    m_currentTime = m_currentTime.currentTime();
    if(m_update)
    {
        m_clothSim->update((float)msecsPassed/1000.f);
    }
    //m_clothSim->update(0.0001);
    updateGL();
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::paintGL(){

    // clear the screen and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    // calculate the framerate
    QTime newTime = m_currentTime.currentTime();
    int msecsPassed = m_currentTime.msecsTo(newTime);
    m_currentTime = m_currentTime.currentTime();

    glm::mat4 rotx, roty;
    rotx = glm::rotate(rotx,m_spinXFace*DtoR,glm::vec3(1,0,0));
    roty = glm::rotate(roty,m_spinYFace*DtoR,glm::vec3(0,1,0));

    glm::mat4 M = rotx*roty;
    M[3][0] = m_modelPos.x;
    M[3][1] = m_modelPos.y;
    M[3][2] = m_modelPos.z;
    glm::mat4 V = m_cam->getViewMatrix();
    glm::mat4 P = m_cam->getProjectionMatrix();
    glm::mat3 norm = glm::inverseTranspose(glm::mat3(V*M));
    m_clothSim->draw(P*V*M,P*V*M,norm);


    //write our framerate
    QString text;
    if(msecsPassed==0){
        text.sprintf("framerate is faster than we can calculate lul :')");
    }
    else{
        text.sprintf("framerate is %f",(float)(1000.0/msecsPassed));
    }
    m_text->renderText(10,20,text);

    //if we want our camera to pan the increment our rotation
    if(m_pan){ m_spinYFace-=INCREMENT*5;}

}


//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::keyPressEvent(QKeyEvent *_event){
    if(_event->key()==Qt::Key_Escape){
        QGuiApplication::exit();
    }
    switch(_event->key())
    {
    case(Qt::Key_E) :m_clothSim->update(0.0001); break;
    case(Qt::Key_W) :glPolygonMode(GL_FRONT_AND_BACK,GL_LINE); break;
    case(Qt::Key_S) :glPolygonMode(GL_FRONT_AND_BACK,GL_FILL); break;
    case(Qt::Key_1) :m_clothSim->usePhongShader(); break;
    case(Qt::Key_2) :m_clothSim->useClothShader(); break;
    case(Qt::Key_Control) : m_fixPoints = true; break;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::keyReleaseEvent(QKeyEvent *_event)
{
    switch(_event->key())
    {
        case(Qt::Key_Control) : m_fixPoints = false; break;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::mouseMoveEvent (QMouseEvent * _event)
{

    // note the method buttons() is the button state when event was called
    // this is different from button() which is used to check which button was
    // pressed when the mousePress/Release event is generated
    if(m_rotate && _event->buttons() == Qt::LeftButton){
        int diffx=_event->x()-m_origX;
        int diffy=_event->y()-m_origY;
        m_spinXFace += (float) 0.5f * diffy;
        m_spinYFace += (float) 0.5f * diffx;
        m_origX = _event->x();
        m_origY = _event->y();
        updateGL();
    }
    // right mouse translate code
    else if(m_translate && _event->buttons() == Qt::RightButton)
    {
        int diffX = (int)(_event->x() - m_origXPos);
        int diffY = (int)(_event->y() - m_origYPos);
        m_origXPos=_event->x();
        m_origYPos=_event->y();
        m_modelPos.x += INCREMENT * diffX;
        m_modelPos.y -= INCREMENT * diffY;
        updateGL();
    }
    else if(m_fixPoints && _event->buttons() ==Qt::LeftButton){
        glm::mat4 temp;
        glm::vec3 x = glm::unProject(glm::vec3(_event->x(),_event->y(),m_cam->getFar()),temp,m_cam->getProjectionMatrix(),glm::vec4(0,0,width(),height()));
        glm::vec3 ray = x - m_cam->getPos();
        glm::mat4 rotx, roty;
        rotx = glm::rotate(rotx,m_spinXFace*DtoR,glm::vec3(1,0,0));
        roty = glm::rotate(roty,m_spinYFace*DtoR,glm::vec3(0,1,0));
        glm::mat4 M = rotx*roty;
        M[3][0] = m_modelPos.x;
        M[3][1] = m_modelPos.y;
        M[3][2] = m_modelPos.z;
        m_clothSim->fixNewPoints(m_cam->getPos(),ray,M);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::mousePressEvent ( QMouseEvent * _event)
{
    // this method is called when the mouse button is pressed in this case we
    // store the value where the maouse was clicked (x,y) and set the Rotate flag to true
    if(_event->button() == Qt::LeftButton){
        m_origX = _event->x();
        m_origY = _event->y();
        m_rotate = true;
    }
    // right mouse translate mode
    else if(_event->button() == Qt::RightButton)
    {
        m_origXPos = _event->x();
        m_origYPos = _event->y();
        m_translate=true;
    }
    else if(m_fixPoints && _event->buttons() ==Qt::LeftButton){
        glm::mat4 temp;
        glm::vec3 x = glm::unProject(glm::vec3(_event->x(),_event->y(),m_cam->getFar()),temp,m_cam->getProjectionMatrix(),glm::vec4(0,0,width(),height()));
        glm::vec3 ray = x - m_cam->getPos();
        glm::mat4 rotx, roty;
        rotx = glm::rotate(rotx,m_spinXFace*DtoR,glm::vec3(1,0,0));
        roty = glm::rotate(roty,m_spinYFace*DtoR,glm::vec3(0,1,0));
        glm::mat4 M = rotx*roty;
        M[3][0] = m_modelPos.x;
        M[3][1] = m_modelPos.y;
        M[3][2] = m_modelPos.z;
        m_clothSim->fixNewPoints(m_cam->getPos(),ray,M);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::mouseReleaseEvent ( QMouseEvent * _event )
{
    // this event is called when the mouse button is released
    // we then set Rotate to false
    if(_event->button() == Qt::LeftButton){
        m_rotate = false;
    }
    // right mouse translate mode
    if (_event->button() == Qt::RightButton)
    {
        m_translate=false;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::wheelEvent(QWheelEvent *_event)
{
    // check the diff of the wheel position (0 means no change)
    if(_event->delta() > 0)
    {
        m_modelPos.z+=ZOOM;
    }
    else if(_event->delta() <0 )
    {
        m_modelPos.z-=ZOOM;
    }
    updateGL();
}
//-----------------------------------------------------------------------------------------------------------------------


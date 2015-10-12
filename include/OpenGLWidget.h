#ifndef OPENGLWIDGET_H
#define OPENGLWIDGET_H

//----------------------------------------------------------------------------------------------------------------------
/// @file OpenGLWidget.h
/// @class OpenGLWidget
/// @brief Basic Qt widget that holds a OpenGL context
/// @author Declan Russell
/// @version 1.0
/// @date 2/3/15 Initial version
//----------------------------------------------------------------------------------------------------------------------

#ifdef DARWIN
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #include <GL/gl.h>
#endif
#define GLM_FORCE_RADIANS

#include <QGLWidget>
#include <QEvent>
#include <QResizeEvent>
#include <QMessageBox>
#include <QString>
#include <QTime>
#include <QColor>

//some math operators
#include <glm/matrix.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Text.h" //< OpenGL text drawing class
#include "Camera.h"
#include "ClothSim.h"

class OpenGLWidget : public QGLWidget
{
    Q_OBJECT //must include to gain access to qt stuff

public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief ctor for our NGL drawing class
    /// @param [in] parent the parent window to the class
    //----------------------------------------------------------------------------------------------------------------------
    explicit OpenGLWidget(const QGLFormat _format, QWidget *_parent=0);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief dtor must close down and release OpenGL resources
    //----------------------------------------------------------------------------------------------------------------------
    ~OpenGLWidget();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the virtual initialize class is called once when the window is created and we have a valid GL context
    /// use this to setup any default GL stuff
    //----------------------------------------------------------------------------------------------------------------------
    void initializeGL();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this is called everytime we want to draw the scene
    //----------------------------------------------------------------------------------------------------------------------
    void paintGL();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief called to resize the window
    //----------------------------------------------------------------------------------------------------------------------
    void resizeGL(const int _w, const int _h );
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief keyboard press event
    //----------------------------------------------------------------------------------------------------------------------
    void keyPressEvent(QKeyEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief keyboard release event
    //----------------------------------------------------------------------------------------------------------------------
    void keyReleaseEvent(QKeyEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mouse move
    //----------------------------------------------------------------------------------------------------------------------
    void mouseMoveEvent(QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mouse button release
    //----------------------------------------------------------------------------------------------------------------------
    void mouseReleaseEvent(QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mouse button press
    //----------------------------------------------------------------------------------------------------------------------
    void mousePressEvent(QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse wheel is moved
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void wheelEvent( QWheelEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a timer event function from the Q_object
    //----------------------------------------------------------------------------------------------------------------------
    void timerEvent(QTimerEvent *);
    //----------------------------------------------------------------------------------------------------------------------
public slots:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief changes rest length of simulation
    /// @param _len - desired rest length
    //----------------------------------------------------------------------------------------------------------------------
    inline void setRestLen(double _len){m_clothSim->setRestLength((float)_len);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief toggles update of simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline void toggleUpdate(){m_update = !m_update;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief resets our simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline void resetSim(){m_clothSim->reset();}
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our cloth simulation class
    //----------------------------------------------------------------------------------------------------------------------
    ClothSim *m_clothSim;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief used for calculating framerate
    //----------------------------------------------------------------------------------------------------------------------
    QTime m_currentTime;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief used for drawing text in openGL
    //----------------------------------------------------------------------------------------------------------------------
    Text *m_text;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our Camera
    //----------------------------------------------------------------------------------------------------------------------
    Camera *m_cam;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Model matrix
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_modelMatrix;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Normal Matrix
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_normalMatrix;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Mouse transforms
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_mouseGlobalTX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief model pos
    //----------------------------------------------------------------------------------------------------------------------
    glm::vec3 m_modelPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Spin face x
    //----------------------------------------------------------------------------------------------------------------------
    float m_spinXFace;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sping face y
    //----------------------------------------------------------------------------------------------------------------------
    float m_spinYFace;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief transform scene bool
    //----------------------------------------------------------------------------------------------------------------------
    bool m_transformScene;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief rotate bool
    //----------------------------------------------------------------------------------------------------------------------
    bool m_rotate;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief translate bool
    //----------------------------------------------------------------------------------------------------------------------
    bool m_translate;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief fix points mode bool
    //----------------------------------------------------------------------------------------------------------------------
    bool m_fixPoints;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief unfix points mode bool
    //----------------------------------------------------------------------------------------------------------------------
    bool m_unFixPoints;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origY;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origXPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origYPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief bool to indicate if we want to pan our camera
    //----------------------------------------------------------------------------------------------------------------------
    bool m_pan;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief bool to indicate if we want to update the simulation
    //----------------------------------------------------------------------------------------------------------------------
    bool m_update;
    //----------------------------------------------------------------------------------------------------------------------

};

#endif // OPENGLWIDGET_H

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QPushButton>
#include <QDesktopServices>
#include <QDoubleSpinBox>
#include <QLabel>


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow){
    ui->setupUi(this);

    QGLFormat format;
    format.setVersion(4,1);
    format.setProfile(QGLFormat::CoreProfile);

    //do this so everything isnt so bunched up
    this->setMinimumHeight(600);

    //add our openGL context to our scene
    m_openGLWidget = new OpenGLWidget(format,this);
    ui->gridLayout->addWidget(m_openGLWidget,0,0,7,1);

    //simulation settings
    QGroupBox *setGrb = new QGroupBox("Simualtion Settings",this);
    ui->gridLayout->addWidget(setGrb,8,0,1,1);
    QGridLayout *setLayout = new QGridLayout(setGrb);
    setGrb->setLayout(setLayout);

    //rest length field
    setLayout->addWidget(new QLabel("Rest Length: ",setGrb),0,0,1,1);
    QDoubleSpinBox *restSpn = new QDoubleSpinBox(setGrb);
    setLayout->addWidget(restSpn,0,1,1,1);
    restSpn->setDecimals(5);
    restSpn->setValue(0.1);
    connect(restSpn,SIGNAL(valueChanged(double)),m_openGLWidget,SLOT(setRestLen(double)));

    //Group box for our general UI buttons
    QGroupBox *docGrb = new QGroupBox("General:",this);
    ui->gridLayout->addWidget(docGrb,9,0,1,1);
    QGridLayout *docLayout = new QGridLayout(docGrb);
    docGrb->setLayout(docLayout);

    //Play/Pause button
    QPushButton *playBtn = new QPushButton("Play/Pause",docGrb);
    connect(playBtn,SIGNAL(pressed()),m_openGLWidget,SLOT(toggleUpdate()));
    docLayout->addWidget(playBtn,1,0,1,1);

    //Reset button
    QPushButton *resetBtn = new QPushButton("Reset",docGrb);
    connect(resetBtn,SIGNAL(pressed()),m_openGLWidget,SLOT(resetSim()));
    docLayout->addWidget(resetBtn,2,0,1,1);

    //open Documation button
    QPushButton *openDocBtn = new QPushButton("Open Documentation",docGrb);
    connect(openDocBtn,SIGNAL(pressed()),this,SLOT(openDoc()));
    docLayout->addWidget(openDocBtn,3,0,1,1);

}



void MainWindow::openDoc(){
    QDesktopServices::openUrl(QUrl(QDir::currentPath() + "/doc/html/index.html"));
}


MainWindow::~MainWindow(){
    delete ui;
    delete m_openGLWidget;

}

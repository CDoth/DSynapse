CONFIG -= qt

TEMPLATE = lib
CONFIG += staticlib

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    DSynapse.cpp \
    layer.cpp \
    net.cpp

HEADERS += \
    DSynapse.h \
    layer.h \
    net.h

# Default rules for deployment.
unix {
    target.path = $$[QT_INSTALL_PLUGINS]/generic
}
!isEmpty(target.path): INSTALLS += target


#======================================================================

#=================================================================== DTL
win32: LIBS += -L$$PWD/../DTL/release/ -lDothTemplateLibrary

INCLUDEPATH += $$PWD/../DTL/
DEPENDPATH += $$PWD/../DTL/

win32:!win32-g++: PRE_TARGETDEPS += $$PWD/../DTL/release/DothTemplateLibrary.lib
else:win32-g++: PRE_TARGETDEPS += $$PWD/../DTL/release/libDothTemplateLibrary.a
#=================================================================== DLogs
win32: LIBS += -L$$PWD/../DLogs/release/ -lDLogs

INCLUDEPATH += $$PWD/../DLogs/
DEPENDPATH += $$PWD/../DLogs/

win32:!win32-g++: PRE_TARGETDEPS += $$PWD/../DLogs/release/DLogs.lib
else:win32-g++: PRE_TARGETDEPS += $$PWD/../DLogs/release/libDLogs.a

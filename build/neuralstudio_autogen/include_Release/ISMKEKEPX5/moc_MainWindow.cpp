/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/app/MainWindow.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>
#include <QtCore/QList>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 69
#error "This file was generated using the moc from 6.10.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {
struct qt_meta_tag_ZN12NeuralStudio10MainWindowE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::MainWindow::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio10MainWindowE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::MainWindow",
        "onDatasetLoaded",
        "",
        "const Dataset*",
        "ds",
        "onNetworkBuilt",
        "std::shared_ptr<NeuralNetwork>",
        "net",
        "onApplyArchToMode1",
        "datasetPath",
        "QList<int>",
        "hiddenSizes",
        "actNames",
        "onRBFNetworkBuilt",
        "std::shared_ptr<RBFNetwork>",
        "onRBFTrainingCompleted",
        "trainErr",
        "valErr",
        "onTrainingCompleted",
        "onStepChanged",
        "index",
        "onOpenFile",
        "onSaveModel",
        "onLoadModel",
        "onSaveProject",
        "onOpenProject",
        "onExportCpp",
        "onExportOnnx",
        "onAbout",
        "onClearRecent",
        "onRecentFileTriggered"
    };

    QtMocHelpers::UintData qt_methods {
        // Slot 'onDatasetLoaded'
        QtMocHelpers::SlotData<void(const Dataset *)>(1, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 3, 4 },
        }}),
        // Slot 'onNetworkBuilt'
        QtMocHelpers::SlotData<void(std::shared_ptr<NeuralNetwork>)>(5, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 6, 7 },
        }}),
        // Slot 'onApplyArchToMode1'
        QtMocHelpers::SlotData<void(QString, QVector<int>, QVector<QString>)>(8, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 9 }, { 0x80000000 | 10, 11 }, { QMetaType::QStringList, 12 },
        }}),
        // Slot 'onRBFNetworkBuilt'
        QtMocHelpers::SlotData<void(std::shared_ptr<RBFNetwork>)>(13, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 14, 7 },
        }}),
        // Slot 'onRBFTrainingCompleted'
        QtMocHelpers::SlotData<void(std::shared_ptr<RBFNetwork>, double, double)>(15, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 14, 7 }, { QMetaType::Double, 16 }, { QMetaType::Double, 17 },
        }}),
        // Slot 'onTrainingCompleted'
        QtMocHelpers::SlotData<void()>(18, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onStepChanged'
        QtMocHelpers::SlotData<void(int)>(19, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 20 },
        }}),
        // Slot 'onOpenFile'
        QtMocHelpers::SlotData<void()>(21, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onSaveModel'
        QtMocHelpers::SlotData<void()>(22, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onLoadModel'
        QtMocHelpers::SlotData<void()>(23, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onSaveProject'
        QtMocHelpers::SlotData<void()>(24, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onOpenProject'
        QtMocHelpers::SlotData<void()>(25, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportCpp'
        QtMocHelpers::SlotData<void()>(26, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportOnnx'
        QtMocHelpers::SlotData<void()>(27, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onAbout'
        QtMocHelpers::SlotData<void()>(28, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onClearRecent'
        QtMocHelpers::SlotData<void()>(29, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onRecentFileTriggered'
        QtMocHelpers::SlotData<void()>(30, 2, QMC::AccessPrivate, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<MainWindow, qt_meta_tag_ZN12NeuralStudio10MainWindowE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio10MainWindowE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio10MainWindowE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio10MainWindowE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<MainWindow *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->onDatasetLoaded((*reinterpret_cast<std::add_pointer_t<const Dataset*>>(_a[1]))); break;
        case 1: _t->onNetworkBuilt((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1]))); break;
        case 2: _t->onApplyArchToMode1((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QList<int>>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QList<QString>>>(_a[3]))); break;
        case 3: _t->onRBFNetworkBuilt((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<RBFNetwork>>>(_a[1]))); break;
        case 4: _t->onRBFTrainingCompleted((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<RBFNetwork>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3]))); break;
        case 5: _t->onTrainingCompleted(); break;
        case 6: _t->onStepChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 7: _t->onOpenFile(); break;
        case 8: _t->onSaveModel(); break;
        case 9: _t->onLoadModel(); break;
        case 10: _t->onSaveProject(); break;
        case 11: _t->onOpenProject(); break;
        case 12: _t->onExportCpp(); break;
        case 13: _t->onExportOnnx(); break;
        case 14: _t->onAbout(); break;
        case 15: _t->onClearRecent(); break;
        case 16: _t->onRecentFileTriggered(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 1:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QList<int> >(); break;
            }
            break;
        }
    }
}

const QMetaObject *NeuralStudio::MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio10MainWindowE_t>.strings))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int NeuralStudio::MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 17)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 17;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 17)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 17;
    }
    return _id;
}
QT_WARNING_POP

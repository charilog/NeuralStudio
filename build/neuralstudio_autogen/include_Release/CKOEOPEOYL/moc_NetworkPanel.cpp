/****************************************************************************
** Meta object code from reading C++ file 'NetworkPanel.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/ui/panels/NetworkPanel.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'NetworkPanel.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN12NeuralStudio12NetworkPanelE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::NetworkPanel::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio12NetworkPanelE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::NetworkPanel",
        "networkBuilt",
        "",
        "std::shared_ptr<NeuralNetwork>",
        "net",
        "rbfBuilt",
        "std::shared_ptr<RBFNetwork>",
        "onDatasetLoaded",
        "const Dataset*",
        "ds",
        "onNumLayersChanged",
        "n",
        "onBuildClicked",
        "onBuildRBFClicked",
        "onTypeToggled"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'networkBuilt'
        QtMocHelpers::SignalData<void(std::shared_ptr<NeuralNetwork>)>(1, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 3, 4 },
        }}),
        // Signal 'rbfBuilt'
        QtMocHelpers::SignalData<void(std::shared_ptr<RBFNetwork>)>(5, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 6, 4 },
        }}),
        // Slot 'onDatasetLoaded'
        QtMocHelpers::SlotData<void(const Dataset *)>(7, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 8, 9 },
        }}),
        // Slot 'onNumLayersChanged'
        QtMocHelpers::SlotData<void(int)>(10, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 11 },
        }}),
        // Slot 'onBuildClicked'
        QtMocHelpers::SlotData<void()>(12, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onBuildRBFClicked'
        QtMocHelpers::SlotData<void()>(13, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onTypeToggled'
        QtMocHelpers::SlotData<void()>(14, 2, QMC::AccessPrivate, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<NetworkPanel, qt_meta_tag_ZN12NeuralStudio12NetworkPanelE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::NetworkPanel::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio12NetworkPanelE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio12NetworkPanelE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio12NetworkPanelE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::NetworkPanel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<NetworkPanel *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->networkBuilt((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1]))); break;
        case 1: _t->rbfBuilt((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<RBFNetwork>>>(_a[1]))); break;
        case 2: _t->onDatasetLoaded((*reinterpret_cast<std::add_pointer_t<const Dataset*>>(_a[1]))); break;
        case 3: _t->onNumLayersChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 4: _t->onBuildClicked(); break;
        case 5: _t->onBuildRBFClicked(); break;
        case 6: _t->onTypeToggled(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (NetworkPanel::*)(std::shared_ptr<NeuralNetwork> )>(_a, &NetworkPanel::networkBuilt, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (NetworkPanel::*)(std::shared_ptr<RBFNetwork> )>(_a, &NetworkPanel::rbfBuilt, 1))
            return;
    }
}

const QMetaObject *NeuralStudio::NetworkPanel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::NetworkPanel::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio12NetworkPanelE_t>.strings))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int NeuralStudio::NetworkPanel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void NeuralStudio::NetworkPanel::networkBuilt(std::shared_ptr<NeuralNetwork> _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1);
}

// SIGNAL 1
void NeuralStudio::NetworkPanel::rbfBuilt(std::shared_ptr<RBFNetwork> _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1);
}
QT_WARNING_POP

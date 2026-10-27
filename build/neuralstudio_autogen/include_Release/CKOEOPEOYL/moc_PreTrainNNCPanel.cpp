/****************************************************************************
** Meta object code from reading C++ file 'PreTrainNNCPanel.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/ui/panels/PreTrainNNCPanel.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>
#include <QtCore/QList>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'PreTrainNNCPanel.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN12NeuralStudio8PhaseBoxE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::PhaseBox::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio8PhaseBoxE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::PhaseBox"
    };

    QtMocHelpers::UintData qt_methods {
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<PhaseBox, qt_meta_tag_ZN12NeuralStudio8PhaseBoxE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::PhaseBox::staticMetaObject = { {
    QMetaObject::SuperData::link<QGroupBox::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio8PhaseBoxE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio8PhaseBoxE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio8PhaseBoxE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::PhaseBox::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<PhaseBox *>(_o);
    (void)_t;
    (void)_c;
    (void)_id;
    (void)_a;
}

const QMetaObject *NeuralStudio::PhaseBox::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::PhaseBox::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio8PhaseBoxE_t>.strings))
        return static_cast<void*>(this);
    return QGroupBox::qt_metacast(_clname);
}

int NeuralStudio::PhaseBox::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGroupBox::qt_metacall(_c, _id, _a);
    return _id;
}
namespace {
struct qt_meta_tag_ZN12NeuralStudio16PreTrainNNCPanelE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::PreTrainNNCPanel::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio16PreTrainNNCPanelE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::PreTrainNNCPanel",
        "runningStateChanged",
        "",
        "running",
        "trainingCompleted",
        "std::shared_ptr<NeuralNetwork>",
        "net",
        "std::shared_ptr<Dataset>",
        "ds",
        "onAddFiles",
        "onClearQueue",
        "onRemoveSelected",
        "onRunClicked",
        "onStopClicked",
        "onDatasetStarted",
        "dsIdx",
        "totalDs",
        "dsName",
        "onPhaseStarted",
        "phase",
        "total",
        "name",
        "onIterationDone",
        "iter",
        "loss",
        "vl",
        "onPhaseFinished",
        "finalLoss",
        "onDatasetFinished",
        "configs",
        "QList<double>",
        "trainLoss",
        "trainErr",
        "valErr",
        "testErr",
        "isCls",
        "onAllDone",
        "onError",
        "msg",
        "onExportLossPng",
        "onExportEvalXlsx",
        "onExportCompXlsx",
        "onExportBarPng"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'runningStateChanged'
        QtMocHelpers::SignalData<void(bool)>(1, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Bool, 3 },
        }}),
        // Signal 'trainingCompleted'
        QtMocHelpers::SignalData<void(std::shared_ptr<NeuralNetwork>, std::shared_ptr<Dataset>)>(4, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 5, 6 }, { 0x80000000 | 7, 8 },
        }}),
        // Slot 'onAddFiles'
        QtMocHelpers::SlotData<void()>(9, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onClearQueue'
        QtMocHelpers::SlotData<void()>(10, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onRemoveSelected'
        QtMocHelpers::SlotData<void()>(11, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onRunClicked'
        QtMocHelpers::SlotData<void()>(12, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onStopClicked'
        QtMocHelpers::SlotData<void()>(13, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onDatasetStarted'
        QtMocHelpers::SlotData<void(int, int, QString)>(14, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 15 }, { QMetaType::Int, 16 }, { QMetaType::QString, 17 },
        }}),
        // Slot 'onPhaseStarted'
        QtMocHelpers::SlotData<void(int, int, QString)>(18, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 19 }, { QMetaType::Int, 20 }, { QMetaType::QString, 21 },
        }}),
        // Slot 'onIterationDone'
        QtMocHelpers::SlotData<void(int, int, int, double, double)>(22, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 19 }, { QMetaType::Int, 23 }, { QMetaType::Int, 20 }, { QMetaType::Double, 24 },
            { QMetaType::Double, 25 },
        }}),
        // Slot 'onPhaseFinished'
        QtMocHelpers::SlotData<void(int, double)>(26, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 19 }, { QMetaType::Double, 27 },
        }}),
        // Slot 'onDatasetFinished'
        QtMocHelpers::SlotData<void(QString, QStringList, QVector<double>, QVector<double>, QVector<double>, QVector<double>, bool)>(28, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 17 }, { QMetaType::QStringList, 29 }, { 0x80000000 | 30, 31 }, { 0x80000000 | 30, 32 },
            { 0x80000000 | 30, 33 }, { 0x80000000 | 30, 34 }, { QMetaType::Bool, 35 },
        }}),
        // Slot 'onAllDone'
        QtMocHelpers::SlotData<void(std::shared_ptr<NeuralNetwork>, std::shared_ptr<Dataset>)>(36, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 5, 6 }, { 0x80000000 | 7, 8 },
        }}),
        // Slot 'onError'
        QtMocHelpers::SlotData<void(QString)>(37, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 38 },
        }}),
        // Slot 'onExportLossPng'
        QtMocHelpers::SlotData<void()>(39, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportEvalXlsx'
        QtMocHelpers::SlotData<void()>(40, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportCompXlsx'
        QtMocHelpers::SlotData<void()>(41, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportBarPng'
        QtMocHelpers::SlotData<void()>(42, 2, QMC::AccessPrivate, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<PreTrainNNCPanel, qt_meta_tag_ZN12NeuralStudio16PreTrainNNCPanelE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::PreTrainNNCPanel::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio16PreTrainNNCPanelE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio16PreTrainNNCPanelE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio16PreTrainNNCPanelE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::PreTrainNNCPanel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<PreTrainNNCPanel *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->runningStateChanged((*reinterpret_cast<std::add_pointer_t<bool>>(_a[1]))); break;
        case 1: _t->trainingCompleted((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<std::shared_ptr<Dataset>>>(_a[2]))); break;
        case 2: _t->onAddFiles(); break;
        case 3: _t->onClearQueue(); break;
        case 4: _t->onRemoveSelected(); break;
        case 5: _t->onRunClicked(); break;
        case 6: _t->onStopClicked(); break;
        case 7: _t->onDatasetStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 8: _t->onPhaseStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 9: _t->onIterationDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[5]))); break;
        case 10: _t->onPhaseFinished((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[2]))); break;
        case 11: _t->onDatasetFinished((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QStringList>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[5])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[6])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[7]))); break;
        case 12: _t->onAllDone((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<std::shared_ptr<Dataset>>>(_a[2]))); break;
        case 13: _t->onError((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 14: _t->onExportLossPng(); break;
        case 15: _t->onExportEvalXlsx(); break;
        case 16: _t->onExportCompXlsx(); break;
        case 17: _t->onExportBarPng(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 11:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 5:
            case 4:
            case 3:
            case 2:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QList<double> >(); break;
            }
            break;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (PreTrainNNCPanel::*)(bool )>(_a, &PreTrainNNCPanel::runningStateChanged, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (PreTrainNNCPanel::*)(std::shared_ptr<NeuralNetwork> , std::shared_ptr<Dataset> )>(_a, &PreTrainNNCPanel::trainingCompleted, 1))
            return;
    }
}

const QMetaObject *NeuralStudio::PreTrainNNCPanel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::PreTrainNNCPanel::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio16PreTrainNNCPanelE_t>.strings))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int NeuralStudio::PreTrainNNCPanel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 18)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 18;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 18)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 18;
    }
    return _id;
}

// SIGNAL 0
void NeuralStudio::PreTrainNNCPanel::runningStateChanged(bool _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1);
}

// SIGNAL 1
void NeuralStudio::PreTrainNNCPanel::trainingCompleted(std::shared_ptr<NeuralNetwork> _t1, std::shared_ptr<Dataset> _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1, _t2);
}
QT_WARNING_POP

/****************************************************************************
** Meta object code from reading C++ file 'EvolBoundsPanel.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/ui/panels/EvolBoundsPanel.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>
#include <QtCore/QList>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'EvolBoundsPanel.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN12NeuralStudio15EvolBoundsPanelE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::EvolBoundsPanel::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio15EvolBoundsPanelE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::EvolBoundsPanel",
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
        "onOpt2Changed",
        "idx",
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
        "onDatasetBoundsReady",
        "QList<double>",
        "lb",
        "rb",
        "initBound",
        "onDatasetFinished",
        "configs",
        "trainLoss",
        "trainErr",
        "valErr",
        "testErr",
        "isCls",
        "onAllDone",
        "onError",
        "msg",
        "onExportLossPng",
        "onExportBoundsPng",
        "onExportEvalXlsx",
        "onExportCompXlsx"
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
        // Slot 'onOpt2Changed'
        QtMocHelpers::SlotData<void(int)>(12, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 13 },
        }}),
        // Slot 'onRunClicked'
        QtMocHelpers::SlotData<void()>(14, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onStopClicked'
        QtMocHelpers::SlotData<void()>(15, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onDatasetStarted'
        QtMocHelpers::SlotData<void(int, int, QString)>(16, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 17 }, { QMetaType::Int, 18 }, { QMetaType::QString, 19 },
        }}),
        // Slot 'onPhaseStarted'
        QtMocHelpers::SlotData<void(int, int, QString)>(20, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 21 }, { QMetaType::Int, 22 }, { QMetaType::QString, 23 },
        }}),
        // Slot 'onIterationDone'
        QtMocHelpers::SlotData<void(int, int, int, double, double)>(24, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 21 }, { QMetaType::Int, 25 }, { QMetaType::Int, 22 }, { QMetaType::Double, 26 },
            { QMetaType::Double, 27 },
        }}),
        // Slot 'onPhaseFinished'
        QtMocHelpers::SlotData<void(int, double)>(28, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 21 }, { QMetaType::Double, 29 },
        }}),
        // Slot 'onDatasetBoundsReady'
        QtMocHelpers::SlotData<void(QString, QVector<double>, QVector<double>, double)>(30, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 19 }, { 0x80000000 | 31, 32 }, { 0x80000000 | 31, 33 }, { QMetaType::Double, 34 },
        }}),
        // Slot 'onDatasetFinished'
        QtMocHelpers::SlotData<void(QString, QStringList, QVector<double>, QVector<double>, QVector<double>, QVector<double>, bool)>(35, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 19 }, { QMetaType::QStringList, 36 }, { 0x80000000 | 31, 37 }, { 0x80000000 | 31, 38 },
            { 0x80000000 | 31, 39 }, { 0x80000000 | 31, 40 }, { QMetaType::Bool, 41 },
        }}),
        // Slot 'onAllDone'
        QtMocHelpers::SlotData<void(std::shared_ptr<NeuralNetwork>, std::shared_ptr<Dataset>)>(42, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 5, 6 }, { 0x80000000 | 7, 8 },
        }}),
        // Slot 'onError'
        QtMocHelpers::SlotData<void(QString)>(43, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 44 },
        }}),
        // Slot 'onExportLossPng'
        QtMocHelpers::SlotData<void()>(45, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportBoundsPng'
        QtMocHelpers::SlotData<void()>(46, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportEvalXlsx'
        QtMocHelpers::SlotData<void()>(47, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportCompXlsx'
        QtMocHelpers::SlotData<void()>(48, 2, QMC::AccessPrivate, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<EvolBoundsPanel, qt_meta_tag_ZN12NeuralStudio15EvolBoundsPanelE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::EvolBoundsPanel::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio15EvolBoundsPanelE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio15EvolBoundsPanelE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio15EvolBoundsPanelE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::EvolBoundsPanel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<EvolBoundsPanel *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->runningStateChanged((*reinterpret_cast<std::add_pointer_t<bool>>(_a[1]))); break;
        case 1: _t->trainingCompleted((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<std::shared_ptr<Dataset>>>(_a[2]))); break;
        case 2: _t->onAddFiles(); break;
        case 3: _t->onClearQueue(); break;
        case 4: _t->onRemoveSelected(); break;
        case 5: _t->onOpt2Changed((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 6: _t->onRunClicked(); break;
        case 7: _t->onStopClicked(); break;
        case 8: _t->onDatasetStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 9: _t->onPhaseStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 10: _t->onIterationDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[5]))); break;
        case 11: _t->onPhaseFinished((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[2]))); break;
        case 12: _t->onDatasetBoundsReady((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[4]))); break;
        case 13: _t->onDatasetFinished((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QStringList>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[5])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[6])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[7]))); break;
        case 14: _t->onAllDone((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<std::shared_ptr<Dataset>>>(_a[2]))); break;
        case 15: _t->onError((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 16: _t->onExportLossPng(); break;
        case 17: _t->onExportBoundsPng(); break;
        case 18: _t->onExportEvalXlsx(); break;
        case 19: _t->onExportCompXlsx(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 12:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 2:
            case 1:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QList<double> >(); break;
            }
            break;
        case 13:
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
        if (QtMocHelpers::indexOfMethod<void (EvolBoundsPanel::*)(bool )>(_a, &EvolBoundsPanel::runningStateChanged, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (EvolBoundsPanel::*)(std::shared_ptr<NeuralNetwork> , std::shared_ptr<Dataset> )>(_a, &EvolBoundsPanel::trainingCompleted, 1))
            return;
    }
}

const QMetaObject *NeuralStudio::EvolBoundsPanel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::EvolBoundsPanel::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio15EvolBoundsPanelE_t>.strings))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int NeuralStudio::EvolBoundsPanel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 20)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 20;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 20)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 20;
    }
    return _id;
}

// SIGNAL 0
void NeuralStudio::EvolBoundsPanel::runningStateChanged(bool _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1);
}

// SIGNAL 1
void NeuralStudio::EvolBoundsPanel::trainingCompleted(std::shared_ptr<NeuralNetwork> _t1, std::shared_ptr<Dataset> _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1, _t2);
}
QT_WARNING_POP

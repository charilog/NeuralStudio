/****************************************************************************
** Meta object code from reading C++ file 'PreTrainWorker.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/core/pretrain/PreTrainWorker.h"
#include <QtCore/qmetatype.h>
#include <QtCore/QList>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'PreTrainWorker.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN12NeuralStudio14PreTrainWorkerE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::PreTrainWorker::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio14PreTrainWorkerE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::PreTrainWorker",
        "datasetStarted",
        "",
        "dsIdx",
        "totalDs",
        "dsName",
        "phaseStarted",
        "phase",
        "totalIter",
        "name",
        "iterationDone",
        "iter",
        "loss",
        "valLoss",
        "phaseFinished",
        "finalLoss",
        "datasetFinished",
        "configs",
        "QList<double>",
        "trainLoss",
        "trainErr",
        "valErr",
        "testErr",
        "isClassification",
        "allDone",
        "std::shared_ptr<NeuralNetwork>",
        "lastNet",
        "std::shared_ptr<Dataset>",
        "lastDs",
        "errorOccurred",
        "msg",
        "run",
        "requestStop"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'datasetStarted'
        QtMocHelpers::SignalData<void(int, int, QString)>(1, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 3 }, { QMetaType::Int, 4 }, { QMetaType::QString, 5 },
        }}),
        // Signal 'phaseStarted'
        QtMocHelpers::SignalData<void(int, int, QString)>(6, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 7 }, { QMetaType::Int, 8 }, { QMetaType::QString, 9 },
        }}),
        // Signal 'iterationDone'
        QtMocHelpers::SignalData<void(int, int, int, double, double)>(10, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 7 }, { QMetaType::Int, 11 }, { QMetaType::Int, 8 }, { QMetaType::Double, 12 },
            { QMetaType::Double, 13 },
        }}),
        // Signal 'phaseFinished'
        QtMocHelpers::SignalData<void(int, double)>(14, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 7 }, { QMetaType::Double, 15 },
        }}),
        // Signal 'datasetFinished'
        QtMocHelpers::SignalData<void(QString, QStringList, QVector<double>, QVector<double>, QVector<double>, QVector<double>, bool)>(16, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::QString, 5 }, { QMetaType::QStringList, 17 }, { 0x80000000 | 18, 19 }, { 0x80000000 | 18, 20 },
            { 0x80000000 | 18, 21 }, { 0x80000000 | 18, 22 }, { QMetaType::Bool, 23 },
        }}),
        // Signal 'allDone'
        QtMocHelpers::SignalData<void(std::shared_ptr<NeuralNetwork>, std::shared_ptr<Dataset>)>(24, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 25, 26 }, { 0x80000000 | 27, 28 },
        }}),
        // Signal 'errorOccurred'
        QtMocHelpers::SignalData<void(QString)>(29, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::QString, 30 },
        }}),
        // Slot 'run'
        QtMocHelpers::SlotData<void()>(31, 2, QMC::AccessPublic, QMetaType::Void),
        // Slot 'requestStop'
        QtMocHelpers::SlotData<void()>(32, 2, QMC::AccessPublic, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<PreTrainWorker, qt_meta_tag_ZN12NeuralStudio14PreTrainWorkerE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::PreTrainWorker::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio14PreTrainWorkerE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio14PreTrainWorkerE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio14PreTrainWorkerE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::PreTrainWorker::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<PreTrainWorker *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->datasetStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 1: _t->phaseStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 2: _t->iterationDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[5]))); break;
        case 3: _t->phaseFinished((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[2]))); break;
        case 4: _t->datasetFinished((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QStringList>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[5])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[6])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[7]))); break;
        case 5: _t->allDone((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<std::shared_ptr<Dataset>>>(_a[2]))); break;
        case 6: _t->errorOccurred((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 7: _t->run(); break;
        case 8: _t->requestStop(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 4:
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
        if (QtMocHelpers::indexOfMethod<void (PreTrainWorker::*)(int , int , QString )>(_a, &PreTrainWorker::datasetStarted, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (PreTrainWorker::*)(int , int , QString )>(_a, &PreTrainWorker::phaseStarted, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (PreTrainWorker::*)(int , int , int , double , double )>(_a, &PreTrainWorker::iterationDone, 2))
            return;
        if (QtMocHelpers::indexOfMethod<void (PreTrainWorker::*)(int , double )>(_a, &PreTrainWorker::phaseFinished, 3))
            return;
        if (QtMocHelpers::indexOfMethod<void (PreTrainWorker::*)(QString , QStringList , QVector<double> , QVector<double> , QVector<double> , QVector<double> , bool )>(_a, &PreTrainWorker::datasetFinished, 4))
            return;
        if (QtMocHelpers::indexOfMethod<void (PreTrainWorker::*)(std::shared_ptr<NeuralNetwork> , std::shared_ptr<Dataset> )>(_a, &PreTrainWorker::allDone, 5))
            return;
        if (QtMocHelpers::indexOfMethod<void (PreTrainWorker::*)(QString )>(_a, &PreTrainWorker::errorOccurred, 6))
            return;
    }
}

const QMetaObject *NeuralStudio::PreTrainWorker::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::PreTrainWorker::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio14PreTrainWorkerE_t>.strings))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int NeuralStudio::PreTrainWorker::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void NeuralStudio::PreTrainWorker::datasetStarted(int _t1, int _t2, QString _t3)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1, _t2, _t3);
}

// SIGNAL 1
void NeuralStudio::PreTrainWorker::phaseStarted(int _t1, int _t2, QString _t3)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1, _t2, _t3);
}

// SIGNAL 2
void NeuralStudio::PreTrainWorker::iterationDone(int _t1, int _t2, int _t3, double _t4, double _t5)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 2, nullptr, _t1, _t2, _t3, _t4, _t5);
}

// SIGNAL 3
void NeuralStudio::PreTrainWorker::phaseFinished(int _t1, double _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 3, nullptr, _t1, _t2);
}

// SIGNAL 4
void NeuralStudio::PreTrainWorker::datasetFinished(QString _t1, QStringList _t2, QVector<double> _t3, QVector<double> _t4, QVector<double> _t5, QVector<double> _t6, bool _t7)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 4, nullptr, _t1, _t2, _t3, _t4, _t5, _t6, _t7);
}

// SIGNAL 5
void NeuralStudio::PreTrainWorker::allDone(std::shared_ptr<NeuralNetwork> _t1, std::shared_ptr<Dataset> _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 5, nullptr, _t1, _t2);
}

// SIGNAL 6
void NeuralStudio::PreTrainWorker::errorOccurred(QString _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 6, nullptr, _t1);
}
QT_WARNING_POP

/****************************************************************************
** Meta object code from reading C++ file 'GEStudioWorker.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/core/ge/GEStudioWorker.h"
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GEStudioWorker.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN12NeuralStudio14GEStudioWorkerE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::GEStudioWorker::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio14GEStudioWorkerE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::GEStudioWorker",
        "datasetStarted",
        "",
        "dsIdx",
        "totalDs",
        "dsName",
        "info",
        "generationDone",
        "gen",
        "totalGens",
        "bestFitness",
        "bestDesc",
        "meanFitness",
        "diversity",
        "uniqueEvals",
        "cacheHits",
        "innerCalls",
        "finalStageStarted",
        "totalRows",
        "finalRowDone",
        "row",
        "label",
        "datasetFinished",
        "GEDatasetResult",
        "result",
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
        QtMocHelpers::SignalData<void(int, int, QString, QString)>(1, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 3 }, { QMetaType::Int, 4 }, { QMetaType::QString, 5 }, { QMetaType::QString, 6 },
        }}),
        // Signal 'generationDone'
        QtMocHelpers::SignalData<void(int, int, double, QString, double, double, int, int, qint64)>(7, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 8 }, { QMetaType::Int, 9 }, { QMetaType::Double, 10 }, { QMetaType::QString, 11 },
            { QMetaType::Double, 12 }, { QMetaType::Double, 13 }, { QMetaType::Int, 14 }, { QMetaType::Int, 15 },
            { QMetaType::LongLong, 16 },
        }}),
        // Signal 'finalStageStarted'
        QtMocHelpers::SignalData<void(int)>(17, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 18 },
        }}),
        // Signal 'finalRowDone'
        QtMocHelpers::SignalData<void(int, int, QString)>(19, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 20 }, { QMetaType::Int, 18 }, { QMetaType::QString, 21 },
        }}),
        // Signal 'datasetFinished'
        QtMocHelpers::SignalData<void(GEDatasetResult)>(22, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 23, 24 },
        }}),
        // Signal 'allDone'
        QtMocHelpers::SignalData<void(std::shared_ptr<NeuralNetwork>, std::shared_ptr<Dataset>)>(25, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 26, 27 }, { 0x80000000 | 28, 29 },
        }}),
        // Signal 'errorOccurred'
        QtMocHelpers::SignalData<void(QString)>(30, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::QString, 31 },
        }}),
        // Slot 'run'
        QtMocHelpers::SlotData<void()>(32, 2, QMC::AccessPublic, QMetaType::Void),
        // Slot 'requestStop'
        QtMocHelpers::SlotData<void()>(33, 2, QMC::AccessPublic, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<GEStudioWorker, qt_meta_tag_ZN12NeuralStudio14GEStudioWorkerE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::GEStudioWorker::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio14GEStudioWorkerE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio14GEStudioWorkerE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio14GEStudioWorkerE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::GEStudioWorker::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<GEStudioWorker *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->datasetStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4]))); break;
        case 1: _t->generationDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[5])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[6])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[7])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[8])),(*reinterpret_cast<std::add_pointer_t<qint64>>(_a[9]))); break;
        case 2: _t->finalStageStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 3: _t->finalRowDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 4: _t->datasetFinished((*reinterpret_cast<std::add_pointer_t<GEDatasetResult>>(_a[1]))); break;
        case 5: _t->allDone((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<std::shared_ptr<Dataset>>>(_a[2]))); break;
        case 6: _t->errorOccurred((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 7: _t->run(); break;
        case 8: _t->requestStop(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (GEStudioWorker::*)(int , int , QString , QString )>(_a, &GEStudioWorker::datasetStarted, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (GEStudioWorker::*)(int , int , double , QString , double , double , int , int , qint64 )>(_a, &GEStudioWorker::generationDone, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (GEStudioWorker::*)(int )>(_a, &GEStudioWorker::finalStageStarted, 2))
            return;
        if (QtMocHelpers::indexOfMethod<void (GEStudioWorker::*)(int , int , QString )>(_a, &GEStudioWorker::finalRowDone, 3))
            return;
        if (QtMocHelpers::indexOfMethod<void (GEStudioWorker::*)(GEDatasetResult )>(_a, &GEStudioWorker::datasetFinished, 4))
            return;
        if (QtMocHelpers::indexOfMethod<void (GEStudioWorker::*)(std::shared_ptr<NeuralNetwork> , std::shared_ptr<Dataset> )>(_a, &GEStudioWorker::allDone, 5))
            return;
        if (QtMocHelpers::indexOfMethod<void (GEStudioWorker::*)(QString )>(_a, &GEStudioWorker::errorOccurred, 6))
            return;
    }
}

const QMetaObject *NeuralStudio::GEStudioWorker::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::GEStudioWorker::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio14GEStudioWorkerE_t>.strings))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int NeuralStudio::GEStudioWorker::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void NeuralStudio::GEStudioWorker::datasetStarted(int _t1, int _t2, QString _t3, QString _t4)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1, _t2, _t3, _t4);
}

// SIGNAL 1
void NeuralStudio::GEStudioWorker::generationDone(int _t1, int _t2, double _t3, QString _t4, double _t5, double _t6, int _t7, int _t8, qint64 _t9)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9);
}

// SIGNAL 2
void NeuralStudio::GEStudioWorker::finalStageStarted(int _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 2, nullptr, _t1);
}

// SIGNAL 3
void NeuralStudio::GEStudioWorker::finalRowDone(int _t1, int _t2, QString _t3)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 3, nullptr, _t1, _t2, _t3);
}

// SIGNAL 4
void NeuralStudio::GEStudioWorker::datasetFinished(GEDatasetResult _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 4, nullptr, _t1);
}

// SIGNAL 5
void NeuralStudio::GEStudioWorker::allDone(std::shared_ptr<NeuralNetwork> _t1, std::shared_ptr<Dataset> _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 5, nullptr, _t1, _t2);
}

// SIGNAL 6
void NeuralStudio::GEStudioWorker::errorOccurred(QString _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 6, nullptr, _t1);
}
QT_WARNING_POP

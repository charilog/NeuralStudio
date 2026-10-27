/****************************************************************************
** Meta object code from reading C++ file 'ArchSearchWorker.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/core/arch/ArchSearchWorker.h"
#include <QtCore/qmetatype.h>
#include <QtCore/QList>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ArchSearchWorker.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN12NeuralStudio16ArchSearchWorkerE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::ArchSearchWorker::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio16ArchSearchWorkerE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::ArchSearchWorker",
        "datasetStarted",
        "",
        "dsIdx",
        "totalDs",
        "dsName",
        "evalDone",
        "evalNum",
        "totalEvals",
        "fitness",
        "archDesc",
        "cacheHit",
        "generationDone",
        "gen",
        "totalGens",
        "bestFitness",
        "bestDesc",
        "uniqueEvals",
        "cacheHits",
        "datasetFinished",
        "QList<ArchResult>",
        "topResults",
        "QList<double>",
        "convergence",
        "isClassification",
        "allDone",
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
        // Signal 'evalDone'
        QtMocHelpers::SignalData<void(int, int, double, QString, bool)>(6, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 7 }, { QMetaType::Int, 8 }, { QMetaType::Double, 9 }, { QMetaType::QString, 10 },
            { QMetaType::Bool, 11 },
        }}),
        // Signal 'generationDone'
        QtMocHelpers::SignalData<void(int, int, double, QString, int, int)>(12, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 13 }, { QMetaType::Int, 14 }, { QMetaType::Double, 15 }, { QMetaType::QString, 16 },
            { QMetaType::Int, 17 }, { QMetaType::Int, 18 },
        }}),
        // Signal 'datasetFinished'
        QtMocHelpers::SignalData<void(QString, QVector<ArchResult>, QVector<double>, bool)>(19, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::QString, 5 }, { 0x80000000 | 20, 21 }, { 0x80000000 | 22, 23 }, { QMetaType::Bool, 24 },
        }}),
        // Signal 'allDone'
        QtMocHelpers::SignalData<void()>(25, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'errorOccurred'
        QtMocHelpers::SignalData<void(QString)>(26, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::QString, 27 },
        }}),
        // Slot 'run'
        QtMocHelpers::SlotData<void()>(28, 2, QMC::AccessPublic, QMetaType::Void),
        // Slot 'requestStop'
        QtMocHelpers::SlotData<void()>(29, 2, QMC::AccessPublic, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<ArchSearchWorker, qt_meta_tag_ZN12NeuralStudio16ArchSearchWorkerE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::ArchSearchWorker::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio16ArchSearchWorkerE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio16ArchSearchWorkerE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio16ArchSearchWorkerE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::ArchSearchWorker::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<ArchSearchWorker *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->datasetStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 1: _t->evalDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[5]))); break;
        case 2: _t->generationDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[5])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[6]))); break;
        case 3: _t->datasetFinished((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QList<ArchResult>>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[4]))); break;
        case 4: _t->allDone(); break;
        case 5: _t->errorOccurred((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 6: _t->run(); break;
        case 7: _t->requestStop(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 2:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QList<double> >(); break;
            }
            break;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (ArchSearchWorker::*)(int , int , QString )>(_a, &ArchSearchWorker::datasetStarted, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (ArchSearchWorker::*)(int , int , double , QString , bool )>(_a, &ArchSearchWorker::evalDone, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (ArchSearchWorker::*)(int , int , double , QString , int , int )>(_a, &ArchSearchWorker::generationDone, 2))
            return;
        if (QtMocHelpers::indexOfMethod<void (ArchSearchWorker::*)(QString , QVector<ArchResult> , QVector<double> , bool )>(_a, &ArchSearchWorker::datasetFinished, 3))
            return;
        if (QtMocHelpers::indexOfMethod<void (ArchSearchWorker::*)()>(_a, &ArchSearchWorker::allDone, 4))
            return;
        if (QtMocHelpers::indexOfMethod<void (ArchSearchWorker::*)(QString )>(_a, &ArchSearchWorker::errorOccurred, 5))
            return;
    }
}

const QMetaObject *NeuralStudio::ArchSearchWorker::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::ArchSearchWorker::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio16ArchSearchWorkerE_t>.strings))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int NeuralStudio::ArchSearchWorker::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void NeuralStudio::ArchSearchWorker::datasetStarted(int _t1, int _t2, QString _t3)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1, _t2, _t3);
}

// SIGNAL 1
void NeuralStudio::ArchSearchWorker::evalDone(int _t1, int _t2, double _t3, QString _t4, bool _t5)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1, _t2, _t3, _t4, _t5);
}

// SIGNAL 2
void NeuralStudio::ArchSearchWorker::generationDone(int _t1, int _t2, double _t3, QString _t4, int _t5, int _t6)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 2, nullptr, _t1, _t2, _t3, _t4, _t5, _t6);
}

// SIGNAL 3
void NeuralStudio::ArchSearchWorker::datasetFinished(QString _t1, QVector<ArchResult> _t2, QVector<double> _t3, bool _t4)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 3, nullptr, _t1, _t2, _t3, _t4);
}

// SIGNAL 4
void NeuralStudio::ArchSearchWorker::allDone()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}

// SIGNAL 5
void NeuralStudio::ArchSearchWorker::errorOccurred(QString _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 5, nullptr, _t1);
}
QT_WARNING_POP

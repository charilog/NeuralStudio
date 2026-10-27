/****************************************************************************
** Meta object code from reading C++ file 'ArchSearchPanel.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/ui/panels/ArchSearchPanel.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>
#include <QtCore/QList>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ArchSearchPanel.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN12NeuralStudio15ArchSearchPanelE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::ArchSearchPanel::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio15ArchSearchPanelE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::ArchSearchPanel",
        "runningStateChanged",
        "",
        "running",
        "applyToMode1",
        "datasetPath",
        "QList<int>",
        "hiddenSizes",
        "actNames",
        "onAddFiles",
        "onClearQueue",
        "onRemoveSelected",
        "onRunClicked",
        "onStopClicked",
        "onDatasetStarted",
        "dsIdx",
        "totalDs",
        "dsName",
        "onEvalDone",
        "evalNum",
        "totalEvals",
        "fitness",
        "archDesc",
        "cacheHit",
        "onGenerationDone",
        "gen",
        "totalGens",
        "bestFitness",
        "bestDesc",
        "uniqueEvals",
        "cacheHits",
        "onDatasetFinished",
        "QList<ArchResult>",
        "topResults",
        "QList<double>",
        "convergence",
        "isClassification",
        "onAllDone",
        "onError",
        "msg",
        "onExportResultsXlsx"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'runningStateChanged'
        QtMocHelpers::SignalData<void(bool)>(1, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Bool, 3 },
        }}),
        // Signal 'applyToMode1'
        QtMocHelpers::SignalData<void(QString, QVector<int>, QVector<QString>)>(4, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::QString, 5 }, { 0x80000000 | 6, 7 }, { QMetaType::QStringList, 8 },
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
        // Slot 'onEvalDone'
        QtMocHelpers::SlotData<void(int, int, double, QString, bool)>(18, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 19 }, { QMetaType::Int, 20 }, { QMetaType::Double, 21 }, { QMetaType::QString, 22 },
            { QMetaType::Bool, 23 },
        }}),
        // Slot 'onGenerationDone'
        QtMocHelpers::SlotData<void(int, int, double, QString, int, int)>(24, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 25 }, { QMetaType::Int, 26 }, { QMetaType::Double, 27 }, { QMetaType::QString, 28 },
            { QMetaType::Int, 29 }, { QMetaType::Int, 30 },
        }}),
        // Slot 'onDatasetFinished'
        QtMocHelpers::SlotData<void(QString, QVector<ArchResult>, QVector<double>, bool)>(31, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 17 }, { 0x80000000 | 32, 33 }, { 0x80000000 | 34, 35 }, { QMetaType::Bool, 36 },
        }}),
        // Slot 'onAllDone'
        QtMocHelpers::SlotData<void()>(37, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onError'
        QtMocHelpers::SlotData<void(QString)>(38, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 39 },
        }}),
        // Slot 'onExportResultsXlsx'
        QtMocHelpers::SlotData<void()>(40, 2, QMC::AccessPrivate, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<ArchSearchPanel, qt_meta_tag_ZN12NeuralStudio15ArchSearchPanelE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::ArchSearchPanel::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio15ArchSearchPanelE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio15ArchSearchPanelE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio15ArchSearchPanelE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::ArchSearchPanel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<ArchSearchPanel *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->runningStateChanged((*reinterpret_cast<std::add_pointer_t<bool>>(_a[1]))); break;
        case 1: _t->applyToMode1((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QList<int>>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QList<QString>>>(_a[3]))); break;
        case 2: _t->onAddFiles(); break;
        case 3: _t->onClearQueue(); break;
        case 4: _t->onRemoveSelected(); break;
        case 5: _t->onRunClicked(); break;
        case 6: _t->onStopClicked(); break;
        case 7: _t->onDatasetStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 8: _t->onEvalDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[5]))); break;
        case 9: _t->onGenerationDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[5])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[6]))); break;
        case 10: _t->onDatasetFinished((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QList<ArchResult>>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QList<double>>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[4]))); break;
        case 11: _t->onAllDone(); break;
        case 12: _t->onError((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 13: _t->onExportResultsXlsx(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 1:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QList<int> >(); break;
            }
            break;
        case 10:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 2:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QList<double> >(); break;
            }
            break;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (ArchSearchPanel::*)(bool )>(_a, &ArchSearchPanel::runningStateChanged, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (ArchSearchPanel::*)(QString , QVector<int> , QVector<QString> )>(_a, &ArchSearchPanel::applyToMode1, 1))
            return;
    }
}

const QMetaObject *NeuralStudio::ArchSearchPanel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::ArchSearchPanel::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio15ArchSearchPanelE_t>.strings))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int NeuralStudio::ArchSearchPanel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 14)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 14)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    }
    return _id;
}

// SIGNAL 0
void NeuralStudio::ArchSearchPanel::runningStateChanged(bool _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1);
}

// SIGNAL 1
void NeuralStudio::ArchSearchPanel::applyToMode1(QString _t1, QVector<int> _t2, QVector<QString> _t3)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1, _t2, _t3);
}
QT_WARNING_POP

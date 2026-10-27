/****************************************************************************
** Meta object code from reading C++ file 'GEStudioPanel.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/ui/panels/GEStudioPanel.h"
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GEStudioPanel.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN12NeuralStudio13GEStudioPanelE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::GEStudioPanel::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio13GEStudioPanelE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::GEStudioPanel",
        "runningStateChanged",
        "",
        "running",
        "trainingCompleted",
        "std::shared_ptr<NeuralNetwork>",
        "net",
        "std::shared_ptr<Dataset>",
        "ds",
        "onAddFiles",
        "onRemoveSelected",
        "onClearQueue",
        "onPresetChanged",
        "idx",
        "onValidateGrammar",
        "onRun",
        "onStop",
        "onGenotypeChanged",
        "onDatasetStarted",
        "dsIdx",
        "totalDs",
        "dsName",
        "info",
        "onGenerationDone",
        "gen",
        "totalGens",
        "bestFitness",
        "bestDesc",
        "meanFitness",
        "diversity",
        "uniqueEvals",
        "cacheHits",
        "innerCalls",
        "onFinalStageStarted",
        "totalRows",
        "onFinalRowDone",
        "row",
        "label",
        "onDatasetFinished",
        "GEDatasetResult",
        "r",
        "onAllDone",
        "onError",
        "msg",
        "onExportXlsx",
        "onBatchRun",
        "onBatchCheck",
        "onBatchBrowseDir",
        "onBatchOpenResults",
        "onBatchUsage",
        "onBatchStarted",
        "jobsToDo",
        "alreadyDone",
        "onBatchProgress",
        "jobIdx",
        "withinJob",
        "onBatchJobFinished",
        "jobsDone",
        "line",
        "ok",
        "onBatchMessage",
        "text",
        "onBatchFinished",
        "stopped",
        "prefix",
        "onBatchError",
        "onRunClicked",
        "onExportClicked",
        "onSetupTabChanged",
        "onNewEmpty",
        "onNewBenchmark",
        "onOpen",
        "onEdit",
        "onSave",
        "onSaveAs",
        "onExportTables",
        "onExperimentModelsPicked"
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
        // Slot 'onRemoveSelected'
        QtMocHelpers::SlotData<void()>(10, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onClearQueue'
        QtMocHelpers::SlotData<void()>(11, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onPresetChanged'
        QtMocHelpers::SlotData<void(int)>(12, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 13 },
        }}),
        // Slot 'onValidateGrammar'
        QtMocHelpers::SlotData<void()>(14, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onRun'
        QtMocHelpers::SlotData<void()>(15, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onStop'
        QtMocHelpers::SlotData<void()>(16, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onGenotypeChanged'
        QtMocHelpers::SlotData<void()>(17, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onDatasetStarted'
        QtMocHelpers::SlotData<void(int, int, QString, QString)>(18, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 19 }, { QMetaType::Int, 20 }, { QMetaType::QString, 21 }, { QMetaType::QString, 22 },
        }}),
        // Slot 'onGenerationDone'
        QtMocHelpers::SlotData<void(int, int, double, QString, double, double, int, int, qint64)>(23, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 24 }, { QMetaType::Int, 25 }, { QMetaType::Double, 26 }, { QMetaType::QString, 27 },
            { QMetaType::Double, 28 }, { QMetaType::Double, 29 }, { QMetaType::Int, 30 }, { QMetaType::Int, 31 },
            { QMetaType::LongLong, 32 },
        }}),
        // Slot 'onFinalStageStarted'
        QtMocHelpers::SlotData<void(int)>(33, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 34 },
        }}),
        // Slot 'onFinalRowDone'
        QtMocHelpers::SlotData<void(int, int, QString)>(35, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 36 }, { QMetaType::Int, 34 }, { QMetaType::QString, 37 },
        }}),
        // Slot 'onDatasetFinished'
        QtMocHelpers::SlotData<void(GEDatasetResult)>(38, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 39, 40 },
        }}),
        // Slot 'onAllDone'
        QtMocHelpers::SlotData<void(std::shared_ptr<NeuralNetwork>, std::shared_ptr<Dataset>)>(41, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 5, 6 }, { 0x80000000 | 7, 8 },
        }}),
        // Slot 'onError'
        QtMocHelpers::SlotData<void(QString)>(42, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 43 },
        }}),
        // Slot 'onExportXlsx'
        QtMocHelpers::SlotData<void()>(44, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onBatchRun'
        QtMocHelpers::SlotData<void()>(45, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onBatchCheck'
        QtMocHelpers::SlotData<void()>(46, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onBatchBrowseDir'
        QtMocHelpers::SlotData<void()>(47, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onBatchOpenResults'
        QtMocHelpers::SlotData<void()>(48, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onBatchUsage'
        QtMocHelpers::SlotData<void()>(49, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onBatchStarted'
        QtMocHelpers::SlotData<void(int, int)>(50, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 51 }, { QMetaType::Int, 52 },
        }}),
        // Slot 'onBatchProgress'
        QtMocHelpers::SlotData<void(int, int, double, QString)>(53, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 54 }, { QMetaType::Int, 51 }, { QMetaType::Double, 55 }, { QMetaType::QString, 37 },
        }}),
        // Slot 'onBatchJobFinished'
        QtMocHelpers::SlotData<void(int, int, QString, bool)>(56, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 57 }, { QMetaType::Int, 51 }, { QMetaType::QString, 58 }, { QMetaType::Bool, 59 },
        }}),
        // Slot 'onBatchMessage'
        QtMocHelpers::SlotData<void(QString)>(60, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 61 },
        }}),
        // Slot 'onBatchFinished'
        QtMocHelpers::SlotData<void(bool, QString)>(62, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 63 }, { QMetaType::QString, 64 },
        }}),
        // Slot 'onBatchError'
        QtMocHelpers::SlotData<void(QString)>(65, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 43 },
        }}),
        // Slot 'onRunClicked'
        QtMocHelpers::SlotData<void()>(66, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportClicked'
        QtMocHelpers::SlotData<void()>(67, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onSetupTabChanged'
        QtMocHelpers::SlotData<void(int)>(68, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 13 },
        }}),
        // Slot 'onNewEmpty'
        QtMocHelpers::SlotData<void()>(69, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onNewBenchmark'
        QtMocHelpers::SlotData<void()>(70, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onOpen'
        QtMocHelpers::SlotData<void()>(71, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onEdit'
        QtMocHelpers::SlotData<void()>(72, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onSave'
        QtMocHelpers::SlotData<void()>(73, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onSaveAs'
        QtMocHelpers::SlotData<void()>(74, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportTables'
        QtMocHelpers::SlotData<void()>(75, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExperimentModelsPicked'
        QtMocHelpers::SlotData<void()>(76, 2, QMC::AccessPrivate, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<GEStudioPanel, qt_meta_tag_ZN12NeuralStudio13GEStudioPanelE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::GEStudioPanel::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio13GEStudioPanelE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio13GEStudioPanelE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio13GEStudioPanelE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::GEStudioPanel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<GEStudioPanel *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->runningStateChanged((*reinterpret_cast<std::add_pointer_t<bool>>(_a[1]))); break;
        case 1: _t->trainingCompleted((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<std::shared_ptr<Dataset>>>(_a[2]))); break;
        case 2: _t->onAddFiles(); break;
        case 3: _t->onRemoveSelected(); break;
        case 4: _t->onClearQueue(); break;
        case 5: _t->onPresetChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 6: _t->onValidateGrammar(); break;
        case 7: _t->onRun(); break;
        case 8: _t->onStop(); break;
        case 9: _t->onGenotypeChanged(); break;
        case 10: _t->onDatasetStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4]))); break;
        case 11: _t->onGenerationDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[5])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[6])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[7])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[8])),(*reinterpret_cast<std::add_pointer_t<qint64>>(_a[9]))); break;
        case 12: _t->onFinalStageStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 13: _t->onFinalRowDone((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 14: _t->onDatasetFinished((*reinterpret_cast<std::add_pointer_t<GEDatasetResult>>(_a[1]))); break;
        case 15: _t->onAllDone((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<std::shared_ptr<Dataset>>>(_a[2]))); break;
        case 16: _t->onError((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 17: _t->onExportXlsx(); break;
        case 18: _t->onBatchRun(); break;
        case 19: _t->onBatchCheck(); break;
        case 20: _t->onBatchBrowseDir(); break;
        case 21: _t->onBatchOpenResults(); break;
        case 22: _t->onBatchUsage(); break;
        case 23: _t->onBatchStarted((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2]))); break;
        case 24: _t->onBatchProgress((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4]))); break;
        case 25: _t->onBatchJobFinished((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[4]))); break;
        case 26: _t->onBatchMessage((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 27: _t->onBatchFinished((*reinterpret_cast<std::add_pointer_t<bool>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[2]))); break;
        case 28: _t->onBatchError((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 29: _t->onRunClicked(); break;
        case 30: _t->onExportClicked(); break;
        case 31: _t->onSetupTabChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 32: _t->onNewEmpty(); break;
        case 33: _t->onNewBenchmark(); break;
        case 34: _t->onOpen(); break;
        case 35: _t->onEdit(); break;
        case 36: _t->onSave(); break;
        case 37: _t->onSaveAs(); break;
        case 38: _t->onExportTables(); break;
        case 39: _t->onExperimentModelsPicked(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (GEStudioPanel::*)(bool )>(_a, &GEStudioPanel::runningStateChanged, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (GEStudioPanel::*)(std::shared_ptr<NeuralNetwork> , std::shared_ptr<Dataset> )>(_a, &GEStudioPanel::trainingCompleted, 1))
            return;
    }
}

const QMetaObject *NeuralStudio::GEStudioPanel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::GEStudioPanel::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio13GEStudioPanelE_t>.strings))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int NeuralStudio::GEStudioPanel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 40)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 40;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 40)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 40;
    }
    return _id;
}

// SIGNAL 0
void NeuralStudio::GEStudioPanel::runningStateChanged(bool _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1);
}

// SIGNAL 1
void NeuralStudio::GEStudioPanel::trainingCompleted(std::shared_ptr<NeuralNetwork> _t1, std::shared_ptr<Dataset> _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1, _t2);
}
QT_WARNING_POP

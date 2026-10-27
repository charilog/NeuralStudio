/****************************************************************************
** Meta object code from reading C++ file 'TrainingPanel.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../src/ui/panels/TrainingPanel.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'TrainingPanel.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN12NeuralStudio13TrainingPanelE_t {};
} // unnamed namespace

template <> constexpr inline auto NeuralStudio::TrainingPanel::qt_create_metaobjectdata<qt_meta_tag_ZN12NeuralStudio13TrainingPanelE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "NeuralStudio::TrainingPanel",
        "trainingCompleted",
        "",
        "requestStop",
        "rbfTrainingCompleted",
        "std::shared_ptr<RBFNetwork>",
        "net",
        "trainErr",
        "valErr",
        "onNetworkBuilt",
        "std::shared_ptr<NeuralNetwork>",
        "onRBFBuilt",
        "onDatasetLoaded",
        "const Dataset*",
        "ds",
        "onTestDatasetLoaded",
        "testDs",
        "onTrainClicked",
        "onCVClicked",
        "onExportLogClicked",
        "onOptimizerChanged",
        "index",
        "onEpochCompleted",
        "EpochResult",
        "result",
        "onTrainingFinished",
        "finalVal",
        "onTrainingError",
        "msg",
        "onFoldCompleted",
        "FoldResult",
        "fr",
        "onCVFinished",
        "CVSummary",
        "sum",
        "onCVError"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'trainingCompleted'
        QtMocHelpers::SignalData<void()>(1, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'requestStop'
        QtMocHelpers::SignalData<void()>(3, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'rbfTrainingCompleted'
        QtMocHelpers::SignalData<void(std::shared_ptr<RBFNetwork>, double, double)>(4, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 5, 6 }, { QMetaType::Double, 7 }, { QMetaType::Double, 8 },
        }}),
        // Slot 'onNetworkBuilt'
        QtMocHelpers::SlotData<void(std::shared_ptr<NeuralNetwork>)>(9, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 10, 6 },
        }}),
        // Slot 'onRBFBuilt'
        QtMocHelpers::SlotData<void(std::shared_ptr<RBFNetwork>)>(11, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 5, 6 },
        }}),
        // Slot 'onDatasetLoaded'
        QtMocHelpers::SlotData<void(const Dataset *)>(12, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 13, 14 },
        }}),
        // Slot 'onTestDatasetLoaded'
        QtMocHelpers::SlotData<void(const Dataset *)>(15, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 13, 16 },
        }}),
        // Slot 'onTrainClicked'
        QtMocHelpers::SlotData<void()>(17, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onCVClicked'
        QtMocHelpers::SlotData<void()>(18, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onExportLogClicked'
        QtMocHelpers::SlotData<void()>(19, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onOptimizerChanged'
        QtMocHelpers::SlotData<void(int)>(20, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 21 },
        }}),
        // Slot 'onEpochCompleted'
        QtMocHelpers::SlotData<void(EpochResult)>(22, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 23, 24 },
        }}),
        // Slot 'onTrainingFinished'
        QtMocHelpers::SlotData<void(double)>(25, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Double, 26 },
        }}),
        // Slot 'onTrainingError'
        QtMocHelpers::SlotData<void(QString)>(27, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 28 },
        }}),
        // Slot 'onFoldCompleted'
        QtMocHelpers::SlotData<void(FoldResult)>(29, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 30, 31 },
        }}),
        // Slot 'onCVFinished'
        QtMocHelpers::SlotData<void(CVSummary)>(32, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { 0x80000000 | 33, 34 },
        }}),
        // Slot 'onCVError'
        QtMocHelpers::SlotData<void(QString)>(35, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 28 },
        }}),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<TrainingPanel, qt_meta_tag_ZN12NeuralStudio13TrainingPanelE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject NeuralStudio::TrainingPanel::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio13TrainingPanelE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio13TrainingPanelE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12NeuralStudio13TrainingPanelE_t>.metaTypes,
    nullptr
} };

void NeuralStudio::TrainingPanel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<TrainingPanel *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->trainingCompleted(); break;
        case 1: _t->requestStop(); break;
        case 2: _t->rbfTrainingCompleted((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<RBFNetwork>>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3]))); break;
        case 3: _t->onNetworkBuilt((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<NeuralNetwork>>>(_a[1]))); break;
        case 4: _t->onRBFBuilt((*reinterpret_cast<std::add_pointer_t<std::shared_ptr<RBFNetwork>>>(_a[1]))); break;
        case 5: _t->onDatasetLoaded((*reinterpret_cast<std::add_pointer_t<const Dataset*>>(_a[1]))); break;
        case 6: _t->onTestDatasetLoaded((*reinterpret_cast<std::add_pointer_t<const Dataset*>>(_a[1]))); break;
        case 7: _t->onTrainClicked(); break;
        case 8: _t->onCVClicked(); break;
        case 9: _t->onExportLogClicked(); break;
        case 10: _t->onOptimizerChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 11: _t->onEpochCompleted((*reinterpret_cast<std::add_pointer_t<EpochResult>>(_a[1]))); break;
        case 12: _t->onTrainingFinished((*reinterpret_cast<std::add_pointer_t<double>>(_a[1]))); break;
        case 13: _t->onTrainingError((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 14: _t->onFoldCompleted((*reinterpret_cast<std::add_pointer_t<FoldResult>>(_a[1]))); break;
        case 15: _t->onCVFinished((*reinterpret_cast<std::add_pointer_t<CVSummary>>(_a[1]))); break;
        case 16: _t->onCVError((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        default: ;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (TrainingPanel::*)()>(_a, &TrainingPanel::trainingCompleted, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (TrainingPanel::*)()>(_a, &TrainingPanel::requestStop, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (TrainingPanel::*)(std::shared_ptr<RBFNetwork> , double , double )>(_a, &TrainingPanel::rbfTrainingCompleted, 2))
            return;
    }
}

const QMetaObject *NeuralStudio::TrainingPanel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *NeuralStudio::TrainingPanel::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12NeuralStudio13TrainingPanelE_t>.strings))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int NeuralStudio::TrainingPanel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 17)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 17;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 17)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 17;
    }
    return _id;
}

// SIGNAL 0
void NeuralStudio::TrainingPanel::trainingCompleted()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void NeuralStudio::TrainingPanel::requestStop()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void NeuralStudio::TrainingPanel::rbfTrainingCompleted(std::shared_ptr<RBFNetwork> _t1, double _t2, double _t3)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 2, nullptr, _t1, _t2, _t3);
}
QT_WARNING_POP

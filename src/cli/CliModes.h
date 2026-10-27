#pragma once
// CliModes.h — the commands of neuralstudio-cli, one per mode of the application.
// Each command receives the arguments after its name and returns the process exit code.
#include <QString>
#include <QStringList>

namespace NeuralStudio::cli {

int cmdTrain    (const QStringList& args);   // Mode 1  Single Workflow (MLP / RBF, CV, evaluation, export)
int cmdPredict  (const QStringList& args);   //         apply a saved .nsmodel to a data file
int cmdBatch    (const QStringList& args);   // Mode 2  Batch Run (+ Wilcoxon / Friedman)
int cmdPretrain (const QStringList& args);   // Mode 3  Pre-Train + NNC
int cmdBounds   (const QStringList& args);   // Mode 4  Evolutionary Bounds
int cmdArch     (const QStringList& args);   // Mode 5  Architecture Search
int cmdGE       (const QStringList& args);   // Mode 6  GE Studio (single run or experiment)
int cmdList     (const QStringList& args);   //         optimizers, GE configurations, reference models

} // namespace NeuralStudio::cli

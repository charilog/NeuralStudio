#pragma once

#include <QDebug>
#include <QString>

// ─── NeuralStudio — Logger ─────────────────────────────────────────────────
//  Thin wrappers around QDebug so we can swap backend later (file, remote…).
//  Usage:
//    NS_INFO  << "loaded" << nSamples << "samples";
//    NS_WARN  << "missing value at row" << row;
//    NS_ERROR << "cannot open file:" << path;
// ──────────────────────────────────────────────────────────────────────────────

#define NS_INFO  qInfo().noquote()
#define NS_WARN  qWarning().noquote()
#define NS_ERROR qCritical().noquote()
#define NS_DEBUG qDebug().noquote()

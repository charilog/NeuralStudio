#include "app/MainWindow.h"

#include <QApplication>
#include <QIcon>
#include <QFont>
#include <QStyleFactory>
#include <QPalette>
#include <QColor>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // ── App identity ──────────────────────────────────────────────────────────
    app.setApplicationName("NeuralStudio");
    app.setApplicationVersion("0.6.2");
    app.setOrganizationName("NeuralStudio");

    // ── Icon (embedded SVG via Qt resources) ─────────────────────────────────
    app.setWindowIcon(QIcon(":/icons/neuralstudio.svg"));

    // ── Style (Fusion dark palette) ───────────────────────────────────────────
    app.setStyle(QStyleFactory::create("Fusion"));

    QPalette dark;
    dark.setColor(QPalette::Window,          QColor(30,  32,  46));
    dark.setColor(QPalette::WindowText,      QColor(228, 228, 236));
    dark.setColor(QPalette::Base,            QColor(22,  24,  36));
    dark.setColor(QPalette::AlternateBase,   QColor(30,  32,  46));
    dark.setColor(QPalette::ToolTipBase,     QColor(22,  24,  36));
    dark.setColor(QPalette::ToolTipText,     QColor(228, 228, 236));
    dark.setColor(QPalette::Text,            QColor(228, 228, 236));
    dark.setColor(QPalette::Button,          QColor(38,  40,  58));
    dark.setColor(QPalette::ButtonText,      QColor(228, 228, 236));
    dark.setColor(QPalette::BrightText,      Qt::white);
    dark.setColor(QPalette::Highlight,       QColor(29,  158, 117));
    dark.setColor(QPalette::HighlightedText, Qt::white);
    dark.setColor(QPalette::Link,            QColor(74,  180, 220));
    dark.setColor(QPalette::Mid,             QColor(80,  84,  106));
    dark.setColor(QPalette::Midlight,        QColor(44,  48,  64));
    dark.setColor(QPalette::Dark,            QColor(18,  20,  30));
    dark.setColor(QPalette::Shadow,          QColor(10,  10,  18));
    app.setPalette(dark);

    // ── Show ──────────────────────────────────────────────────────────────────
    NeuralStudio::MainWindow win;
    win.show();

    return app.exec();
}

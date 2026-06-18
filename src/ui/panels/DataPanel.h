#pragma once

#include "core/dataset/Dataset.h"
#include "core/dataset/DatasetBundle.h"
#include <QWidget>
#include <QLabel>
#include <QTableWidget>
#include <QTabWidget>
#include <QPushButton>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <memory>

namespace NeuralStudio {

class DataPanel : public QWidget {
    Q_OBJECT
public:
    explicit DataPanel(QWidget* parent = nullptr);
    const Dataset* dataset()     const { return m_bundle.train.get(); }
    const Dataset* testDataset() const { return m_bundle.test.get();  }

public slots:
    // Public entry point so external code (e.g. recent-files menu) can request a load.
    void loadFile(const QString& path);

signals:
    // Train dataset (always primary)
    void datasetLoaded(const Dataset* ds);
    // Test dataset (may be nullptr if no companion found)
    void testDatasetLoaded(const Dataset* testDs);

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e)           override;

private slots:
    void onLoadClicked();

private:
    QPushButton*  m_loadBtn      = nullptr;
    QLabel*       m_fileLabel    = nullptr;
    QLabel*       m_formatBadge  = nullptr;
    QLabel*       m_companionLbl = nullptr;   // shows auto-loaded companion file
    QLabel*       m_infoBar      = nullptr;
    QLabel*       m_balanceLbl   = nullptr;   // class balance info
    QTabWidget*   m_tabs         = nullptr;
    QTableWidget* m_previewTable = nullptr;
    QTableWidget* m_statsTable   = nullptr;

    DatasetBundle m_bundle;

    void buildUi();
    void populatePreview();
    void populateStats();
    void updateInfoBar();
    void updateBalanceInfo();
    void setStatus(const QString& msg, bool error = false);

    static constexpr int kPreviewRows = 100;
};

} // namespace NeuralStudio

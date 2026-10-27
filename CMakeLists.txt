cmake_minimum_required(VERSION 3.20)
project(NeuralStudio VERSION 1.0.6 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Charts Svg)
find_package(Eigen3 QUIET)
# OpenMP (optional): parallelises the per-sample fitness loop in NSProblem —
# the hot path of every evolutionary optimizer in Modes 1-5. The build works
# identically without it (serial fallback via #ifdef _OPENMP in the code).
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found — parallel fitness evaluation enabled")
endif()
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found")
endif()

set(SOURCES
    src/main.cpp
    src/app/MainWindow.cpp
    src/resources.qrc

    src/core/dataset/Dataset.cpp
    src/core/dataset/DatasetLoader.cpp
    src/core/dataset/CompanionFinder.cpp
    src/core/dataset/loaders/NeuralDesignerLoader.cpp
    src/core/dataset/loaders/CsvLoader.cpp

    src/core/nn/Layer.cpp
    src/core/nn/NeuralNetwork.cpp
    src/core/nn/Trainer.cpp
    src/core/nn/ModelSerializer.cpp
    src/core/nn/RBFNetwork.cpp
    src/core/nn/RBFTrainer.cpp

    src/core/optimizers/NSProblem.cpp
    src/core/optimizers/MetaTrainer.cpp
    src/core/optimizers/lbfgs.cpp
    src/core/optimizers/genetic.cpp
    src/core/optimizers/conjgrad.cpp
    src/core/optimizers/levenberg.cpp
    src/core/optimizers/simann.cpp
    src/core/optimizers/neldermead.cpp
    src/core/optimizers/pbil.cpp
    src/core/optimizers/umda.cpp
    src/core/optimizers/acor.cpp
    src/core/optimizers/arq3.cpp
    src/core/optimizers/clpso.cpp
    src/core/optimizers/cmaes.cpp
    src/core/optimizers/de.cpp
    src/core/optimizers/jso.cpp
    src/core/optimizers/lmcmaes.cpp
    src/core/optimizers/mlshaderl.cpp
    src/core/optimizers/pso.cpp

    src/core/validation/CrossValidator.cpp

    src/core/batch/BatchRunner.cpp

    src/core/export/CppExporter.cpp
    src/core/export/OnnxExporter.cpp
    src/core/export/XlsxWriter.cpp

    src/core/stats/Statistics.cpp

    src/ui/panels/BatchPlotWidget.cpp
    src/ui/panels/WilcoxonTab.cpp
    src/ui/panels/FriedmanTab.cpp

    src/ui/panels/DataPanel.cpp
    src/ui/panels/NetworkPanel.cpp
    src/ui/panels/TrainingPanel.cpp
    src/ui/panels/EvaluationPanel.cpp
    src/ui/panels/PredictionPanel.cpp
    src/ui/panels/BatchPanel.cpp
    src/ui/panels/PreTrainNNCPanel.cpp      # v0.9.0
    src/ui/panels/EvolBoundsPanel.cpp       # v0.9.0
    src/ui/panels/ArchSearchPanel.cpp       # v1.0.0 Mode 5
    src/ui/panels/GEStudioPanel.cpp         # Mode 6
    src/ui/panels/GEExperimentDialog.cpp    # Mode 6 — experiment editor
    src/ui/panels/GEModelPicker.cpp         # Mode 6 — model lists

    src/core/pretrain/PreTrainWorker.cpp    # v0.9.0
    src/core/bounds/EvolBoundsWorker.cpp    # v0.9.0
    src/core/arch/ArchSearchWorker.cpp      # v1.0.0 Mode 5

    src/core/ge/Grammar.cpp                 # Mode 6 — GE Studio
    src/core/ge/GEEngine.cpp
    src/core/ge/NetSpec.cpp
    src/core/ge/GEStudioWorker.cpp
    src/core/ge/GEBatch.cpp
    src/core/ge/GEReport.cpp
    src/core/ge/GEExperiment.cpp
    src/core/ge/Baselines.cpp
)

# Windows application icon
if(WIN32)
    list(APPEND SOURCES resources/app.rc)
endif()

add_executable(neuralstudio ${SOURCES})
target_include_directories(neuralstudio PRIVATE src src/core/optimizers src/core/stats src/core/pretrain src/core/bounds src/core/arch)
target_link_libraries(neuralstudio PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Charts Qt6::Svg)
if(OpenMP_CXX_FOUND)
    target_link_libraries(neuralstudio PRIVATE OpenMP::OpenMP_CXX)
endif()

if(Eigen3_FOUND)
    target_link_libraries(neuralstudio PRIVATE Eigen3::Eigen)
    target_compile_definitions(neuralstudio PRIVATE NS_HAVE_EIGEN)
endif()

if(MSVC)
    set_target_properties(neuralstudio PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
if(WIN32)
    set_target_properties(neuralstudio PROPERTIES WIN32_EXECUTABLE ON)
endif()
if(APPLE)
    set_target_properties(neuralstudio PROPERTIES
        MACOSX_BUNDLE ON
        MACOSX_BUNDLE_GUI_IDENTIFIER com.neuralstudio.app
        MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION})
endif()

install(TARGETS neuralstudio
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ── Console program: every mode without the GUI (Qt Core only) ───────────────
#   neuralstudio-cli train | batch | pretrain | bounds | archsearch | ge | predict | list
# It is built from the same core sources as the application, so a command line
# computes exactly what the corresponding panel computes. See docs/CLI.md.
option(NS_BUILD_CLI "Build the console program neuralstudio-cli" ON)
if(NS_BUILD_CLI)
    set(NS_CORE_SOURCES ${SOURCES})
    list(FILTER NS_CORE_SOURCES INCLUDE REGEX "^src/core/")
    add_executable(neuralstudio-cli
        src/cli/CliMain.cpp
        src/cli/CliArgs.cpp
        src/cli/CliCommon.cpp
        src/cli/CliTrain.cpp
        src/cli/CliBatch.cpp
        src/cli/CliPhases.cpp
        src/cli/CliArch.cpp
        src/cli/CliGE.cpp
        ${NS_CORE_SOURCES})
    target_include_directories(neuralstudio-cli PRIVATE src src/core/optimizers src/core/stats
                               src/core/pretrain src/core/bounds src/core/arch)
    target_compile_definitions(neuralstudio-cli PRIVATE NS_VERSION="${PROJECT_VERSION}")
    target_link_libraries(neuralstudio-cli PRIVATE Qt6::Core)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(neuralstudio-cli PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(Eigen3_FOUND)
        target_link_libraries(neuralstudio-cli PRIVATE Eigen3::Eigen)
        target_compile_definitions(neuralstudio-cli PRIVATE NS_HAVE_EIGEN)
    endif()
    if(MSVC)
        set_target_properties(neuralstudio-cli PROPERTIES
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
    install(TARGETS neuralstudio-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Headless unit test of the Mode 6 grammar / GE core (no Qt needed).
#   cmake -B build -DNS_BUILD_TESTS=ON && cmake --build build --target ge_core_test
option(NS_BUILD_TESTS "Build headless tests" OFF)
if(NS_BUILD_TESTS)
    add_executable(ge_core_test tests/ge_core_test.cpp
        src/core/ge/Grammar.cpp src/core/ge/GEEngine.cpp src/core/ge/NetSpec.cpp)
    target_include_directories(ge_core_test PRIVATE src)
endif()

if(NS_BUILD_TESTS)
    file(GLOB NS_TEST_CORE src/core/dataset/*.cpp src/core/dataset/loaders/*.cpp
         src/core/nn/*.cpp src/core/optimizers/*.cpp)
    add_executable(ge_worker_smoke tests/ge_worker_smoke.cpp ${NS_TEST_CORE}
        src/core/ge/Grammar.cpp src/core/ge/GEEngine.cpp src/core/ge/NetSpec.cpp
        src/core/ge/GEStudioWorker.cpp src/core/ge/Baselines.cpp)
    target_include_directories(ge_worker_smoke PRIVATE src src/core/optimizers)
    target_link_libraries(ge_worker_smoke PRIVATE Qt6::Core)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ge_worker_smoke PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if(NS_BUILD_TESTS)
    set(NS_APP_SRC ${SOURCES})
    list(REMOVE_ITEM NS_APP_SRC src/main.cpp)
    add_executable(ge_panel_smoke tests/ge_panel_smoke.cpp ${NS_APP_SRC})
    add_executable(ge_batch_gui_smoke tests/ge_batch_gui_smoke.cpp ${NS_APP_SRC})
    add_executable(ge_exp_open_smoke tests/ge_exp_open_smoke.cpp ${NS_APP_SRC})
    add_executable(grad_check tests/grad_check.cpp ${NS_APP_SRC})
    add_executable(modes_check tests/modes_check.cpp ${NS_APP_SRC})
    target_include_directories(modes_check PRIVATE src src/core/optimizers src/core/stats src/core/pretrain src/core/bounds src/core/arch)
    target_link_libraries(modes_check PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Charts Qt6::Svg)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(modes_check PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_include_directories(grad_check PRIVATE src src/core/optimizers src/core/stats src/core/pretrain src/core/bounds src/core/arch)
    target_link_libraries(grad_check PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Charts Qt6::Svg)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(grad_check PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_include_directories(ge_exp_open_smoke PRIVATE src src/core/optimizers src/core/stats src/core/pretrain src/core/bounds src/core/arch)
    target_link_libraries(ge_exp_open_smoke PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Charts Qt6::Svg)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ge_exp_open_smoke PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_include_directories(ge_batch_gui_smoke PRIVATE src src/core/optimizers src/core/stats src/core/pretrain src/core/bounds src/core/arch)
    target_link_libraries(ge_batch_gui_smoke PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Charts Qt6::Svg)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ge_batch_gui_smoke PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_include_directories(ge_panel_smoke PRIVATE src src/core/optimizers src/core/stats src/core/pretrain src/core/bounds src/core/arch)
    target_link_libraries(ge_panel_smoke PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Charts Qt6::Svg)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ge_panel_smoke PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Batch runner: all folds / seeds / configs, pooled statistics (Qt Core only).
#   cmake -B build -DNS_BUILD_TESTS=ON && cmake --build build --target ge_batch
if(NS_BUILD_TESTS)
    file(GLOB NS_BATCH_CORE src/core/dataset/*.cpp src/core/dataset/loaders/*.cpp
         src/core/nn/*.cpp src/core/optimizers/*.cpp)
    add_executable(ge_batch tools/ge_batch.cpp ${NS_BATCH_CORE}
        src/core/ge/Grammar.cpp src/core/ge/GEEngine.cpp src/core/ge/NetSpec.cpp
        src/core/ge/GEStudioWorker.cpp src/core/ge/GEBatch.cpp src/core/ge/GEReport.cpp src/core/ge/GEExperiment.cpp src/core/ge/Baselines.cpp src/core/stats/Statistics.cpp
        src/core/export/XlsxWriter.cpp)
    target_include_directories(ge_batch PRIVATE src src/core/optimizers)
    target_link_libraries(ge_batch PRIVATE Qt6::Core)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ge_batch PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if(NS_BUILD_TESTS)
    add_executable(ge_experiment_test tests/ge_experiment_test.cpp ${NS_BATCH_CORE}
        src/core/ge/Grammar.cpp src/core/ge/GEEngine.cpp src/core/ge/NetSpec.cpp
        src/core/ge/GEStudioWorker.cpp src/core/ge/GEBatch.cpp src/core/ge/GEReport.cpp src/core/ge/GEExperiment.cpp src/core/ge/Baselines.cpp
        src/core/stats/Statistics.cpp src/core/export/XlsxWriter.cpp)
    target_include_directories(ge_experiment_test PRIVATE src src/core/optimizers)
    target_link_libraries(ge_experiment_test PRIVATE Qt6::Core)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ge_experiment_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if(NS_BUILD_TESTS)
    add_executable(baselines_check tests/baselines_check.cpp ${NS_BATCH_CORE} src/core/ge/Baselines.cpp)
    target_include_directories(baselines_check PRIVATE src src/core/optimizers)
    target_link_libraries(baselines_check PRIVATE Qt6::Core)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(baselines_check PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

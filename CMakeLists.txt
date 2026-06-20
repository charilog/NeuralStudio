cmake_minimum_required(VERSION 3.20)
project(NeuralStudio VERSION 0.8.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Charts Svg)
find_package(Eigen3 QUIET)
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
)

# Windows application icon
if(WIN32)
    list(APPEND SOURCES resources/app.rc)
endif()

add_executable(neuralstudio ${SOURCES})
target_include_directories(neuralstudio PRIVATE src src/core/optimizers src/core/stats)
target_link_libraries(neuralstudio PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Charts Qt6::Svg)

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

#include "PreTrainWorker.h"
#include "core/dataset/DatasetLoader.h"
#include "core/optimizers/NSProblem.h"
#include "core/optimizers/optimizer.h"
#include "core/optimizers/genetic.h"
#include "core/optimizers/de.h"
#include "core/optimizers/pso.h"
#include "core/optimizers/cmaes.h"
#include "core/optimizers/jso.h"
#include "core/optimizers/lmcmaes.h"
#include "core/optimizers/acor.h"
#include "core/optimizers/clpso.h"
#include "core/optimizers/mlshaderl.h"
#include "core/optimizers/arq3.h"
#include "core/optimizers/lbfgs.h"
#include "core/optimizers/simann.h"
#include "core/optimizers/neldermead.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <functional>
#include <QFileInfo>

#include <chrono>
#include <atomic>

// Cross-platform seed: combines timestamp + atomic counter
// (fixes MinGW std::random_device determinism bug on Windows)
static unsigned int makeSeed() {
    static std::atomic<uint64_t> ctr{0};
    const auto t = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return static_cast<unsigned int>(t ^ (++ctr * 6364136223846793005ULL));
}


namespace NeuralStudio {

PreTrainWorker::PreTrainWorker(QObject* parent) : QObject(parent) {}

// ── helpers ───────────────────────────────────────────────────────────────────
static std::unique_ptr<optimsolution::Optimizer> makeOpt(const QString& n) {
    if(n=="lbfgs"){auto p=std::make_unique<optimsolution::LBFGS>();p->setMemory(7); return p;}
    if(n=="bfgs") {auto p=std::make_unique<optimsolution::LBFGS>();p->setMemory(20);return p;}
    if(n=="sa")         return std::make_unique<optimsolution::SimulatedAnnealing>();
    if(n=="neldermead") return std::make_unique<optimsolution::NelderMead>();
    if(n=="de")         return std::make_unique<optimsolution::DE>();
    if(n=="pso")        return std::make_unique<optimsolution::PSO>();
    if(n=="cmaes")      return std::make_unique<optimsolution::CMAES>();
    if(n=="jso")        return std::make_unique<optimsolution::JSO>();
    if(n=="lmcmaes")    return std::make_unique<optimsolution::LMCMAES>();
    if(n=="acor")       return std::make_unique<optimsolution::ACOR>();
    if(n=="clpso")      return std::make_unique<optimsolution::CLPSO>();
    if(n=="mlshaderl")  return std::make_unique<optimsolution::mLSHADE_RL>();
    if(n=="arq3")       return std::make_unique<optimsolution::ARQ3>();
    return std::make_unique<optimsolution::GeneticAlgorithm>();
}
static std::vector<int> parseHidden(const QString& s) {
    std::vector<int> o;
    for(const QString& t:s.split(',',Qt::SkipEmptyParts)){bool ok;int n=t.trimmed().toInt(&ok);if(ok&&n>0)o.push_back(n);}
    return o.empty()?std::vector<int>{16}:o;
}
static Activation parseAct(const QString& s){
    if(s=="sigmoid")return Activation::Sigmoid;if(s=="tanh")return Activation::Tanh;
    if(s=="linear")return Activation::Linear;return Activation::ReLU;
}
static std::shared_ptr<NeuralNetwork> buildNet(const Dataset* ds,
    const std::vector<int>& h,Activation act){
    NetworkConfig cfg;cfg.task=detectTask(ds);cfg.hiddenSizes=h;
    cfg.hiddenActivations=std::vector<Activation>(h.size(),act);
    auto uv=ds->uniqueOutputValues(0);
    // MultiClassClassification needs one output neuron per class, not per output column
    // Output size: multiclass = one node per class; binary = single node
    // (same rule as BatchRunner); regression = dataset output columns.
    const int netOut=(cfg.task==TaskType::MultiClassClassification)
                     ? static_cast<int>(uv.size())
                     : (cfg.task==TaskType::BinaryClassification)
                     ? 1 : ds->outputCount;
    auto net=std::make_shared<NeuralNetwork>();net->build(ds->inputCount,netOut,cfg);
    net->setNormalization(ds->inputStats);
    net->fitOutputScaling(ds);   // regression: standardised output layer
    if(cfg.task!=TaskType::Regression){net->setClassValues(uv);if(!uv.empty())net->setOutputMapping(uv.front(),uv.back());}
    else if(!ds->outputStats.empty()) net->setOutputMapping(ds->outputStats[0].min,ds->outputStats[0].max);
    return net;
}
// Transfer the pre-trained small network (one hidden layer of Iw nodes) into the main
// network neuron by neuron. The old version copied the first entries of the FLAT weight
// vector, so after the first-layer weights the small net's biases and output weights
// landed on unrelated parameters of the main network.
//   * first hidden layer: neurons 0..Iw-1 receive the small net's input weights and biases,
//     the other neurons small random weights (U(-0.1, 0.1));
//   * one hidden layer in the main net: the output weights of the copied neurons and the
//     output biases are copied and the extra neurons get output weight 0, so the main
//     network starts as exactly the pre-trained function;
//   * more hidden layers: the deeper layers keep their Glorot initialisation.
static void transferWeights(NeuralNetwork& small, NeuralNetwork& main){
    auto& S=small.layers(); auto& M=main.layers();
    if(S.size()!=2 || M.size()<2) return;
    std::mt19937 rng(makeSeed());
    std::uniform_real_distribution<double> dist(-0.1,0.1);
    const int Iw=S[0].outputSize(), H=M[0].outputSize(), nIn=M[0].inputSize();
    for(int j=0;j<H;++j){
        for(int i=0;i<nIn;++i) M[0].weights[j][i] = (j<Iw && i<S[0].inputSize()) ? S[0].weights[j][i] : dist(rng);
        M[0].biases[j] = j<Iw ? S[0].biases[j] : dist(rng);
    }
    if(M.size()==2){
        const int nOut=M[1].outputSize();
        for(int o=0;o<nOut && o<S[1].outputSize();++o){
            for(int j=0;j<H;++j) M[1].weights[o][j] = j<Iw ? S[1].weights[o][j] : 0.0;
            M[1].biases[o] = S[1].biases[o];
        }
    }
}

// ── runLoop: generic optimizer loop ──────────────────────────────────────────
static std::vector<double> runLoop(optimsolution::Optimizer* opt,
    NSProblem& prob,NeuralNetwork& net,int iters,std::atomic<bool>& stop,
    const std::function<void(int,int,double,double)>& cb)
{
    opt->init();
    for(int it=1;it<=iters&&!stop.load()&&!opt->done();++it){
        opt->one_iteration();opt->advanceIter();
        if(!opt->bestX().empty())NSProblem::unflatten(net,opt->bestX());
        cb(it,iters,opt->bestF(),prob.valLoss());
    }
    opt->end();
    if(!opt->bestX().empty())NSProblem::unflatten(net,opt->bestX());
    return opt->bestX();
}

// ── measureNet ────────────────────────────────────────────────────────────────
// Uses NSProblem's built-in shuffled split (seed=42) — same as during training.
PreTrainRunResult PreTrainWorker::measureNet(NeuralNetwork* net,Dataset* ds) const {
    const bool isCls=(net->taskType()!=TaskType::Regression);
    PreTrainRunResult r;

    // Evaluate on the SAME shuffled partition used during training
    NSProblem prob(net,ds,10.0,m_cfg.valSplit);
    r.trainLoss = prob.evaluate(NSProblem::flatten(*net));

    if(isCls){
        r.trainErr = (1.0 - prob.trainAcc()) * 100.0;   // error %
        const double va = (m_cfg.valSplit>0) ? prob.valAcc() : -1.0;   // -1: no validation rows
        r.valErr   = va>=0 ? (1.0 - va) * 100.0 : -1.0;
    } else {
        r.trainErr = r.trainLoss;                        // MSE
        r.valErr   = (m_cfg.valSplit>0) ? prob.valLoss() : -1.0;
    }
    if(m_testDs && m_testDs->sampleCount>0 && m_testDs->inputCount==ds->inputCount){
        NSProblem tp(net,m_testDs.get(),10.0,0.0);       // all test rows
        const double tl = tp.evaluate(NSProblem::flatten(*net));
        r.testErr = isCls ? (1.0 - tp.trainAcc()) * 100.0 : tl;
    }
    return r;
}


// ── runBaseline: standard training without pre-training ───────────────────────
PreTrainRunResult PreTrainWorker::runBaseline(NeuralNetwork* net,Dataset* ds,
                                               int phNum,const QString& dsName){
    PhaseConfig pc=m_cfg.phase2;
    if(m_cfg.equalBudgetBaseline){
        // same total evaluations as the pre-training route (Phase 1 + 2 [+ 3]), with the Phase-2
        // optimizer and population (it used to get the Phase-2 budget only)
        qint64 total=static_cast<qint64>(m_cfg.phase1.population)*m_cfg.phase1.iterations
                    +static_cast<qint64>(m_cfg.phase2.population)*m_cfg.phase2.iterations;
        if(m_cfg.phase3.optimizer.toLower()!="none"&&!m_cfg.phase3.optimizer.isEmpty())
            total+=static_cast<qint64>(m_cfg.phase3.population)*m_cfg.phase3.iterations;
        const qint64 pop=std::max(1,pc.population);
        pc.iterations=static_cast<int>(std::min<qint64>(1000000,(total+pop-1)/pop));
    }
    emit phaseStarted(phNum,pc.iterations,
        QString("[%1]  Baseline  —  %2  (no pre-training%3)").arg(dsName,pc.optimizer.toUpper())
            .arg(m_cfg.equalBudgetBaseline?QString(", equal total budget"):QString()));
        NSProblem prob(net,ds,pc.wBound,m_cfg.valSplit);
    auto opt=makeOpt(pc.optimizer);
    {optimsolution::MethodConfig mc;mc.data["population"]=std::to_string(pc.population);opt->configure(mc);}
    opt->setup(&prob,pc.iterations*pc.population,pc.population,makeSeed());
    runLoop(opt.get(),prob,*net,pc.iterations,m_stop,
        [&](int it,int tot,double l,double vl){emit iterationDone(phNum,it,tot,l,vl);});
    emit phaseFinished(phNum,opt->bestF());
    auto r=measureNet(net,ds); r.trainLoss=opt->bestF(); return r;
}

// ── runOnDataset ──────────────────────────────────────────────────────────────
bool PreTrainWorker::runOnDataset(int dsIdx,const QString& dsPath){
    const QString dsName=QFileInfo(dsPath).baseName();
    emit datasetStarted(dsIdx,(int)m_cfg.datasetPaths.size(),dsName);

    // Train on the TRAIN file only (loadCombined merged the test file into the training /
    // validation data); the companion test file is only scored.
    DatasetBundle bundle=DatasetLoader::loadBundle(dsPath);
    auto ds=bundle.train;
    if(!ds) throw std::runtime_error("Cannot load "+dsPath.toStdString());
    ds->computeStatistics();
    m_testDs=bundle.test;
    if(m_testDs) m_testDs->computeStatistics();
    const bool isCls=(detectTask(ds.get())!=TaskType::Regression);
    const auto hidden=parseHidden(m_cfg.hiddenLayout);
    const auto act=parseAct(m_cfg.activation);

    QStringList   configs;
    QVector<double> trainLoss,trainErr,valErr,testErr;
    int ph=1;

    // ── 0. Baseline (no pre-training) ────────────────────────────────────
    auto netBase=buildNet(ds.get(),hidden,act); netBase->setTrainingMode(false);
    auto base=runBaseline(netBase.get(),ds.get(),ph++,dsName);
    if(m_stop.load())return false;
    // Save Standard's best weights — every subsequent phase must match or beat it
    const auto standardWeights = NSProblem::flatten(*netBase);
    configs<<(m_cfg.equalBudgetBaseline?"Standard (no pre-train, equal budget)":"Standard (no pre-train)");
    trainLoss<<base.trainLoss; trainErr<<base.trainErr; valErr<<base.valErr; testErr<<base.testErr;

    // ── 1. Phase 1: pre-train small network (Iw nodes) ───────────────────
    {
        const auto& pc=m_cfg.phase1;
        emit phaseStarted(ph,pc.iterations,
            QString("[%1]  Phase %2  —  %3  (Iw=%4 nodes)")
                .arg(dsName).arg(ph).arg(pc.optimizer.toUpper()).arg(m_cfg.Iw));
                auto net1=buildNet(ds.get(),{m_cfg.Iw},act); // same activation as main network
        net1->setTrainingMode(false);
        NSProblem prob(net1.get(),ds.get(),pc.wBound,m_cfg.valSplit);
        auto opt=makeOpt(pc.optimizer);
        {optimsolution::MethodConfig mc;mc.data["population"]=std::to_string(pc.population);opt->configure(mc);}
        opt->setup(&prob,pc.iterations*pc.population,pc.population,makeSeed());
        auto w1=runLoop(opt.get(),prob,*net1,pc.iterations,m_stop,
            [&](int it,int tot,double l,double vl){emit iterationDone(ph,it,tot,l,vl);});
        emit phaseFinished(ph,opt->bestF());
        if(m_stop.load())return false;
        // Seed main network with adapted Phase-1 weights
        auto netMain=buildNet(ds.get(),hidden,act); netMain->setTrainingMode(false);
        (void)w1;   // net1 holds the best Phase-1 weights (runLoop applied them)
        transferWeights(*net1,*netMain);
        ++ph;

        // ── 2. Phase 2: fine-tune main network ────────────────────────────
        {
            const auto& pc2=m_cfg.phase2;
            emit phaseStarted(ph,pc2.iterations,
                QString("[%1]  Phase %2  —  %3  (fine-tune)")
                    .arg(dsName).arg(ph).arg(pc2.optimizer.toUpper()));

            // Save Phase 1 adapted state to restore if Phase 2 regresses
            const auto phase1Weights = NSProblem::flatten(*netMain);
            auto rInit = measureNet(netMain.get(), ds.get());

            // Focused bound based on Phase 1 weight magnitudes (same strategy as Phase 3)
            double maxW1 = 0;
            for (double w : phase1Weights) maxW1 = std::max(maxW1, std::abs(w));
            const double ph2Bound = std::min(std::max(maxW1 * 2.0, 1.0), pc2.wBound);

            NSProblem prob2(netMain.get(),ds.get(),ph2Bound,m_cfg.valSplit);
            auto opt2=makeOpt(pc2.optimizer);
            {optimsolution::MethodConfig mc;mc.data["population"]=std::to_string(pc2.population);opt2->configure(mc);}
            opt2->setup(&prob2,pc2.iterations*pc2.population,pc2.population,makeSeed());
            opt2->setInitialPoint(phase1Weights);  // warm-start from Phase 1 adapted
            runLoop(opt2.get(),prob2,*netMain,pc2.iterations,m_stop,
                [&](int it,int tot,double l,double vl){emit iterationDone(ph,it,tot,l,vl);});
            emit phaseFinished(ph,opt2->bestF());
            ++ph;
            // Use measureNet (same NSProblem/split as Standard) for a fair comparison
            auto r = measureNet(netMain.get(), ds.get());

            // Phase 1+2 must NEVER be worse than Standard on the PRIMARY metric:
            // Error% for classification, Loss (MSE) for regression.
            // NOTE: Loss and Error% are NOT monotonically related for classification —
            // a lower BCE/CCE does not guarantee lower classification error.
            const double ph2Primary = isCls ? r.trainErr   : r.trainLoss;
            const double stdPrimary = isCls ? base.trainErr : base.trainLoss;
            const bool ph2TrainWorse = (ph2Primary > stdPrimary);
            // Also protect validation: overfitting in Phase 2 is rejected
            const bool ph2ValWorse = (m_cfg.valSplit > 0 && base.valErr >= 0
                                      && r.valErr >= 0 && r.valErr > base.valErr);
            // The row reports what Phase 1+2 achieved (it was overwritten with the Standard
            // numbers when worse); the Standard weights are then carried forward instead.
            configs<<"Phase 1+2"; trainLoss<<r.trainLoss; trainErr<<r.trainErr; valErr<<r.valErr; testErr<<r.testErr;
            const bool fromStandard = ph2TrainWorse || ph2ValWorse;
            if(fromStandard){
                NSProblem::unflatten(*netMain, standardWeights);
                r = measureNet(netMain.get(), ds.get());   // reference for Phase 3 = kept weights
            }
            if(m_stop.load())return false;

            // ── 3. Phase 3: optional ──────────────────────────────────────
            if(m_cfg.phase3.optimizer.toLower()!="none"&&!m_cfg.phase3.optimizer.isEmpty()){
                const auto& pc3=m_cfg.phase3;
                emit phaseStarted(ph,pc3.iterations,
                    QString("[%1]  Phase %2  —  %3  (final fine-tune)")
                        .arg(dsName).arg(ph).arg(pc3.optimizer.toUpper()));

                // Save Phase 2 best weights and metrics — Phase 3 must match or beat them
                const auto phase2Weights = NSProblem::flatten(*netMain);
                const double phase2Loss  = r.trainLoss;
                const double phase2Err   = r.trainErr;
                const double phase2Val   = r.valErr;

                // Focused bound around Phase 2 solution (never wider than wBound)
                double maxW = 0;
                for (double w : phase2Weights) maxW = std::max(maxW, std::abs(w));
                const double ph3Bound = std::min(std::max(maxW * 2.0, 1.0), pc3.wBound);

                NSProblem prob3(netMain.get(),ds.get(),ph3Bound,m_cfg.valSplit);
                auto opt3=makeOpt(pc3.optimizer);
                {optimsolution::MethodConfig mc;mc.data["population"]=std::to_string(pc3.population);opt3->configure(mc);}
                opt3->setup(&prob3,pc3.iterations*pc3.population,pc3.population,makeSeed());
                opt3->setInitialPoint(phase2Weights);  // warm-start from Phase 2
                runLoop(opt3.get(),prob3,*netMain,pc3.iterations,m_stop,
                    [&](int it,int tot,double l,double vl){emit iterationDone(ph,it,tot,l,vl);});
                emit phaseFinished(ph,opt3->bestF());

                // Use measureNet (same scale as Phase 2) for fair comparison
                auto r3 = measureNet(netMain.get(), ds.get());

                // Phase 1+2+3 must NEVER be worse than Phase 1+2 on the primary metric
                const double ph3Primary = isCls ? r3.trainErr  : r3.trainLoss;
                const double ph2Primary = isCls ? phase2Err    : phase2Loss;
                const bool ph3TrainWorse = (ph3Primary > ph2Primary);
                // Also protect validation: overfitting in Phase 3 is rejected
                const bool ph3ValWorse = (m_cfg.valSplit > 0 && phase2Val >= 0
                                          && r3.valErr >= 0 && r3.valErr > phase2Val);
                // Phase 3 continued from the Standard weights when Phase 1+2 was worse: the row then
                // is "Standard + Phase 3", not a pre-trained network (it was labelled Phase 1+2+3)
                configs<<(fromStandard?"Standard + Phase 3 (Phase 1+2 rejected)":"Phase 1+2+3"); trainLoss<<r3.trainLoss; trainErr<<r3.trainErr; valErr<<r3.valErr; testErr<<r3.testErr;
                if(ph3TrainWorse || ph3ValWorse) NSProblem::unflatten(*netMain, phase2Weights);   // final network: never worse
            }
        }
        // Save last trained network (still in scope) for allDone signal
        m_lastNet = netMain;
        m_lastDs  = ds;
    }

    emit datasetFinished(dsName,configs,trainLoss,trainErr,valErr,testErr,isCls);
    return true;
}

// ── run ───────────────────────────────────────────────────────────────────────
void PreTrainWorker::run(){
    try{
        if(m_cfg.datasetPaths.isEmpty())throw std::runtime_error("No datasets queued.");
        m_stop.store(false);
        for(int i=0;i<m_cfg.datasetPaths.size()&&!m_stop.load();++i)
            runOnDataset(i,m_cfg.datasetPaths[i]);
        emit allDone(m_lastNet, m_lastDs);
    }catch(const std::exception& ex){
        emit errorOccurred(QString::fromStdString(ex.what()));
    }
}

} // namespace NeuralStudio

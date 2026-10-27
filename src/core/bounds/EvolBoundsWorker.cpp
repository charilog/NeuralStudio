#include "EvolBoundsWorker.h"
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
#include <cmath>
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

EvolBoundsWorker::EvolBoundsWorker(QObject* parent) : QObject(parent) {}

// ── factory / parse helpers ───────────────────────────────────────────────────
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
static std::vector<int> parseH(const QString& s){
    std::vector<int> o;
    for(const QString& t:s.split(',',Qt::SkipEmptyParts)){bool ok;int n=t.trimmed().toInt(&ok);if(ok&&n>0)o.push_back(n);}
    return o.empty()?std::vector<int>{16}:o;
}
static Activation parseA(const QString& s){
    if(s=="sigmoid")return Activation::Sigmoid;if(s=="tanh")return Activation::Tanh;
    if(s=="linear")return Activation::Linear;return Activation::ReLU;
}
static std::shared_ptr<NeuralNetwork> buildNet(const Dataset* ds,const std::vector<int>& h,Activation act){
    NetworkConfig cfg;cfg.task=detectTask(ds);cfg.hiddenSizes=h;cfg.hiddenActivations=std::vector<Activation>(h.size(),act);
    auto uv=ds->uniqueOutputValues(0);
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

// ── problem wrappers ──────────────────────────────────────────────────────────
class PenalizedNSProblem:public NSProblem{
public:
    PenalizedNSProblem(NeuralNetwork*n,const Dataset*d,double wb,double vs,double lam,double a)
        :NSProblem(n,d,wb,vs),m_lam(lam),m_a(a),m_net(n){}
    double evaluate(const std::vector<double>&x)override{
        const double loss=NSProblem::evaluate(x);
        // Count individual weights exceeding threshold a
        // (per-weight check, not per-neuron wNorm — avoids input-size sensitivity)
        double sat=0,tot=0;
        for(int li=0;li<(int)m_net->layers().size()-1;++li){
            const auto&lay=m_net->layers()[li];
            for(int ni=0;ni<lay.outputSize();++ni)
                for(int wi=0;wi<lay.inputSize();++wi){
                    if(std::abs(lay.weights[ni][wi])>m_a) sat++;
                    tot++;
                }
        }
        const double B=(tot>0)?sat/tot:0.0;
        return loss*(1.0+m_lam*B*B);
    }
private:double m_lam,m_a;NeuralNetwork*m_net;
};

class BoundedProblem:public optimsolution::Problem{
public:
    BoundedProblem(NeuralNetwork*n,const Dataset*d,const std::vector<double>&lb,const std::vector<double>&ub,double vs)
        :m_inner(n,d,1.0,vs),m_lb(lb),m_ub(ub){}
    int dimension()const override{return(int)m_lb.size();}
    const std::vector<double>&lb()const override{return m_lb;}
    const std::vector<double>&ub()const override{return m_ub;}
    double evaluate(const std::vector<double>&x)override{return m_inner.evaluate(x);}
    int calls()const override{return m_inner.calls();}
    double valLoss()const{return m_inner.valLoss();}
    std::pair<double,std::vector<double>>evalAndGrad(const std::vector<double>&x)override{return m_inner.evalAndGrad(x);}
private:NSProblem m_inner;std::vector<double>m_lb,m_ub;
};

// ── generic loop helpers ──────────────────────────────────────────────────────
static std::vector<double> loopNS(optimsolution::Optimizer*opt,NSProblem&prob,
    NeuralNetwork&net,int iters,std::atomic<bool>&stop,
    const std::function<void(int,int,double,double)>&cb){
    opt->init();
    for(int it=1;it<=iters&&!stop.load()&&!opt->done();++it){opt->one_iteration();opt->advanceIter();
        if(!opt->bestX().empty())NSProblem::unflatten(net,opt->bestX());cb(it,iters,opt->bestF(),prob.valLoss());}
    opt->end();if(!opt->bestX().empty())NSProblem::unflatten(net,opt->bestX());return opt->bestX();
}
static std::vector<double> loopBnd(optimsolution::Optimizer*opt,BoundedProblem&prob,
    NeuralNetwork&net,int iters,std::atomic<bool>&stop,
    const std::function<void(int,int,double,double)>&cb){
    opt->init();
    for(int it=1;it<=iters&&!stop.load()&&!opt->done();++it){opt->one_iteration();opt->advanceIter();
        if(!opt->bestX().empty())NSProblem::unflatten(net,opt->bestX());cb(it,iters,opt->bestF(),prob.valLoss());}
    opt->end();if(!opt->bestX().empty())NSProblem::unflatten(net,opt->bestX());return opt->bestX();
}

// ── measureNet ────────────────────────────────────────────────────────────────
// Uses NSProblem's built-in shuffled split (seed=42) — same as during training.
// This ensures trainErr/valErr are consistent with the optimizer's loss.
EvolBoundsWorker::RunResult
EvolBoundsWorker::measureNet(NeuralNetwork*net,Dataset*ds)const{
    const bool isCls=(net->taskType()!=TaskType::Regression);
    RunResult r;

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


// ── Baseline ──────────────────────────────────────────────────────────────────
EvolBoundsWorker::RunResult
EvolBoundsWorker::runBaseline(NeuralNetwork*net,Dataset*ds,int phNum,const QString&dsName){
    const QString optN=(m_cfg.opt2.toLower()=="none"||m_cfg.opt2.toLower()=="ge_grammar")?m_cfg.opt1:m_cfg.opt2;
    int iters=m_cfg.Ng2;
    if(m_cfg.equalBudgetBaseline){
        // evaluations of the bounded route: Phase 1, Phase 2 (optimizer, or the GE scoring of bound
        // programs: Ns random networks per candidate, local search with probability pl over Nl
        // neighbours), Phase 3 and the final BFGS polish
        const QString o2=m_cfg.opt2.toLower(), o3=m_cfg.opt3.toLower();
        double total=static_cast<double>(m_cfg.Nc1)*m_cfg.Ng1;
        if(o2=="ge_grammar")
            total+=m_cfg.Ns_ge*(1.0+static_cast<double>(m_cfg.Ng2)*m_cfg.Nc2*(1.0+m_cfg.pl_ge*(1.0+m_cfg.Nl_ge)));
        else if(o2!="none"&&!o2.isEmpty())
            total+=static_cast<double>(m_cfg.Nc2)*m_cfg.Ng2;
        if(o3!="none"&&!o3.isEmpty())
            total+=static_cast<double>(m_cfg.Nc3)*m_cfg.Ng3+(m_cfg.bfgsEnd?2000.0:0.0);
        iters=static_cast<int>(std::min(1.0e6,std::ceil(total/std::max(1,m_cfg.Nc2))));
    }
    emit phaseStarted(phNum,iters,
        QString("[%1]  Baseline  —  %2  (±%3%4)").arg(dsName,optN.toUpper()).arg(m_cfg.initialBound)
            .arg(m_cfg.equalBudgetBaseline?QString(", equal total budget"):QString()));
        NSProblem prob(net,ds,m_cfg.initialBound,m_cfg.valSplit);
    auto opt=makeOpt(optN);
    {optimsolution::MethodConfig mc;mc.data["population"]=std::to_string(m_cfg.Nc2);opt->configure(mc);}
    opt->setup(&prob,iters*m_cfg.Nc2,m_cfg.Nc2,makeSeed());
    loopNS(opt.get(),prob,*net,iters,m_stop,[&](int it,int tot,double l,double vl){emit iterationDone(phNum,it,tot,l,vl);});
    emit phaseFinished(phNum,opt->bestF());
    auto r=measureNet(net,ds); r.trainLoss=opt->bestF(); return r;
}

// ── Phase 1 ───────────────────────────────────────────────────────────────────
EvolBoundsWorker::Bounds
EvolBoundsWorker::runPhase1(NeuralNetwork*net,Dataset*ds,int phNum,const QString&dsName){
    emit phaseStarted(phNum,m_cfg.Ng1,
        QString("[%1]  Phase %2  —  %3  (penalised, λ=%4)")
            .arg(dsName).arg(phNum).arg(m_cfg.opt1.toUpper()).arg(m_cfg.lambda));
        PenalizedNSProblem prob(net,ds,m_cfg.initialBound,m_cfg.valSplit,m_cfg.lambda,m_cfg.a);
    auto opt=makeOpt(m_cfg.opt1);
    {optimsolution::MethodConfig mc;mc.data["population"]=std::to_string(m_cfg.Nc1);opt->configure(mc);}
    opt->setup(&prob,m_cfg.Ng1*m_cfg.Nc1,m_cfg.Nc1,makeSeed());
    loopNS(opt.get(),prob,*net,m_cfg.Ng1,m_stop,[&](int it,int tot,double l,double vl){emit iterationDone(phNum,it,tot,l,vl);});
    emit phaseFinished(phNum,opt->bestF());
    const auto&g=opt->bestX();const int dim=net->totalParams();
    Bounds b;b.lb.resize(dim);b.ub.resize(dim);
    QVector<double>qlb(dim),qub(dim);
    for(int i=0;i<dim;++i){const double mag=m_cfg.f*std::abs(g.empty()?1.0:g[i]);
        // Bounds in [minHalf, initialBound]: never wider than standard search space,
        // never so tight they collapse to zero
        const double minHalf = m_cfg.initialBound * 0.1;
        const double halfWidth = std::min(std::max(mag,minHalf), m_cfg.initialBound);
        b.lb[i]=-halfWidth;b.ub[i]=+halfWidth;qlb[i]=b.lb[i];qub[i]=b.ub[i];}
    emit datasetBoundsReady(dsName,qlb,qub,m_cfg.initialBound);
    return b;
}

// ── Phase 2 (optimizer) ───────────────────────────────────────────────────────
EvolBoundsWorker::Bounds
EvolBoundsWorker::runPhase2Opt(NeuralNetwork*net,Dataset*ds,const Bounds&b,int phNum){
    emit phaseStarted(phNum,m_cfg.Ng2,
        QString("Phase %1  —  %2  (training within bounds)").arg(phNum).arg(m_cfg.opt2.toUpper()));
        BoundedProblem prob(net,ds,b.lb,b.ub,m_cfg.valSplit);
    auto opt=makeOpt(m_cfg.opt2);
    {optimsolution::MethodConfig mc;mc.data["population"]=std::to_string(m_cfg.Nc2);opt->configure(mc);}
    opt->setup(&prob,m_cfg.Ng2*m_cfg.Nc2,m_cfg.Nc2,makeSeed());
    // Warm-start Phase 2 with Phase 1's best weights (net already holds them after runPhase1)
    opt->setInitialPoint(NSProblem::flatten(*net));
    auto bw=loopBnd(opt.get(),prob,*net,m_cfg.Ng2,m_stop,[&](int it,int tot,double l,double vl){emit iterationDone(phNum,it,tot,l,vl);});
    emit phaseFinished(phNum,opt->bestF());
    Bounds refined=b;
    // refined half-width f*0.6*|w|, but never narrower than |w| itself: with f < 1/0.6 the refined
    // box excluded the Phase-2 solution, and Phase 3 started from a clipped (worse) network
    if(!bw.empty())for(int i=0;i<(int)bw.size();++i){const double t=std::max({std::abs(bw[i])*m_cfg.f*0.6,std::abs(bw[i]),0.01});refined.lb[i]=std::max(b.lb[i],-t);refined.ub[i]=std::min(b.ub[i],+t);}
    return refined;
}

// ── Phase 2 (GE grammar) ──────────────────────────────────────────────────────
EvolBoundsWorker::Bounds
EvolBoundsWorker::runPhase2GE(NeuralNetwork*net,Dataset*ds,const Bounds&init,int phNum){
    const int dim=(int)init.lb.size();
    emit phaseStarted(phNum,m_cfg.Ng2,QString("Phase %1  —  GE grammar").arg(phNum));
    static constexpr double kCmd[]={1.0,0.5,1.5,0.75,0.75,0.25,2.25};static constexpr int kN=7;
    std::mt19937 rng(makeSeed());
    std::uniform_int_distribution<int>cmdD(0,kN-1);std::uniform_real_distribution<double>unitD(0,1);std::uniform_int_distribution<int>dimD(0,dim-1);
    // scored on the TRAINING part of the same split as the other phases (seed 42); with 0.0 the
    // validation rows took part in choosing the bounds that are later judged on them
    NSProblem evalProb(net,ds,1.0,m_cfg.valSplit);
    auto score=[&](const Bounds&bnd){double tot=0;for(int s=0;s<m_cfg.Ns_ge;++s){std::vector<double>w(dim);for(int i=0;i<dim;++i)w[i]=bnd.lb[i]+unitD(rng)*(bnd.ub[i]-bnd.lb[i]);tot+=evalProb.evaluate(w);}return tot/m_cfg.Ns_ge;};
    Bounds best=init;double bestS=score(best);
    for(int gen=1;gen<=m_cfg.Ng2&&!m_stop.load();++gen){
        for(int p=0;p<m_cfg.Nc2&&!m_stop.load();++p){Bounds cand;cand.lb.resize(dim);cand.ub.resize(dim);for(int i=0;i<dim;++i){const double m=std::max(std::abs(init.ub[i])*kCmd[cmdD(rng)],0.01);cand.lb[i]=-m;cand.ub[i]=+m;}
            if(unitD(rng)<m_cfg.pl_ge){double cs=score(cand);for(int ni=0;ni<m_cfg.Nl_ge;++ni){Bounds tr=cand;const int d=dimD(rng);const double m2=std::max(std::abs(init.ub[d])*kCmd[cmdD(rng)],0.01);tr.lb[d]=-m2;tr.ub[d]=+m2;const double ts=score(tr);if(ts<cs){cand=tr;cs=ts;}}}
            const double s=score(cand);if(s<bestS){bestS=s;best=cand;}}
        emit iterationDone(phNum,gen,m_cfg.Ng2,bestS,-1.0);}
    emit phaseFinished(phNum,bestS);return best;
}

// ── Phase 3 ───────────────────────────────────────────────────────────────────
void EvolBoundsWorker::runPhase3(NeuralNetwork*net,Dataset*ds,const Bounds&bounds,int phNum){
    emit phaseStarted(phNum,m_cfg.Ng3,
        QString("Phase %1  —  %2  (final training)").arg(phNum).arg(m_cfg.opt3.toUpper()));
        BoundedProblem prob(net,ds,bounds.lb,bounds.ub,m_cfg.valSplit);
    auto opt=makeOpt(m_cfg.opt3);
    {optimsolution::MethodConfig mc;mc.data["population"]=std::to_string(m_cfg.Nc3);opt->configure(mc);}
    opt->setup(&prob,m_cfg.Ng3*m_cfg.Nc3,m_cfg.Nc3,makeSeed());
    // Warm-start Phase 3 with Phase 2's best weights
    opt->setInitialPoint(NSProblem::flatten(*net));
    auto bw=loopBnd(opt.get(),prob,*net,m_cfg.Ng3,m_stop,[&](int it,int tot,double l,double vl){emit iterationDone(phNum,it,tot,l,vl);});
    if(m_cfg.bfgsEnd&&!m_stop.load()&&!bw.empty()){
        auto bfgs=std::make_unique<optimsolution::LBFGS>();bfgs->setMemory(20);
        // budget = 2000 evaluations from NOW (prob.calls() already counts the earlier phase;
        // an absolute 2000 meant the polish usually never ran)
        bfgs->setup(&prob,prob.calls()+2000,1,makeSeed());bfgs->setInitialPoint(bw);bfgs->init();
        while(!bfgs->done()&&!m_stop.load()){const int c0=prob.calls();bfgs->one_iteration();bfgs->advanceIter();if(prob.calls()==c0)break;}   // zero gradient: no progress possible
        bfgs->end();if(!bfgs->bestX().empty())NSProblem::unflatten(*net,bfgs->bestX());}
    emit phaseFinished(phNum,opt->bestF());
}

// ── runOnDataset ──────────────────────────────────────────────────────────────
bool EvolBoundsWorker::runOnDataset(int dsIdx,const QString&dsPath){
    const QString dsName=QFileInfo(dsPath).baseName();
    emit datasetStarted(dsIdx,(int)m_cfg.datasetPaths.size(),dsName);

    // Train on the TRAIN file only. (loadCombined merged the companion test file into the
    // data that is trained on and split for validation, so the test rows influenced the
    // bounds, the training and the "never worse" decisions.) The test file is only scored.
    DatasetBundle bundle=DatasetLoader::loadBundle(dsPath);
    auto ds=bundle.train;
    if(!ds) throw std::runtime_error("Cannot load "+dsPath.toStdString());
    ds->computeStatistics();
    m_testDs=bundle.test;
    if(m_testDs) m_testDs->computeStatistics();
    const bool isCls=(detectTask(ds.get())!=TaskType::Regression);

    const auto hidden=parseH(m_cfg.hiddenLayout);
    const auto act=parseA(m_cfg.activation);

    QStringList   configs;
    QVector<double> trainLoss,trainErr,valErr,testErr;
    int ph=1;   // phase slot counter (increments across baseline + phases)

    // ── 0. Baseline ───────────────────────────────────────────────────────
    auto netBase=buildNet(ds.get(),hidden,act);netBase->setTrainingMode(false);
    auto base=runBaseline(netBase.get(),ds.get(),ph++,dsName);
    if(m_stop.load())return false;
    // Save Standard's best weights — every subsequent phase must match or beat it
    const auto standardWeights = NSProblem::flatten(*netBase);
    configs<<(m_cfg.equalBudgetBaseline?"Standard (no bounds, equal budget)":"Standard (no bounds)");
    trainLoss<<base.trainLoss;trainErr<<base.trainErr;valErr<<base.valErr;testErr<<base.testErr;

    // ── 1. Phase 1: bounds discovery ─────────────────────────────────────
    auto netB=buildNet(ds.get(),hidden,act);netB->setTrainingMode(false);
    Bounds b=runPhase1(netB.get(),ds.get(),ph++,dsName);
    if(m_stop.load())return false;

    // ── 2. Phase 2: train within bounds ──────────────────────────────────
    bool fromStandard=false;   // Phase 3 starts from the Standard weights (Phase 1+2 was worse)
    if(m_cfg.opt2.toLower()!="none"&&!m_cfg.opt2.isEmpty()){
        if(m_cfg.opt2.toLower()=="ge_grammar") b=runPhase2GE(netB.get(),ds.get(),b,ph++);
        else                                    b=runPhase2Opt(netB.get(),ds.get(),b,ph++);
        auto r=measureNet(netB.get(),ds.get());

        // The row reports what Phase 1+2 actually achieved. (It used to be overwritten with
        // the Standard numbers whenever Phase 1+2 was worse, so the table could never show
        // the phases losing.) If it is worse on train or validation, the Standard weights
        // are carried into Phase 3 instead (the final network is never worse than Standard
        // on the validation split).
        const double ph2Primary = isCls ? r.trainErr   : r.trainLoss;
        const double stdPrimary = isCls ? base.trainErr : base.trainLoss;
        const bool ph2TrainWorse = (ph2Primary > stdPrimary);
        const bool ph2ValWorse   = (m_cfg.valSplit>0 && base.valErr>=0
                                    && r.valErr>=0 && r.valErr>base.valErr);
        // with the GE phase, Phase 2 only chooses bounds: the row is the Phase-1 network (it was
        // labelled as if Phase 2 had trained it)
        configs<<(m_cfg.opt2.toLower()=="ge_grammar"?"Phase 1+2 (GE bounds, Phase-1 weights)":"Phase 1+2");trainLoss<<r.trainLoss;trainErr<<r.trainErr;valErr<<r.valErr;testErr<<r.testErr;
        if(ph2TrainWorse || ph2ValWorse){ NSProblem::unflatten(*netB, standardWeights); fromStandard=true; }
        if(m_stop.load())return false;
    }

    // ── 3. Phase 3: final training ────────────────────────────────────────
    if(m_cfg.opt3.toLower()!="none"&&!m_cfg.opt3.isEmpty()){
        // Save Phase 2 result — Phase 3 must never be worse
        // reference = the weights Phase 3 starts from (Phase 1+2, or Standard if those were kept)
        const auto phase2Weights = NSProblem::flatten(*netB);
        const RunResult ref2     = measureNet(netB.get(),ds.get());
        const double phase2Loss  = ref2.trainLoss;
        const double phase2Err   = ref2.trainErr;
        const double phase2Val   = ref2.valErr;

        runPhase3(netB.get(),ds.get(),b,ph++);
        auto r=measureNet(netB.get(),ds.get());

        // Phase 1+2+3 must NEVER be worse than Phase 1+2 on the primary metric.
        // For classification compare Error%; for regression compare Loss/MSE.
        // Also reject Phase 3 if validation worsens (overfitting).
        const double ph3Primary  = isCls ? r.trainErr  : r.trainLoss;
        const double ph2Primary  = isCls ? phase2Err   : phase2Loss;
        const bool ph3TrainWorse = (ph3Primary > ph2Primary);
        const bool ph3ValWorse   = (m_cfg.valSplit>0 && phase2Val>=0
                                    && r.valErr>=0 && r.valErr>phase2Val);
        // with the Standard weights carried in, the row is "Standard + Phase 3" (it was labelled
        // Phase 1+2+3 although no Phase-1/2 weights were used)
        configs<<(fromStandard?"Standard + Phase 3 (Phase 1+2 rejected)":"Phase 1+2+3");trainLoss<<r.trainLoss;trainErr<<r.trainErr;valErr<<r.valErr;testErr<<r.testErr;
        if(ph3TrainWorse || ph3ValWorse) NSProblem::unflatten(*netB, phase2Weights);   // final network: never worse
    }

    emit datasetFinished(dsName,configs,trainLoss,trainErr,valErr,testErr,isCls);
    m_lastNet = netB;   // save for allDone
    m_lastDs  = ds;
    return true;
}

// ── run ───────────────────────────────────────────────────────────────────────
void EvolBoundsWorker::run(){
    try{
        if(m_cfg.datasetPaths.isEmpty())throw std::runtime_error("No datasets queued.");
        m_stop.store(false);

        std::shared_ptr<NeuralNetwork> lastNet;
        std::shared_ptr<Dataset>       lastDs;

        for(int i=0;i<m_cfg.datasetPaths.size()&&!m_stop.load();++i){
            if(!runOnDataset(i,m_cfg.datasetPaths[i]))break;
        }
        emit allDone(m_lastNet, m_lastDs);
    }catch(const std::exception&ex){
        emit errorOccurred(QString::fromStdString(ex.what()));
    }
}
} // namespace NeuralStudio

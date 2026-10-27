#pragma once
// GEEngine.h — Mode 6 (GE Studio): evolutionary loop over a BNF grammar.
// Two genotypes share one phenotype pipeline:
//   Codons         classic Grammatical Evolution (integer chromosome, codon mod n)
//   Probabilistic  Probabilistic GE: every non-terminal owns a distribution over
//                  its productions; the distributions are learned (PBIL-style)
//                  from the elite derivations. Individuals are choice traces.
// Pure STL — no Qt.
#include "Grammar.h"
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace NeuralStudio::ge {

enum class Genotype { Codons, Probabilistic };
enum class Crossover { OnePoint, Local };

struct GEConfig {
    Genotype  genotype = Genotype::Codons;
    int       popSize = 40;
    int       generations = 40;

    // Codon genotype
    int       genomeLen = 120;
    int       codonMax  = 256;
    int       maxWraps  = 2;

    // Derivation limits
    int       maxDepth = 24;
    int       maxNodes = 4000;

    // Generation scheme (Tsoulos-style): the best (1-replaceRate)*pop survive
    // unchanged, the rest are replaced by new individuals each generation.
    double    replaceRate = 0.30;

    // Codon operators
    int       tournament   = 4;
    double    mutationRate = 0.05;
    Crossover crossover    = Crossover::OnePoint;
    double    localRate    = 0.10;   // pcr — chance a chromosome gets Local Crossover
    int       localNcr     = 3;      // Ncr — partners tried per chromosome

    // Probabilistic genotype
    double    pgeAlpha = 0.30;       // learning rate
    double    pgeElite = 0.25;       // fraction of the population used as elite
    double    pgeFloor = 0.02;       // min probability of any production

    unsigned  seed = 1;
};

struct Individual {
    std::vector<int>    genes;     // Codons genotype
    std::vector<Choice> trace;     // Probabilistic genotype (and for reporting)
    std::string         text;      // phenotype text (program)
    double              fitness = std::numeric_limits<double>::infinity();
    bool                valid = false;
    int                 codonsUsed = 0;
};

class GEEngine {
public:
    // Evaluate a derivation tree. Return +inf (or NaN) for an unusable program.
    using EvalFn = std::function<double(const Node&)>;

    GEEngine(const Grammar& g, const GEConfig& cfg, EvalFn fn);

    void initPopulation();
    void step();                               // one generation

    int  generation()  const { return m_gen; }
    long evaluations() const { return m_evals; }   // EvalFn calls
    const Individual& best() const { return m_pop.front(); }
    const std::vector<Individual>& population() const { return m_pop; }
    bool derive(const Individual& ind, Node& tree) const;

    double meanFitness() const;                // over valid individuals
    double diversity() const;                  // unique programs / population

    // Probabilistic genotype state, indexed [nt][production].
    const std::vector<std::vector<double>>& probs() const { return m_probs; }
    void setProbs(const std::vector<std::vector<double>>& p);

private:
    const Grammar& m_g;
    GEConfig       m_cfg;
    EvalFn         m_eval;
    std::mt19937_64 m_rng;
    std::vector<Individual> m_pop;
    std::vector<std::vector<double>> m_probs;
    int  m_gen = 0;
    long m_evals = 0;

    Individual randomIndividual();
    void evaluate(Individual& ind);
    void sortPop();
    int  tournament();
    void onePoint(const Individual& a, const Individual& b,
                  Individual& c1, Individual& c2);
    void mutate(Individual& ind);
    void localCrossover(Individual& g);
    void stepCodons();
    void stepProbabilistic();
    void updateProbs();
    void normalise(std::vector<double>& p) const;
    int  randInt(int lo, int hi);
    double rand01();
};

} // namespace NeuralStudio::ge

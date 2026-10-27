#include "GEEngine.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace NeuralStudio::ge {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

// Replays a recorded sequence of choices.
struct TraceSource : ChoiceSource {
    const std::vector<Choice>& t;
    size_t pos = 0;
    explicit TraceSource(const std::vector<Choice>& tr) : t(tr) {}
    int choose(int /*nt*/, int /*n*/) override {
        if (pos >= t.size()) return -1;
        return t[pos++].prod;
    }
};

// Samples choices from per-non-terminal distributions.
struct ProbSource : ChoiceSource {
    const std::vector<std::vector<double>>& p;
    std::mt19937_64& rng;
    ProbSource(const std::vector<std::vector<double>>& pr, std::mt19937_64& r)
        : p(pr), rng(r) {}
    int choose(int nt, int n) override {
        const auto& d = p[nt];
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng), acc = 0.0;
        for (int k = 0; k < n; ++k) {
            acc += d[k];
            if (r <= acc) return k;
        }
        return n - 1;
    }
};
} // namespace

GEEngine::GEEngine(const Grammar& g, const GEConfig& cfg, EvalFn fn)
    : m_g(g), m_cfg(cfg), m_eval(std::move(fn)), m_rng(cfg.seed) {
    m_probs.resize(g.rules().size());
    for (size_t i = 0; i < g.rules().size(); ++i) {
        const size_t n = g.rules()[i].prods.size();
        m_probs[i].assign(n, 1.0 / static_cast<double>(n));
    }
}

void GEEngine::setProbs(const std::vector<std::vector<double>>& p) {
    if (p.size() != m_probs.size()) return;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i].size() == m_probs[i].size()) m_probs[i] = p[i];
}

int GEEngine::randInt(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(m_rng);
}
double GEEngine::rand01() {
    return std::uniform_real_distribution<double>(0.0, 1.0)(m_rng);
}

bool GEEngine::derive(const Individual& ind, Node& tree) const {
    Grammar::DeriveOptions o; o.maxDepth = m_cfg.maxDepth; o.maxNodes = m_cfg.maxNodes;
    if (m_cfg.genotype == Genotype::Codons) {
        CodonSource src(ind.genes, m_cfg.maxWraps);
        return m_g.derive(src, tree, o, nullptr);
    }
    TraceSource src(ind.trace);
    return m_g.derive(src, tree, o, nullptr);
}

// ── individuals ───────────────────────────────────────────────────────────────
Individual GEEngine::randomIndividual() {
    Individual ind;
    if (m_cfg.genotype == Genotype::Codons) {
        ind.genes.resize(m_cfg.genomeLen);
        for (int& c : ind.genes) c = randInt(0, m_cfg.codonMax - 1);
    } else {
        Grammar::DeriveOptions o; o.maxDepth = m_cfg.maxDepth; o.maxNodes = m_cfg.maxNodes;
        ProbSource src(m_probs, m_rng);
        Node tmp;
        m_g.derive(src, tmp, o, &ind.trace);
    }
    return ind;
}

void GEEngine::evaluate(Individual& ind) {
    Grammar::DeriveOptions o; o.maxDepth = m_cfg.maxDepth; o.maxNodes = m_cfg.maxNodes;
    Node tree;
    bool ok;
    if (m_cfg.genotype == Genotype::Codons) {
        CodonSource src(ind.genes, m_cfg.maxWraps);
        ok = m_g.derive(src, tree, o, &ind.trace);
        ind.codonsUsed = src.codonsUsed();
    } else {
        TraceSource src(ind.trace);
        std::vector<Choice> dummy;
        ok = m_g.derive(src, tree, o, &dummy);
        ind.codonsUsed = static_cast<int>(ind.trace.size());
    }
    if (!ok) { ind.valid = false; ind.fitness = kInf; ind.text.clear(); return; }
    ind.text = Grammar::text(tree);
    const double f = m_eval(tree);
    ++m_evals;
    if (!std::isfinite(f)) { ind.valid = false; ind.fitness = kInf; }
    else { ind.valid = true; ind.fitness = f; }
}

void GEEngine::sortPop() {
    std::stable_sort(m_pop.begin(), m_pop.end(),
        [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
}

void GEEngine::initPopulation() {
    m_pop.clear();
    m_pop.reserve(m_cfg.popSize);
    for (int i = 0; i < m_cfg.popSize; ++i) {
        Individual ind = randomIndividual();
        evaluate(ind);
        m_pop.push_back(std::move(ind));
    }
    sortPop();
    m_gen = 0;
}

// ── codon operators ───────────────────────────────────────────────────────────
int GEEngine::tournament() {
    int best = randInt(0, static_cast<int>(m_pop.size()) - 1);
    for (int i = 1; i < m_cfg.tournament; ++i) {
        const int c = randInt(0, static_cast<int>(m_pop.size()) - 1);
        if (m_pop[c].fitness < m_pop[best].fitness) best = c;
    }
    return best;
}

void GEEngine::onePoint(const Individual& a, const Individual& b,
                        Individual& c1, Individual& c2) {
    const int n = static_cast<int>(a.genes.size());
    const int p = n > 1 ? randInt(1, n - 1) : 0;
    c1 = Individual{}; c2 = Individual{};
    c1.genes.assign(a.genes.begin(), a.genes.begin() + p);
    c1.genes.insert(c1.genes.end(), b.genes.begin() + p, b.genes.end());
    c2.genes.assign(b.genes.begin(), b.genes.begin() + p);
    c2.genes.insert(c2.genes.end(), a.genes.begin() + p, a.genes.end());
}

void GEEngine::mutate(Individual& ind) {
    for (int& c : ind.genes)
        if (rand01() <= m_cfg.mutationRate) c = randInt(0, m_cfg.codonMax - 1);
}

// Local Crossover (Tsoulos, Charilogis, Tsalikakis — Algorithms 2024):
// repeatedly one-point-cross a chromosome with random partners and keep any
// offspring that is not worse.
void GEEngine::localCrossover(Individual& g) {
    for (int i = 0; i < m_cfg.localNcr; ++i) {
        const Individual& x = m_pop[randInt(0, static_cast<int>(m_pop.size()) - 1)];
        Individual g1, g2;
        onePoint(g, x, g1, g2);
        evaluate(g1);
        if (g1.fitness <= g.fitness) { g = std::move(g1); continue; }
        evaluate(g2);
        if (g2.fitness <= g.fitness) g = std::move(g2);
    }
}

void GEEngine::stepCodons() {
    int nChild = static_cast<int>(std::lround(m_cfg.replaceRate * m_cfg.popSize));
    nChild = std::max(2, std::min(m_cfg.popSize, nChild));
    std::vector<Individual> kids;
    kids.reserve(nChild + 1);
    while (static_cast<int>(kids.size()) < nChild) {
        const Individual& p1 = m_pop[tournament()];
        const Individual& p2 = m_pop[tournament()];
        Individual c1, c2;
        onePoint(p1, p2, c1, c2);
        mutate(c1); mutate(c2);
        evaluate(c1); evaluate(c2);
        kids.push_back(std::move(c1));
        kids.push_back(std::move(c2));
    }
    kids.resize(nChild);
    const int base = static_cast<int>(m_pop.size()) - nChild;
    for (int i = 0; i < nChild; ++i) m_pop[base + i] = std::move(kids[i]);
    sortPop();

    if (m_cfg.crossover == Crossover::Local) {
        for (size_t i = 0; i < m_pop.size(); ++i)
            if (rand01() <= m_cfg.localRate) localCrossover(m_pop[i]);
        sortPop();
    }
}

// ── probabilistic genotype ────────────────────────────────────────────────────
void GEEngine::normalise(std::vector<double>& p) const {
    const int n = static_cast<int>(p.size());
    const double fl = std::min(m_cfg.pgeFloor, 0.5 / std::max(1, n));
    double s = 0;
    for (double& v : p) { v = std::max(v, fl); s += v; }
    for (double& v : p) v /= s;
}

void GEEngine::updateProbs() {
    const int nElite = std::max(1, static_cast<int>(
        std::ceil(m_cfg.pgeElite * static_cast<double>(m_pop.size()))));
    std::vector<std::vector<double>> cnt(m_probs.size());
    for (size_t i = 0; i < cnt.size(); ++i) cnt[i].assign(m_probs[i].size(), 0.0);
    int used = 0;
    for (const Individual& ind : m_pop) {
        if (used >= nElite) break;
        if (!ind.valid) continue;
        for (const Choice& c : ind.trace) cnt[c.nt][c.prod] += 1.0;
        ++used;
    }
    for (size_t i = 0; i < m_probs.size(); ++i) {
        if (m_probs[i].size() < 2) continue;
        double tot = 0; for (double v : cnt[i]) tot += v;
        if (tot <= 0) continue;
        for (size_t k = 0; k < m_probs[i].size(); ++k)
            m_probs[i][k] = (1.0 - m_cfg.pgeAlpha) * m_probs[i][k]
                          + m_cfg.pgeAlpha * (cnt[i][k] / tot);
        normalise(m_probs[i]);
    }
}

void GEEngine::stepProbabilistic() {
    updateProbs();
    int nNew = static_cast<int>(std::lround(m_cfg.replaceRate * m_cfg.popSize));
    nNew = std::max(1, std::min(m_cfg.popSize, nNew));
    const int base = static_cast<int>(m_pop.size()) - nNew;
    for (int i = 0; i < nNew; ++i) {
        Individual ind = randomIndividual();
        evaluate(ind);
        m_pop[base + i] = std::move(ind);
    }
    sortPop();
}

void GEEngine::step() {
    if (m_cfg.genotype == Genotype::Codons) stepCodons();
    else                                    stepProbabilistic();
    ++m_gen;
}

double GEEngine::meanFitness() const {
    double s = 0; int n = 0;
    for (const auto& i : m_pop) if (i.valid) { s += i.fitness; ++n; }
    return n ? s / n : kInf;
}

double GEEngine::diversity() const {
    std::set<std::string> u;
    for (const auto& i : m_pop) if (i.valid) u.insert(i.text);
    return m_pop.empty() ? 0.0 : static_cast<double>(u.size()) / m_pop.size();
}

} // namespace NeuralStudio::ge

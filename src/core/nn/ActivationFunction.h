#pragma once
#include <cmath>
#include <QString>

namespace NeuralStudio {

enum class Activation { Linear, ReLU, Sigmoid, Tanh };

inline double applyActivation(Activation a, double x) {
    switch (a) {
        case Activation::Linear:  return x;
        case Activation::ReLU:    return x > 0.0 ? x : 0.0;
        case Activation::Sigmoid: return 1.0 / (1.0 + std::exp(-x));
        case Activation::Tanh:    return std::tanh(x);
    }
    return x;
}

// activatedVal = the already-computed activation output (avoids re-computing)
inline double activationDerivative(Activation a, double z, double activatedVal) {
    switch (a) {
        case Activation::Linear:  return 1.0;
        case Activation::ReLU:    return z > 0.0 ? 1.0 : 0.0;
        case Activation::Sigmoid: return activatedVal * (1.0 - activatedVal);
        case Activation::Tanh:    return 1.0 - activatedVal * activatedVal;
    }
    return 1.0;
}

inline QString activationName(Activation a) {
    switch (a) {
        case Activation::Linear:  return "Linear";
        case Activation::ReLU:    return "ReLU";
        case Activation::Sigmoid: return "Sigmoid";
        case Activation::Tanh:    return "Tanh";
    }
    return "Unknown";
}

inline Activation activationFromName(const QString& name) {
    if (name == "ReLU")    return Activation::ReLU;
    if (name == "Sigmoid") return Activation::Sigmoid;
    if (name == "Tanh")    return Activation::Tanh;
    return Activation::Linear;
}

} // namespace NeuralStudio

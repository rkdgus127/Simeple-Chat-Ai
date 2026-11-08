#include "Adam.h"

#include <cmath>
#include <stdexcept>

Adam::Adam(
    std::vector<std::shared_ptr<Tensor>> parameters,
    double learningRate,
    double beta1,
    double beta2,
    double epsilon
)
    : parameters(std::move(parameters)),
      learningRate(learningRate),
      beta1(beta1),
      beta2(beta2),
      epsilon(epsilon),
      timestep(0) {

    for (const auto& parameter : this->parameters) {
        if (!parameter->isRequiresGrad()) {
            throw std::invalid_argument("Parameter must require gradients");
        }

        states.push_back({
            std::vector<double>(parameter->size(), 0.0),
            std::vector<double>(parameter->size(), 0.0)
        });
    }
}

void Adam::step() {
    timestep++;

    double biasCorrection1 = 1.0 - std::pow(beta1, timestep);
    double biasCorrection2 = 1.0 - std::pow(beta2, timestep);

    for (size_t p = 0; p < parameters.size(); p++) {
        auto& parameter = parameters[p];
        auto& state = states[p];

        const auto& data = parameter->getData();
        const auto& grad = parameter->getGrad();

        if (grad.size() != data.size()) {
            throw std::runtime_error("Gradient size mismatch");
        }

        std::vector<double> updatedData = data;

        for (size_t i = 0; i < data.size(); i++) {
            state.m[i] =
                beta1 * state.m[i] +
                (1.0 - beta1) * grad[i];

            state.v[i] =
                beta2 * state.v[i] +
                (1.0 - beta2) * grad[i] * grad[i];

            double correctedM = state.m[i] / biasCorrection1;
            double correctedV = state.v[i] / biasCorrection2;

            updatedData[i] -= learningRate *
                correctedM /
                (std::sqrt(correctedV) + epsilon);
        }

        parameter->setData(updatedData);
    }
}
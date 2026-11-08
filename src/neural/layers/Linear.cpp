#include "Linear.h"

#include <random>
#include <cmath>
#include <stdexcept>

Linear::Linear(size_t inputSize, size_t outputSize)
    : inputSize(inputSize),
      outputSize(outputSize) {

    if (inputSize == 0 || outputSize == 0) {
        throw std::invalid_argument("Linear dimensions cannot be zero");
    }

    std::mt19937 generator(std::random_device{}());

    double limit = std::sqrt(6.0 / (inputSize + outputSize));

    std::uniform_real_distribution<double> distribution(
        -limit,
        limit
    );

    std::vector<double> weightData(inputSize * outputSize);

    for (double& value : weightData) {
        value = distribution(generator);
    }

    std::vector<double> biasData(outputSize, 0.0);

    weights = Tensor::create(
        weightData,
        {inputSize, outputSize},
        true
    );

    bias = Tensor::create(
        biasData,
        {1, outputSize},
        true
    );
}

std::shared_ptr<Tensor> Linear::forward(
    const std::shared_ptr<Tensor>& input
) {
    if (input->getShape().size() != 2 ||
        input->getShape()[1] != inputSize) {
        throw std::invalid_argument("Linear input shape mismatch");
        }

    return input->matmul(weights)->addBias(bias);
}

std::shared_ptr<Tensor> Linear::getWeights() const {
    return weights;
}

std::shared_ptr<Tensor> Linear::getBias() const {
    return bias;
}
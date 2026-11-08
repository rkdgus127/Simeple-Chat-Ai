#include "LayerNorm.h"

#include <stdexcept>
#include <vector>

LayerNorm::LayerNorm(size_t embeddingDim, double epsilon) {
    if (embeddingDim == 0 || epsilon <= 0.0) {
        throw std::invalid_argument("Invalid LayerNorm configuration");
    }

    this->embeddingDim = embeddingDim;
    this->epsilon = epsilon;

    gamma = Tensor::create(
        std::vector<double>(embeddingDim, 1.0),
        {1, embeddingDim},
        true
    );

    beta = Tensor::create(
        std::vector<double>(embeddingDim, 0.0),
        {1, embeddingDim},
        true
    );
}

std::shared_ptr<Tensor> LayerNorm::forward(
    const std::shared_ptr<Tensor>& input
) {
    if (input->getShape().size() != 2 ||
        input->getShape()[1] != embeddingDim) {
        throw std::invalid_argument("Invalid LayerNorm input shape");
        }

    return input->layerNorm(gamma, beta, epsilon);
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::getParameters() {
    return {gamma, beta};
}
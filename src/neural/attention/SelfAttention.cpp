#include "SelfAttention.h"

#include <cmath>
#include <stdexcept>

SelfAttention::SelfAttention(size_t embeddingDim)
    : embeddingDim(embeddingDim),
      queryLayer(embeddingDim, embeddingDim),
      keyLayer(embeddingDim, embeddingDim),
      valueLayer(embeddingDim, embeddingDim) {

    if (embeddingDim == 0) {
        throw std::invalid_argument("Embedding dimension cannot be zero");
    }
}

std::shared_ptr<Tensor> SelfAttention::forward(
    const std::shared_ptr<Tensor>& input
) {
    if (input->getShape().size() != 2 ||
        input->getShape()[1] != embeddingDim) {
        throw std::invalid_argument("Invalid input shape");
        }

    auto query = queryLayer.forward(input);
    auto key = keyLayer.forward(input);
    auto value = valueLayer.forward(input);

    auto scores = query->matmul(key->transpose());

    double scale = 1.0 / std::sqrt(static_cast<double>(embeddingDim));

    std::vector<double> scaleData(
        scores->size(),
        scale
    );

    auto scaleTensor = Tensor::create(
        scaleData,
        scores->getShape(),
        false
    );

    scores = scores->multiply(scaleTensor);

    auto attentionWeights = scores->softmax();

    return attentionWeights->matmul(value);
}

std::vector<std::shared_ptr<Tensor>> SelfAttention::getParameters() const {
    return {
        queryLayer.getWeights(),
        queryLayer.getBias(),
        keyLayer.getWeights(),
        keyLayer.getBias(),
        valueLayer.getWeights(),
        valueLayer.getBias()
    };
}
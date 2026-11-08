#include "PositionalEncoding.h"

#include <cmath>
#include <stdexcept>
#include <vector>

PositionalEncoding::PositionalEncoding(size_t embeddingDim)
    : embeddingDim(embeddingDim) {

    if (embeddingDim == 0) {
        throw std::invalid_argument("Embedding dimension cannot be zero");
    }
}

std::shared_ptr<Tensor> PositionalEncoding::forward(
    const std::shared_ptr<Tensor>& input
) const {

    const auto& shape = input->getShape();

    if (shape.size() != 2) {
        throw std::invalid_argument("Input must be a 2D tensor");
    }

    if (shape[1] != embeddingDim) {
        throw std::invalid_argument("Embedding dimension mismatch");
    }

    size_t sequenceLength = shape[0];

    std::vector<double> positionalData(
        sequenceLength * embeddingDim,
        0.0
    );

    for (size_t pos = 0; pos < sequenceLength; pos++) {
        for (size_t i = 0; i < embeddingDim; i++) {

            double exponent =
                static_cast<double>(2 * (i / 2)) /
                static_cast<double>(embeddingDim);

            double angle =
                static_cast<double>(pos) /
                std::pow(10000.0, exponent);

            if (i % 2 == 0) {
                positionalData[pos * embeddingDim + i] =
                    std::sin(angle);
            } else {
                positionalData[pos * embeddingDim + i] =
                    std::cos(angle);
            }
        }
    }

    auto positionalEncoding = Tensor::create(
        positionalData,
        {sequenceLength, embeddingDim},
        false
    );

    return input->add(positionalEncoding);
}

size_t PositionalEncoding::getEmbeddingDim() const {
    return embeddingDim;
}
#include "Embedding.h"

#include <cmath>
#include <random>
#include <stdexcept>

Embedding::Embedding(size_t vocabSize, size_t embeddingDim)
    : vocabSize(vocabSize),
      embeddingDim(embeddingDim) {

    if (vocabSize == 0 || embeddingDim == 0) {
        throw std::invalid_argument("Invalid embedding dimensions");
    }

    std::mt19937 generator(std::random_device{}());

    double limit = std::sqrt(
        6.0 / static_cast<double>(vocabSize + embeddingDim)
    );

    std::uniform_real_distribution<double> distribution(-limit, limit);

    std::vector<double> data(vocabSize * embeddingDim);

    for (double& value : data) {
        value = distribution(generator);
    }

    weights = Tensor::create(
        data,
        {vocabSize, embeddingDim},
        true
    );
}

std::shared_ptr<Tensor> Embedding::forward(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        throw std::invalid_argument("Tokens cannot be empty");
    }

    std::vector<double> oneHot(tokens.size() * vocabSize, 0.0);

    for (size_t i = 0; i < tokens.size(); i++) {
        int token = tokens[i];

        if (token < 0 || static_cast<size_t>(token) >= vocabSize) {
            throw std::out_of_range("Token ID out of range");
        }

        oneHot[i * vocabSize + token] = 1.0;
    }

    auto input = Tensor::create(
        oneHot,
        {tokens.size(), vocabSize},
        false
    );

    return input->matmul(weights);
}

std::shared_ptr<Tensor> Embedding::getWeights() const {
    return weights;
}

size_t Embedding::getVocabSize() const {
    return vocabSize;
}

size_t Embedding::getEmbeddingDim() const {
    return embeddingDim;
}
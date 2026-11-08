#include "MultiHeadAttention.h"

#include <cmath>
#include <stdexcept>

MultiHeadAttention::MultiHeadAttention(
    size_t embeddingDim,
    size_t numHeads
) {
    if (embeddingDim == 0 || numHeads == 0 ||
        embeddingDim % numHeads != 0) {
        throw std::invalid_argument("Invalid attention dimensions");
    }

    this->embeddingDim = embeddingDim;
    this->numHeads = numHeads;
    this->headDim = embeddingDim / numHeads;

    for (size_t i = 0; i < numHeads; i++) {
        queryLayers.push_back(
            std::make_unique<Linear>(embeddingDim, headDim)
        );

        keyLayers.push_back(
            std::make_unique<Linear>(embeddingDim, headDim)
        );

        valueLayers.push_back(
            std::make_unique<Linear>(embeddingDim, headDim)
        );
    }

    outputLayer = std::make_unique<Linear>(
        embeddingDim,
        embeddingDim
    );
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(
    const std::shared_ptr<Tensor>& input
) {
    if (input->getShape().size() != 2 ||
        input->getShape()[1] != embeddingDim) {
        throw std::invalid_argument("Invalid input shape");
    }

    std::vector<std::shared_ptr<Tensor>> headOutputs;

    for (size_t i = 0; i < numHeads; i++) {
        auto query = queryLayers[i]->forward(input);
        auto key = keyLayers[i]->forward(input);
        auto value = valueLayers[i]->forward(input);

        auto scores = query->matmul(key->transpose());

        double scale = 1.0 / std::sqrt(static_cast<double>(headDim));

        std::vector<double> scaleData(scores->size(), scale);

        auto scaleTensor = Tensor::create(
            scaleData,
            scores->getShape(),
            false
        );

        scores = scores->multiply(scaleTensor);

        auto attentionWeights = scores->softmax();

        auto output = attentionWeights->matmul(value);

        headOutputs.push_back(output);
    }

    auto concatenated = Tensor::concatColumns(headOutputs);

    return outputLayer->forward(concatenated);
}

std::vector<std::shared_ptr<Tensor>>
MultiHeadAttention::getParameters() {
    std::vector<std::shared_ptr<Tensor>> parameters;

    for (size_t i = 0; i < numHeads; i++) {
        parameters.push_back(queryLayers[i]->getWeights());
        parameters.push_back(queryLayers[i]->getBias());

        parameters.push_back(keyLayers[i]->getWeights());
        parameters.push_back(keyLayers[i]->getBias());

        parameters.push_back(valueLayers[i]->getWeights());
        parameters.push_back(valueLayers[i]->getBias());
    }

    parameters.push_back(outputLayer->getWeights());
    parameters.push_back(outputLayer->getBias());

    return parameters;
}
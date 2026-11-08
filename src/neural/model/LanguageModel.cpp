#include "LanguageModel.h"

#include <stdexcept>

LanguageModel::LanguageModel(
    size_t vocabSize,
    size_t embeddingDim,
    size_t numHeads,
    size_t numLayers,
    size_t hiddenDim
) {
    if (vocabSize == 0 || embeddingDim == 0 ||
        numHeads == 0 || numLayers == 0 ||
        embeddingDim % numHeads != 0) {
        throw std::invalid_argument("Invalid LanguageModel configuration");
    }

    this->vocabSize = vocabSize;
    this->embeddingDim = embeddingDim;
    this->numLayers = numLayers;

    embedding = std::make_unique<Embedding>(
        vocabSize,
        embeddingDim
    );

    positionalEncoding = std::make_unique<PositionalEncoding>(
        embeddingDim
    );

    for (size_t i = 0; i < numLayers; i++) {
        blocks.push_back(
            std::make_unique<TransformerBlock>(
                embeddingDim,
                numHeads,
                hiddenDim
            )
        );
    }

    outputLayer = std::make_unique<Linear>(
        embeddingDim,
        vocabSize
    );
}

std::shared_ptr<Tensor> LanguageModel::forward(
    const std::vector<int>& tokens,
    bool causal
) {
    if (tokens.empty()) {
        throw std::invalid_argument("Input tokens cannot be empty");
    }

    auto output = embedding->forward(tokens);

    output = positionalEncoding->forward(output);

    for (auto& block : blocks) {
        output = block->forward(output, causal);
    }

    return outputLayer->forward(output);
}

std::vector<std::shared_ptr<Tensor>> LanguageModel::getParameters() {
    std::vector<std::shared_ptr<Tensor>> parameters;

    auto embeddingParameters = std::vector<std::shared_ptr<Tensor>>{
        embedding->getWeights()
    };

    parameters.insert(
        parameters.end(),
        embeddingParameters.begin(),
        embeddingParameters.end()
    );

    for (auto& block : blocks) {
        auto blockParameters = block->getParameters();

        parameters.insert(
            parameters.end(),
            blockParameters.begin(),
            blockParameters.end()
        );
    }

    auto outputParameters = std::vector<std::shared_ptr<Tensor>>{
        outputLayer->getWeights(),
        outputLayer->getBias()
    };

    parameters.insert(
        parameters.end(),
        outputParameters.begin(),
        outputParameters.end()
    );

    return parameters;
}
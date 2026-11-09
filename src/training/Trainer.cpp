#include "Trainer.h"

#include <stdexcept>
#include <algorithm>
#include <random>

Trainer::Trainer(LanguageModel& model, double learningRate)
    : model(model), optimizer(model.getParameters(), learningRate) {}

double Trainer::trainStep(const std::vector<int>& tokens) {
    if (tokens.size() < 2) {
        throw std::invalid_argument("Training sequence must contain at least 2 tokens");
    }

    std::vector<int> inputTokens(tokens.begin(), tokens.end() - 1);
    std::vector<int> targetTokens(tokens.begin() + 1, tokens.end());

    auto prediction = model.forward(inputTokens, true);
    auto loss = prediction->crossEntropy(targetTokens);

    double lossValue = loss->getData()[0];

    loss->backward();
    optimizer.step();

    return lossValue;
}

double Trainer::trainEpoch(
    const std::vector<std::vector<int>>& dataset
) {
    if (dataset.empty()) {
        throw std::invalid_argument("Dataset cannot be empty");
    }

    std::vector<size_t> indices(dataset.size());

    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }

    static std::mt19937 randomEngine(std::random_device{}());

    std::shuffle(
        indices.begin(),
        indices.end(),
        randomEngine
    );

    double totalLoss = 0.0;

    for (size_t index : indices) {
        totalLoss += trainStep(dataset[index]);
    }

    return totalLoss / static_cast<double>(dataset.size());
}

double Trainer::evaluate(
    const std::vector<std::vector<int>>& dataset
) {
    if (dataset.empty()) {
        throw std::invalid_argument("Validation dataset cannot be empty");
    }

    double totalLoss = 0.0;

    for (const auto& tokens : dataset) {
        if (tokens.size() < 2) {
            throw std::invalid_argument("Validation sequence must contain at least 2 tokens");
        }

        std::vector<int> inputTokens(tokens.begin(), tokens.end() - 1);
        std::vector<int> targetTokens(tokens.begin() + 1, tokens.end());

        auto prediction = model.forward(inputTokens, true);
        auto loss = prediction->crossEntropy(targetTokens);

        totalLoss += loss->getData()[0];
    }

    return totalLoss / static_cast<double>(dataset.size());
}
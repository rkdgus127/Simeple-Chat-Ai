#include "Dataset.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <utility>

std::string Dataset::readText(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset");
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

void Dataset::load(
    const std::filesystem::path& path,
    BPETokenizer& tokenizer,
    double validationRatio
) {
    if (validationRatio <= 0.0 || validationRatio >= 1.0) {
        throw std::invalid_argument("Invalid validation ratio");
    }

    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset");
    }

    trainSequences.clear();
    validationSequences.clear();

    std::vector<std::vector<int>> sequences;

    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<int> tokens = tokenizer.encodeSequence(line);

        if (tokens.size() < 3) {
            continue;
        }

        sequences.push_back(std::move(tokens));
    }

    if (sequences.size() < 2) {
        throw std::runtime_error("Dataset requires at least 2 sequences");
    }

    std::mt19937 randomEngine(std::random_device{}());

    std::shuffle(
        sequences.begin(),
        sequences.end(),
        randomEngine
    );

    size_t validationCount = static_cast<size_t>(
        sequences.size() * validationRatio
    );

    validationCount = std::max<size_t>(1, validationCount);
    validationCount = std::min(validationCount, sequences.size() - 1);

    size_t trainCount = sequences.size() - validationCount;

    trainSequences.assign(
        sequences.begin(),
        sequences.begin() + static_cast<std::ptrdiff_t>(trainCount)
    );

    validationSequences.assign(
        sequences.begin() + static_cast<std::ptrdiff_t>(trainCount),
        sequences.end()
    );
}

const std::vector<std::vector<int>>& Dataset::getTrainSequences() const {
    return trainSequences;
}

const std::vector<std::vector<int>>& Dataset::getValidationSequences() const {
    return validationSequences;
}

size_t Dataset::trainSize() const {
    return trainSequences.size();
}

size_t Dataset::validationSize() const {
    return validationSequences.size();
}
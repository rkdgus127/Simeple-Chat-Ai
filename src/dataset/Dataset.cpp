#include "Dataset.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

std::string Dataset::readText(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

void Dataset::load(
    const std::string& path,
    BPETokenizer& tokenizer
) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset: " + path);
    }

    sequences.clear();

    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<int> tokens = tokenizer.encode(line);

        if (tokens.size() < 2) {
            continue;
        }

        sequences.push_back(std::move(tokens));
    }

    if (sequences.empty()) {
        throw std::runtime_error("Dataset contains no valid sequences");
    }
}

const std::vector<std::vector<int>>& Dataset::getSequences() const {
    return sequences;
}

size_t Dataset::size() const {
    return sequences.size();
}
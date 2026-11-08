#include "Dataset.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
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
    BPETokenizer& tokenizer
) {
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset");
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
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <map>
#include <vector>

#include "TokenizerSaver.h"

void TokenizerSaver::save(
    const BPETokenizer& tokenizer,
    const std::string& path
) {
    std::ofstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open tokenizer file");
    }

    const uint32_t magic = 0x42504554;
    const uint32_t version = 1;

    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    uint64_t vocabularySize = tokenizer.vocabulary.size();

    file.write(
        reinterpret_cast<const char*>(&vocabularySize),
        sizeof(vocabularySize)
    );

    for (const auto& [id, token] : tokenizer.vocabulary) {
        int32_t tokenId = id;
        uint64_t tokenLength = token.size();

        file.write(reinterpret_cast<const char*>(&tokenId), sizeof(tokenId));
        file.write(reinterpret_cast<const char*>(&tokenLength), sizeof(tokenLength));

        file.write(
            token.data(),
            static_cast<std::streamsize>(tokenLength)
        );
    }

    uint64_t mergeCount = tokenizer.merges.size();

    file.write(
        reinterpret_cast<const char*>(&mergeCount),
        sizeof(mergeCount)
    );

    for (const auto& merge : tokenizer.merges) {
        int32_t first = merge.first;
        int32_t second = merge.second;
        int32_t result = merge.result;

        file.write(reinterpret_cast<const char*>(&first), sizeof(first));
        file.write(reinterpret_cast<const char*>(&second), sizeof(second));
        file.write(reinterpret_cast<const char*>(&result), sizeof(result));
    }

    if (!file) {
        throw std::runtime_error("Failed to save tokenizer");
    }
}

void TokenizerSaver::load(
    BPETokenizer& tokenizer,
    const std::string& path
) {
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open tokenizer file");
    }

    uint32_t magic;
    uint32_t version;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (!file || magic != 0x42504554 || version != 1) {
        throw std::runtime_error("Invalid tokenizer file");
    }

    uint64_t vocabularySize;

    file.read(
        reinterpret_cast<char*>(&vocabularySize),
        sizeof(vocabularySize)
    );

    if (!file || vocabularySize > 1000000) {
        throw std::runtime_error("Invalid vocabulary size");
    }

    std::map<int, std::string> vocabulary;

    for (uint64_t i = 0; i < vocabularySize; i++) {
        int32_t tokenId;
        uint64_t tokenLength;

        file.read(reinterpret_cast<char*>(&tokenId), sizeof(tokenId));
        file.read(reinterpret_cast<char*>(&tokenLength), sizeof(tokenLength));

        if (!file || tokenLength > 10000000) {
            throw std::runtime_error("Invalid token length");
        }

        std::string token(tokenLength, '\0');

        file.read(
            token.data(),
            static_cast<std::streamsize>(tokenLength)
        );

        if (!file || !vocabulary.emplace(tokenId, std::move(token)).second) {
            throw std::runtime_error("Invalid vocabulary data");
        }
    }

    uint64_t mergeCount;

    file.read(
        reinterpret_cast<char*>(&mergeCount),
        sizeof(mergeCount)
    );

    if (!file || mergeCount > 1000000) {
        throw std::runtime_error("Invalid merge count");
    }

    std::vector<BPETokenizer::Merge> merges;

    merges.reserve(mergeCount);

    for (uint64_t i = 0; i < mergeCount; i++) {
        int32_t first;
        int32_t second;
        int32_t result;

        file.read(reinterpret_cast<char*>(&first), sizeof(first));
        file.read(reinterpret_cast<char*>(&second), sizeof(second));
        file.read(reinterpret_cast<char*>(&result), sizeof(result));

        if (!file) {
            throw std::runtime_error("Corrupted tokenizer file");
        }

        merges.push_back({
            first,
            second,
            result
        });
    }

    if (file.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Unexpected tokenizer data");
    }

    tokenizer.vocabulary = std::move(vocabulary);
    tokenizer.merges = std::move(merges);
}
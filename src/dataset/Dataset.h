#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "../tokenizer/BPETokenizer.h"

class Dataset {
private:
    std::vector<std::vector<int>> trainSequences;
    std::vector<std::vector<int>> validationSequences;

public:
    static std::string readText(const std::filesystem::path& path);

    void load(
        const std::filesystem::path& path,
        BPETokenizer& tokenizer,
        double validationRatio = 0.2
    );

    const std::vector<std::vector<int>>& getTrainSequences() const;
    const std::vector<std::vector<int>>& getValidationSequences() const;

    size_t trainSize() const;
    size_t validationSize() const;
};
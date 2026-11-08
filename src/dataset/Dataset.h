#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "../tokenizer/BPETokenizer.h"

class Dataset {
private:
    std::vector<std::vector<int>> sequences;

public:
    static std::string readText(const std::filesystem::path& path);

    void load(
        const std::filesystem::path& path,
        BPETokenizer& tokenizer
    );

    const std::vector<std::vector<int>>& getSequences() const;

    size_t size() const;
};
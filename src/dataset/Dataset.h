#pragma once

#include <string>
#include <vector>

#include "../tokenizer/BPETokenizer.h"

class Dataset {
private:
    std::vector<std::vector<int>> sequences;

public:
    static std::string readText(const std::string& path);

    void load(
        const std::string& path,
        BPETokenizer& tokenizer
    );

    const std::vector<std::vector<int>>& getSequences() const;

    size_t size() const;
};
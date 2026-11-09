#pragma once

#include <string>

#include "BPETokenizer.h"

class TokenizerSaver {
public:
    static void save(
        const BPETokenizer& tokenizer,
        const std::string& path
    );

    static void load(
        BPETokenizer& tokenizer,
        const std::string& path
    );
};
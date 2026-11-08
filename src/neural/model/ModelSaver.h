#pragma once

#include <string>

#include "LanguageModel.h"

class ModelSaver {
public:
    static void save(
        LanguageModel& model,
        const std::string& path
    );

    static void load(
        LanguageModel& model,
        const std::string& path
    );
};
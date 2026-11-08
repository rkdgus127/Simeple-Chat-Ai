#pragma once

#include <vector>

#include "../neural/model/LanguageModel.h"
#include "../optimizer/Adam.h"

class Trainer {
private:
    LanguageModel& model;
    Adam optimizer;

public:
    explicit Trainer(
        LanguageModel& model,
        double learningRate = 0.001
    );

    double trainStep(const std::vector<int>& tokens);

    double trainEpoch(
        const std::vector<std::vector<int>>& dataset
    );
};
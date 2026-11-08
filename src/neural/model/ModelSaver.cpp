#include "ModelSaver.h"

#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <vector>

void ModelSaver::save(
    LanguageModel& model,
    const std::string& path
) {
    std::ofstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open model file");
    }

    const uint32_t magic = 0x41494D44;
    const uint32_t version = 1;

    file.write(
        reinterpret_cast<const char*>(&magic),
        sizeof(magic)
    );

    file.write(
        reinterpret_cast<const char*>(&version),
        sizeof(version)
    );

    auto parameters = model.getParameters();

    uint64_t parameterCount = parameters.size();

    file.write(
        reinterpret_cast<const char*>(&parameterCount),
        sizeof(parameterCount)
    );

    for (const auto& parameter : parameters) {
        uint64_t size = parameter->size();

        file.write(
            reinterpret_cast<const char*>(&size),
            sizeof(size)
        );

        file.write(
            reinterpret_cast<const char*>(parameter->getData().data()),
            static_cast<std::streamsize>(size * sizeof(double))
        );
    }

    if (!file) {
        throw std::runtime_error("Failed to save model");
    }
}

void ModelSaver::load(
    LanguageModel& model,
    const std::string& path
) {
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open model file");
    }

    uint32_t magic;
    uint32_t version;

    file.read(
        reinterpret_cast<char*>(&magic),
        sizeof(magic)
    );

    file.read(
        reinterpret_cast<char*>(&version),
        sizeof(version)
    );

    if (!file || magic != 0x41494D44 || version != 1) {
        throw std::runtime_error("Invalid model file");
    }

    auto parameters = model.getParameters();

    uint64_t parameterCount;

    file.read(
        reinterpret_cast<char*>(&parameterCount),
        sizeof(parameterCount)
    );

    if (!file || parameterCount != parameters.size()) {
        throw std::runtime_error("Model parameter count mismatch");
    }

    for (const auto& parameter : parameters) {
        uint64_t size;

        file.read(
            reinterpret_cast<char*>(&size),
            sizeof(size)
        );

        if (!file || size != parameter->size()) {
            throw std::runtime_error("Model parameter size mismatch");
        }

        std::vector<double> data(size);

        file.read(
            reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(size * sizeof(double))
        );

        if (!file) {
            throw std::runtime_error("Corrupted model file");
        }

        parameter->setData(data);
    }

    if (file.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Unexpected data in model file");
    }
}
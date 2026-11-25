// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    /// Clear any state the policy carries between steps. No-op for a feedforward policy.
    virtual void reset() {}

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Every output, not just the action. A recurrent policy returns its next hidden
        // state alongside the action, and asking only for output 0 silently discards it.
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(output_name.release());
        }

        // Output 0 is the action; rsl_rl's exporter emits it first.
        output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        action.resize(output_shape[1]);

        // A recurrent policy exports its state as an input/output pair whose names differ
        // only by the "_in"/"_out" suffix -- ActorCriticRecurrent gives ("h_in", "h_out"),
        // and an LSTM adds ("c_in", "c_out"). Those inputs are not observations: they are
        // fed from the previous step's matching output, so pair them up here and carry
        // them in recurrent_state_. Anything left unpaired must come from the observation
        // manager, and is still reported if it is missing.
        for (size_t i = 0; i < input_names.size(); ++i) {
            const std::string name(input_names[i]);
            const std::string suffix("_in");
            if (name.size() <= suffix.size()
                || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
                continue;
            }
            const std::string paired = name.substr(0, name.size() - suffix.size()) + "_out";
            for (size_t j = 0; j < output_names.size(); ++j) {
                if (paired == output_names[j]) {
                    recurrent_state_[name] = std::vector<float>(input_sizes[i], 0.0f);
                    state_output_index_[name] = j;
                    break;
                }
            }
        }
        if (!recurrent_state_.empty()) {
            std::cout << "[OrtRunner] recurrent policy: carrying " << recurrent_state_.size()
                      << " state tensor(s) across steps" << std::endl;
        }
    }

    /// Zero the recurrent state, so a freshly entered controller starts from the same
    /// point training did rather than from whatever the last run left behind.
    void reset() override { reset_state_.store(true); }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        if (reset_state_.exchange(false)) {
            for (auto& entry : recurrent_state_) {
                std::fill(entry.second.begin(), entry.second.end(), 0.0f);
            }
        }

        // make sure all input names are in obs, ignoring the ones we carry ourselves
        for (const auto& name : input_names) {
            if (recurrent_state_.count(name) == 0 && obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto state_it = recurrent_state_.find(name_str);
            auto& input_data = (state_it != recurrent_state_.end()) ? state_it->second : obs.at(name_str);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

        // Carry each state output forward into the matching input for the next step.
        for (auto& entry : recurrent_state_) {
            const auto* out = output_tensors[state_output_index_.at(entry.first)].GetTensorData<float>();
            std::memcpy(entry.second.data(), out, entry.second.size() * sizeof(float));
        }

        // Copy output data
        auto floatarr = output_tensors.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;

    /// Recurrent state, keyed by input name, fed from the previous step's paired output.
    std::unordered_map<std::string, std::vector<float>> recurrent_state_;
    std::unordered_map<std::string, size_t> state_output_index_;
    std::atomic<bool> reset_state_{true};
};
};
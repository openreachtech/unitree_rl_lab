// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }

    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

// Runs any ONNX policy exported by isaaclab_rl's exporter.py (rsl_rl/exporter.py),
// including recurrent (GRU/LSTM) ones. A recurrent export declares extra inputs
// ("h_in", and "c_in" for LSTM) and extra outputs ("h_out"/"c_out") to carry the
// hidden/cell state across calls -- see _OnnxPolicyExporter.forward_gru/forward_lstm
// there. Those names are detected here by exact match and owned internally as
// persistent state (zero-initialized at construction, updated from every call's
// h_out/c_out), so callers only ever need to supply "obs" in act()'s map -- identical
// to the non-recurrent case. A plain (non-recurrent) ONNX export has no "h_in"/"c_in"
// among its declared inputs, so this class's behaviour for it is unchanged.
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

        // Get all output names/shapes. A recurrent model exports more than one output
        // (actions, then h_out[, c_out]) -- grabbing only output 0, as this used to,
        // silently dropped the recurrent state on every single call.
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            output_shapes.push_back(output_type.GetTensorTypeAndShapeInfo().GetShape());
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(output_name.release());
        }

        action.resize(output_shapes[0][1]);

        // Detect recurrent input/output pairs by name and set up the persisted state
        // that feeds "h_in"/"c_in" on every act() call.
        for (size_t i = 0; i < input_names.size(); ++i) {
            const std::string name(input_names[i]);
            if (name == "h_in" || name == "c_in") {
                recurrent_state_[name] = std::vector<float>(input_sizes[i], 0.0f);
            }
        }
        for (size_t i = 0; i < output_names.size(); ++i) {
            const std::string name(output_names[i]);
            if (name == "h_out") recurrent_output_index_["h_in"] = i;
            if (name == "c_out") recurrent_output_index_["c_in"] = i;
        }
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all *non-recurrent* input names are in obs -- h_in/c_in are owned
        // internally (recurrent_state_), never supplied by the caller.
        for (const auto& name : input_names) {
            if (recurrent_state_.count(name)) continue;
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto state_it = recurrent_state_.find(name_str);
            float* input_data = (state_it != recurrent_state_.end())
                ? state_it->second.data()
                : obs.at(name_str).data();
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data, input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model, requesting every declared output (not just "actions") so a
        // recurrent model's h_out/c_out are actually returned.
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

        // Copy action output data (always output 0 -- every forward() variant in
        // exporter.py returns actions first).
        auto floatarr = output_tensors[0].GetTensorMutableData<float>();
        {
            std::lock_guard<std::mutex> lock(act_mtx_);
            std::memcpy(action.data(), floatarr, output_shapes[0][1] * sizeof(float));
        }

        // Carry h_out/c_out forward into the state that feeds h_in/c_in next call.
        for (const auto& [state_name, out_idx] : recurrent_output_index_) {
            auto& state = recurrent_state_.at(state_name);
            auto* out_data = output_tensors[out_idx].GetTensorMutableData<float>();
            std::memcpy(state.data(), out_data, state.size() * sizeof(float));
        }

        return action;
    }

    // Zero the recurrent hidden/cell state, e.g. when the policy should start fresh
    // (re-entering the RL state from Passive) -- mirrors exporter.py's own
    // reset_memory() for the equivalent JIT/Torch export. No-op for a non-recurrent
    // model (recurrent_state_ is empty).
    void reset()
    {
        for (auto& [name, state] : recurrent_state_) {
            std::fill(state.begin(), state.end(), 0.0f);
        }
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
    std::vector<std::vector<int64_t>> output_shapes;

    // Recurrent state, keyed by the *input* name ("h_in"/"c_in") it feeds into on the
    // next act() call.
    std::unordered_map<std::string, std::vector<float>> recurrent_state_;
    // Input-state-name ("h_in"/"c_in") -> index of the matching output ("h_out"/"c_out")
    // in output_names/output_tensors.
    std::unordered_map<std::string, size_t> recurrent_output_index_;
};
};
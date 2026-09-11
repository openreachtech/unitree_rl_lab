// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

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

        // Recurrent (GRU) policies export an extra state input/output pair beyond the
        // plain "obs" -> "actions" mapping (e.g. "h_in" -> "h_out", as produced by
        // IsaacLab's rsl_rl _OnnxPolicyExporter.forward_gru). Any input name ending in
        // "_in" is treated as persistent recurrent state: fed from a zero-initialized
        // buffer we own instead of the obs map, and refreshed after each Run() from the
        // matching "*_out" output. Non-recurrent (plain MLP) policies have no such
        // input, so this is a no-op for them.
        for (size_t i = 0; i < input_names.size(); ++i) {
            const std::string name(input_names[i]);
            if (name.size() > 3 && name.compare(name.size() - 3, 3, "_in") == 0) {
                recurrent_state_buffers.emplace(i, std::vector<float>(input_sizes[i], 0.0f));
                recurrent_output_names.emplace(i, name.substr(0, name.size() - 3) + "_out");
            }
        }

        // Get output names/shape. Request all outputs from Run() (not just index 0) so
        // recurrent state outputs like "h_out" come back too; output 0 is always the
        // policy action.
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(output_name.release());
        }
        output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all non-recurrent input names are in obs
        for (size_t i = 0; i < input_names.size(); ++i) {
            if (recurrent_state_buffers.count(i)) continue;
            if (obs.find(input_names[i]) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(input_names[i]) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            auto recurrent_it = recurrent_state_buffers.find(i);
            float* input_data = (recurrent_it != recurrent_state_buffers.end())
                ? recurrent_it->second.data()
                : obs.at(std::string(input_names[i])).data();
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data, input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model, requesting every declared output.
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

        // Copy the action output (always index 0).
        {
            auto floatarr = output_tensors[0].GetTensorMutableData<float>();
            std::lock_guard<std::mutex> lock(act_mtx_);
            std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        }

        // Feed recurrent state outputs (e.g. "h_out") back into their buffers so the
        // next act() call continues the same GRU trajectory.
        for (size_t out_i = 0; out_i < output_names.size(); ++out_i) {
            const std::string out_name(output_names[out_i]);
            for (auto& [input_idx, expected_out_name] : recurrent_output_names) {
                if (expected_out_name == out_name) {
                    auto& buf = recurrent_state_buffers.at(input_idx);
                    std::memcpy(buf.data(), output_tensors[out_i].GetTensorMutableData<float>(), buf.size() * sizeof(float));
                }
            }
        }

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

    // input index -> owned zero-initialized state buffer (e.g. GRU hidden state)
    std::unordered_map<size_t, std::vector<float>> recurrent_state_buffers;
    // input index -> the output name that refreshes that buffer each step
    std::unordered_map<size_t, std::string> recurrent_output_names;
};
};
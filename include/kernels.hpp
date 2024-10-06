#pragma once
#include "includes.hpp"

#define CASE_N_LOOP(n)                                \
    case n:                                           \
        element_wise_loop##n(out.m_shape, call_back); \
        break;

#define ELEMENT_WISE_LOOP(n) \
    switch (n) {             \
        CASE_N_LOOP(1);      \
        CASE_N_LOOP(2);      \
        CASE_N_LOOP(3);      \
        CASE_N_LOOP(4);      \
    }

template <typename T>
class Tensor;

template <typename CallBackFn>
void element_wise_loop1(uint32_t* const shape, CallBackFn& call_back) {
    for (uint32_t i = 0; i < shape[0]; i++) {
        call_back({i});
    }
}

template <typename CallBackFn>
void element_wise_loop2(uint32_t* const shape, CallBackFn& call_back) {
    for (uint32_t i = 0; i < shape[0]; i++) {
        for (uint32_t j = 0; j < shape[1]; j++) {
            call_back({i, j});
        }
    }
}

template <typename CallBackFn>
void element_wise_loop3(uint32_t* const shape, CallBackFn& call_back) {
    for (uint32_t i = 0; i < shape[0]; i++) {
        for (uint32_t j = 0; j < shape[1]; j++) {
            for (uint32_t k = 0; k < shape[2]; k++) {
                call_back({i, j, k});
            }
        }
    }
}

template <typename CallBackFn>
void element_wise_loop4(uint32_t* const shape, CallBackFn& call_back) {
    for (uint32_t i = 0; i < shape[0]; i++) {
        for (uint32_t j = 0; j < shape[1]; j++) {
            for (uint32_t k = 0; k < shape[2]; k++) {
                for (uint32_t l = 0; l < shape[3]; l++) {
                    call_back({i, j, k, l});
                }
            }
        }
    }
}

template <typename T>
Tensor<T> get_element_wise_empty_output(const Tensor<T>& in1, const Tensor<T>& in2) {
    auto ndims = std::max(in1.m_ndims, in2.m_ndims);
    std::vector<uint32_t> out_shape(ndims, 0);

    for (int8_t i = 0; i < ndims; i++) {
        auto in1_shape = in1.m_shape[in1.m_ndims - i - 1];
        auto in2_shape = in2.m_shape[in2.m_ndims - i - 1];
        if (i < in1.m_ndims && i < in2.m_ndims) {
            LOG_IF(FATAL, (in1_shape != in2_shape) && (in1_shape != 1 && in2_shape != 1))
                << "Broadcasting is not possible between input1 with shape=" << in1_shape
                << " and input2 with shape=" << in2_shape;

            out_shape[ndims - i - 1] = std::max(in1_shape, in2_shape);
        } else if (i < in1.m_ndims) {
            out_shape[ndims - i - 1] = in1_shape;
        } else {
            out_shape[ndims - i - 1] = in2_shape;
        }
    }

    return Tensor<T>(ndims, &out_shape[0]);
}

template <typename T>
Tensor<T> add(const Tensor<T>& in1, const Tensor<T>& in2) {
    Tensor<T> out = get_element_wise_empty_output(in1, in2);

    auto call_back = [&](std::initializer_list<uint32_t> indices) {
        out[indices] = in1.broadcasted_read(indices) + in2.broadcasted_read(indices);
    };

    ELEMENT_WISE_LOOP(out.m_ndims);
    return out;
}

template <typename T>
Tensor<T> mul(const Tensor<T>& in1, const Tensor<T>& in2) {
    Tensor<T> out = get_element_wise_empty_output(in1, in2);

    auto call_back = [&](std::initializer_list<uint32_t> indices) {
        out[indices] = in1.broadcasted_read(indices) * in2.broadcasted_read(indices);
    };

    ELEMENT_WISE_LOOP(out.m_ndims);
    return out;
}
#pragma once
#include "includes.hpp"

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
Tensor<T> add(const Tensor<T>& in1, const Tensor<T>& in2) {
    auto ndims = std::max(in1.m_ndims, in2.m_ndims);
    std::vector<uint32_t> out_shape(ndims, 0);

    for (int8_t i = 0; i < ndims; i++) {
        if (i < in1.m_ndims && i < in2.m_ndims) {
            LOG_IF(FATAL, (in1.m_shape[in1.m_ndims - i - 1] != in2.m_shape[in2.m_ndims - i - 1]) &&
                              (in1.m_shape[in1.m_ndims - i - 1] != 1 && in2.m_shape[in2.m_ndims - i - 1] != 1))
                << "can't add 2 tensors with different shapes";

            out_shape[ndims - i - 1] = std::max(in1.m_shape[in1.m_ndims - i - 1], in2.m_shape[in2.m_ndims - i - 1]);
        } else if (i < in1.m_ndims) {
            out_shape[ndims - i - 1] = in1.m_shape[in1.m_ndims - i - 1];
        } else {
            out_shape[ndims - i - 1] = in2.m_shape[in2.m_ndims - i - 1];
        }
    }

    Tensor<T> out(ndims, &out_shape[0]);

    LOG(INFO) << out;

    auto call_back = [&](std::initializer_list<uint32_t> indices) mutable {
        out[indices] = in1.broadcasted_read(indices) + in2.broadcasted_read(indices);
    };

    switch (ndims) {
        case 1:
            element_wise_loop1(out.m_shape, call_back);
            break;

        case 2:
            element_wise_loop2(out.m_shape, call_back);
            break;

        case 3:
            element_wise_loop3(out.m_shape, call_back);
            break;

        case 4:
            element_wise_loop4(out.m_shape, call_back);
            break;
    }

    return out;
}
#include "kernels.hpp"

#include "tensor.hpp"

template <typename CallBackFn>
void iterate_tensor(const std::vector<uint32_t>& shape, CallBackFn& call_back) {
    int32_t ndims = shape.size();
    if (ndims <= 0) return;
    LOG_IF(FATAL, ndims >= 5) << "Looping " << ndims << " times is not yet supported";
    for (uint32_t i = 0; i < shape[0]; i++) {
        if (ndims == 1) {
            call_back({i});
            continue;
        }

        for (uint32_t j = 0; j < shape[1]; j++) {
            if (ndims == 2) {
                call_back({i, j});
                continue;
            }

            for (uint32_t k = 0; k < shape[2]; k++) {
                if (ndims == 3) {
                    call_back({i, j, k});
                    continue;
                }

                for (uint32_t l = 0; l < shape[3]; l++) {
                    call_back({i, j, k, l});
                }
            }
        }
    }
}

Tensor get_element_wise_empty_output(const Tensor& in1, const Tensor& in2) {
    int32_t in1_ndims = in1.m_shape.size();
    int32_t in2_ndims = in2.m_shape.size();
    auto ndims = std::max(in1_ndims, in2_ndims);
    std::vector<uint32_t> out_shape(ndims, 0);

    for (int8_t i = 0; i < ndims; i++) {
        auto in1_shape = in1.m_shape[in1_ndims - i - 1];
        auto in2_shape = in2.m_shape[in2_ndims - i - 1];
        if (i < in1_ndims && i < in2_ndims) {
            LOG_IF(FATAL, (in1_shape != in2_shape) && (in1_shape != 1 && in2_shape != 1))
                << "Broadcasting is not possible between input1 with shape=" << in1_shape
                << " and input2 with shape=" << in2_shape;

            out_shape[ndims - i - 1] = std::max(in1_shape, in2_shape);
        } else if (i < in1_ndims) {
            out_shape[ndims - i - 1] = in1_shape;
        } else {
            out_shape[ndims - i - 1] = in2_shape;
        }
    }

    // TODO: set datatpye
    return Tensor(out_shape);
}

Tensor add(const Tensor& in1, const Tensor& in2) {
    Tensor out = get_element_wise_empty_output(in1, in2);

    auto call_back = [&](std::initializer_list<uint32_t> indices) {
        out[indices] = in1.broadcasted_read(indices) + in2.broadcasted_read(indices);
    };

    iterate_tensor(out.m_shape, call_back);
    return out;
}

Tensor sub(const Tensor& in1, const Tensor& in2) {
    Tensor out = get_element_wise_empty_output(in1, in2);

    auto call_back = [&](std::initializer_list<uint32_t> indices) {
        out[indices] = in1.broadcasted_read(indices) - in2.broadcasted_read(indices);
    };

    iterate_tensor(out.m_shape, call_back);
    return out;
}

Tensor mul(const Tensor& in1, const Tensor& in2) {
    Tensor out = get_element_wise_empty_output(in1, in2);

    auto call_back = [&](std::initializer_list<uint32_t> indices) {
        out[indices] = in1.broadcasted_read(indices) * in2.broadcasted_read(indices);
    };

    iterate_tensor(out.m_shape, call_back);
    return out;
}

Tensor div(const Tensor& in1, const Tensor& in2) {
    Tensor out = get_element_wise_empty_output(in1, in2);

    auto call_back = [&](std::initializer_list<uint32_t> indices) {
        out[indices] = in1.broadcasted_read(indices) / in2.broadcasted_read(indices);
    };

    iterate_tensor(out.m_shape, call_back);
    return out;
}
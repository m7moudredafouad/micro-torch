#include "tensor.hpp"

namespace micro {

#define EXECUTE_OPERATION(out, in1, in2, odtype, operation)                             \
    {                                                                                   \
        auto& out_alias = out;                                                          \
        auto in1_alias = in1;                                                           \
        auto in2_alias = in2;                                                           \
        switch (odtype) {                                                               \
            case Type::UINT32:                                                          \
                out_alias.data.i32 = uint32_t(in1_alias) operation uint32_t(in2_alias); \
                break;                                                                  \
            case Type::INT32:                                                           \
                out_alias.data.i32 = int32_t(in1_alias) operation int32_t(in2_alias);   \
                break;                                                                  \
            case Type::FLOAT32:                                                         \
                out_alias.data.f32 = float(in1_alias) operation float(in2_alias);       \
                break;                                                                  \
            default:                                                                    \
                LOG(FATAL) << "Can't do element wise operation with unsupported types"; \
        }                                                                               \
    }

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

Type get_output_type(const Type& t1, const Type& t2) {
    if (t1 == Type::UNKONWN && t2 == Type::UNKONWN) {
        LOG(WARNING) << "Setting element type to Unkown";
        return Type::UNKONWN;
    }

    if (t1 == Type::UNKONWN) return t2;
    if (t2 == Type::UNKONWN) return t1;

    return std::max(t1, t2);
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

    return Tensor(out_shape, get_output_type(in1.m_dtype, in2.m_dtype));
}

Tensor get_matmul_empty_output(const Tensor& in1, const Tensor& in2) {
    LOG_IF(FATAL, in1.m_shape.size() != in2.m_shape.size()) << "Batch/Broadcast Matmul is not yet supported";
    int32_t ndims = in1.m_shape.size();
    LOG_IF(FATAL, ndims == 0) << "Matmul can't operate on tensor with shape = 0";
    std::vector<uint32_t> out_shape(ndims);

    for (int32_t i = 0; i < ndims - 2; i++) {
        out_shape[i] = std::max(in1.m_shape[i], in2.m_shape[i]);
    }

    if (ndims == 1) {
        LOG_IF(FATAL, in1.m_shape[0] != in2.m_shape[0]) << "Matmul input shapes are not compatible";
        out_shape[0] = 1;
    } else {
        LOG_IF(FATAL, in1.m_shape[ndims - 1] != in2.m_shape[ndims - 2]) << "Matmul input shapes are not compatible";
        out_shape[ndims - 2] = in1.m_shape[ndims - 2];
        out_shape[ndims - 1] = in2.m_shape[ndims - 1];
    }

    return Tensor(out_shape, get_output_type(in1.m_dtype, in2.m_dtype));
}

void Tensor::add_impl(const Tensor& in1, const Tensor& in2, Tensor& out) {
    auto call_back = [&](std::vector<uint32_t> indices) {
        EXECUTE_OPERATION(out[indices], in1.broadcasted_read(indices), in2.broadcasted_read(indices), out.m_dtype, +);
    };

    iterate_tensor(out.m_shape, call_back);
}

void Tensor::sub_impl(const Tensor& in1, const Tensor& in2, Tensor& out) {
    auto call_back = [&](std::vector<uint32_t> indices) {
        EXECUTE_OPERATION(out[indices], in1.broadcasted_read(indices), in2.broadcasted_read(indices), out.m_dtype, -);
    };

    iterate_tensor(out.m_shape, call_back);
}

void Tensor::mul_impl(const Tensor& in1, const Tensor& in2, Tensor& out) {
    auto call_back = [&](std::vector<uint32_t> indices) {
        EXECUTE_OPERATION(out[indices], in1.broadcasted_read(indices), in2.broadcasted_read(indices), out.m_dtype, *);
    };

    iterate_tensor(out.m_shape, call_back);
}

void Tensor::div_impl(const Tensor& in1, const Tensor& in2, Tensor& out) {
    auto call_back = [&](std::vector<uint32_t> indices) {
        EXECUTE_OPERATION(out[indices], in1.broadcasted_read(indices), in2.broadcasted_read(indices), out.m_dtype, /);
    };

    iterate_tensor(out.m_shape, call_back);
}

void Tensor::matmul_impl(const Tensor& in1, const Tensor& in2, Tensor& out) {
    auto call_back = [&](std::vector<uint32_t> indices) {
        int32_t ndims = indices.size();
        auto& out_value = out[indices];
        out_value = 0;

        if (ndims == 1) {
            LOG_IF(FATAL, in1.m_shape.size() != in2.m_shape.size());
            LOG_IF(FATAL, in1.m_shape.size() != out.m_shape.size());
            LOG_IF(FATAL, in1.m_shape.size() != 1);
            LOG_IF(FATAL, in1.m_shape[0] != in2.m_shape[0]);

            for (uint32_t i = 0; i < in1.m_shape[0]; i++) {
                auto v1 = in1.broadcasted_read(indices);
                auto v2 = in2.broadcasted_read(indices);

                Element tmp_value;
                EXECUTE_OPERATION(tmp_value, v1, v2, out.m_dtype, *);
                EXECUTE_OPERATION(out_value, out_value, tmp_value, out.m_dtype, +);
            }
            return;
        }

        // TODO: Add asserts
        auto tmp1_indices = indices;
        auto tmp2_indices = indices;
        for (uint32_t i = 0; i < out.m_shape[ndims - 2]; i++) {
            tmp1_indices[ndims - 1] = i;
            tmp2_indices[ndims - 2] = i;
            auto v1 = in1.broadcasted_read(tmp1_indices);
            auto v2 = in2.broadcasted_read(tmp2_indices);

            Element tmp_value;
            EXECUTE_OPERATION(tmp_value, v1, v2, out.m_dtype, *);
            EXECUTE_OPERATION(out_value, out_value, tmp_value, out.m_dtype, +);
        }
    };

    iterate_tensor(out.m_shape, call_back);
}

void Tensor::add_backward_impl(Tensor& out) {
    if (!out.m_requires_grad) return;
    LOG_IF(FATAL, !out.grad) << "Grad tensor is not initialized";
    LOG_IF(FATAL, out.m_parents.size() != 2) << "Add backward function expected 2 parents only";
    LOG_IF(FATAL, nullptr == out.m_parents[0]);
    LOG_IF(FATAL, nullptr == out.m_parents[1]);

    with_no_grad();

    auto& in1 = *out.m_parents[0];
    auto& in2 = *out.m_parents[1];

    if (!in1.grad) {
        in1.grad = std::make_shared<Tensor>(in1.m_shape);
        *(in1.grad) = 0;
    }

    if (!in2.grad) {
        in2.grad = std::make_shared<Tensor>(in2.m_shape);
        *(in2.grad) = 0;
    }

    *(in1.grad) = *(in1.grad) + *(out.grad);
    *(in2.grad) = *(in2.grad) + *(out.grad);

    with_grad();
}

void Tensor::sub_backward_impl(Tensor& out) {
    if (!out.m_requires_grad) return;

    LOG_IF(FATAL, !out.grad) << "Grad tensor is not initialized";
    LOG_IF(FATAL, out.m_parents.size() != 2) << "Subtract backward function expected 2 parents only";
    LOG_IF(FATAL, nullptr == out.m_parents[0]);
    LOG_IF(FATAL, nullptr == out.m_parents[1]);

    with_no_grad();
    auto& in1 = *out.m_parents[0];
    auto& in2 = *out.m_parents[1];

    if (!in1.grad) {
        in1.grad = std::make_shared<Tensor>(in1.m_shape);
        *(in1.grad) = 0;
    }

    if (!in2.grad) {
        in2.grad = std::make_shared<Tensor>(in2.m_shape);
        *(in2.grad) = 0;
    }

    *(in1.grad) = *(in1.grad) + *(out.grad);
    *(in2.grad) = *(in2.grad) - *(out.grad);

    with_grad();
}

void Tensor::mul_backward_impl(Tensor& out) {
    if (!out.m_requires_grad) return;

    LOG_IF(FATAL, !out.grad) << "Grad tensor is not initialized";
    LOG_IF(FATAL, out.m_parents.size() != 2) << "Subtract backward function expected 2 parents only";
    LOG_IF(FATAL, nullptr == out.m_parents[0]);
    LOG_IF(FATAL, nullptr == out.m_parents[1]);

    with_no_grad();

    auto& in1 = *out.m_parents[0];
    auto& in2 = *out.m_parents[1];

    if (!in1.grad) {
        in1.grad = std::make_shared<Tensor>(in1.m_shape);
        *(in1.grad) = 0;
    }

    if (!in2.grad) {
        in2.grad = std::make_shared<Tensor>(in2.m_shape);
        *(in2.grad) = 0;
    }

    if (&out != &in1) {
        *(in1.grad) = *(in1.grad) + (*(out.grad) * in2);
    }

    if (&out != &in2) {
        *(in2.grad) = *(in2.grad) + (*(out.grad) * in1);
    }

    with_grad();
}

void Tensor::div_backward_impl(Tensor& out) {
    (void)out;
    LOG(FATAL) << "Not yet implemented";
}

void Tensor::matmul_backward_impl(Tensor& out) {
    (void)out;
    LOG(FATAL) << "Not yet implemented";
}

};  // namespace micro
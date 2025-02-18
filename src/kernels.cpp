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

#define EXECUTE_INLINE_OPERATION(out, in, odtype, operation)                           \
    {                                                                                  \
        auto& out_alias = out;                                                         \
        auto in_alias = in;                                                            \
        switch (odtype) {                                                              \
            case Type::UINT32:                                                         \
                out_alias.data.i32 = uint32_t(out_alias) operation uint32_t(in_alias); \
                break;                                                                 \
            case Type::INT32:                                                          \
                out_alias.data.i32 = int32_t(out_alias) operation int32_t(in_alias);   \
                break;                                                                 \
            case Type::FLOAT32:                                                        \
                out_alias.data.f32 = float(out_alias) operation float(in_alias);       \
                break;                                                                 \
            default:                                                                   \
                LOG(FATAL) << "Can't do inline operation with unsupported types";      \
        }                                                                              \
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

void align_gradient_with_tensor(const Tensor& tensor, Tensor& gradient) {
    auto t_shape = tensor.m_shape;
    auto g_shape = gradient.m_shape;

    LOG_IF(FATAL, t_shape.size() > g_shape.size()) << "for now tensor shape can have less dims than its gradient";

    size_t dims = t_shape.size();

    with_no_grad();

    for (size_t i = 0; i < dims; i++) {
        if (t_shape[i] == g_shape[i]) continue;

        if (t_shape[i] == 1) {
            gradient = gradient.sum(i, true);
            continue;
        }

        LOG(FATAL) << "Tensor and its gradient have incompatible shapes";
    }

    for (size_t i = g_shape.size() - 1; i >= dims; i--) {
        gradient = gradient.sum(i);
    }

    with_grad();
}

void Tensor::add_forward_impl(const Tensor& in1, const Tensor& in2, Tensor& out) {
    auto call_back = [&](std::vector<uint32_t> indices) {
        EXECUTE_OPERATION(out[indices], in1.broadcasted_read(indices), in2.broadcasted_read(indices), out.m_dtype, +);
    };

    iterate_tensor(out.m_shape, call_back);
}

void Tensor::sub_forward_impl(const Tensor& in1, const Tensor& in2, Tensor& out) {
    auto call_back = [&](std::vector<uint32_t> indices) {
        EXECUTE_OPERATION(out[indices], in1.broadcasted_read(indices), in2.broadcasted_read(indices), out.m_dtype, -);
    };

    iterate_tensor(out.m_shape, call_back);
}

void Tensor::mul_forward_impl(const Tensor& in1, const Tensor& in2, Tensor& out) {
    auto call_back = [&](std::vector<uint32_t> indices) {
        EXECUTE_OPERATION(out[indices], in1.broadcasted_read(indices), in2.broadcasted_read(indices), out.m_dtype, *);
    };

    iterate_tensor(out.m_shape, call_back);
}

void Tensor::div_forward_impl(const Tensor& in1, const Tensor& in2, Tensor& out) {
    auto call_back = [&](std::vector<uint32_t> indices) {
        EXECUTE_OPERATION(out[indices], in1.broadcasted_read(indices), in2.broadcasted_read(indices), out.m_dtype, /);
    };

    iterate_tensor(out.m_shape, call_back);
}

void Tensor::matmul_forward_impl(const Tensor& in1, const Tensor& in2, Tensor& out) {
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
        uint32_t inner = in1.m_shape[ndims - 1];
        for (uint32_t i = 0; i < inner; i++) {
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

void Tensor::sum_forward_impl(const Tensor& in, const uint32_t dim, Tensor& out) {
    auto in_shape = in.m_shape;
    auto out_shape = out.m_shape;

    LOG_IF(FATAL, in_shape.size() != out_shape.size()) << "Shapes are not compatible";

    uint32_t dim_size = in_shape[dim];

    auto call_back = [&](std::vector<uint32_t> indices) {
        auto tmp_indices = indices;

        Element sum(0);

        for (uint32_t i = 0; i < dim_size; i++) {
            tmp_indices[dim] = i;
            EXECUTE_INLINE_OPERATION(sum, in[tmp_indices], out.m_dtype, +);
        }

        out[indices] = sum;
    };

    iterate_tensor(out.m_shape, call_back);
}

void Tensor::add_backward_impl(Tensor& out) {
    if (!out.m_requires_grad) return;
    LOG_IF(FATAL, !out.m_saved_context->grad()) << "Grad tensor is not initialized";

    auto parents = out.m_saved_context->get_saved_variables();
    LOG_IF(FATAL, parents.size() != 2) << "Add backward function expected 2 parents only";

    with_no_grad();

    auto& in1 = parents[0];
    auto& in2 = parents[1];

    auto& in1_grad = in1.m_saved_context->grad();
    auto& in2_grad = in2.m_saved_context->grad();
    auto& out_grad = out.m_saved_context->grad();

    if (!in1_grad) {
        in1_grad = std::make_shared<Tensor>(in1.m_shape);
        *(in1_grad) = 0;
    }

    if (!in2_grad) {
        in2_grad = std::make_shared<Tensor>(in2.m_shape);
        *(in2_grad) = 0;
    }

    *(in1_grad) = *(in1_grad) + *(out_grad);
    *(in2_grad) = *(in2_grad) + *(out_grad);

    align_gradient_with_tensor(in1, *(in1_grad));
    align_gradient_with_tensor(in2, *(in2_grad));

    with_grad();
}

void Tensor::sub_backward_impl(Tensor& out) {
    if (!out.m_requires_grad) return;

    LOG_IF(FATAL, !out.m_saved_context->grad()) << "Grad tensor is not initialized";

    auto parents = out.m_saved_context->get_saved_variables();

    LOG_IF(FATAL, parents.size() != 2) << "Subtract backward function expected 2 parents only";

    with_no_grad();

    auto& in1 = parents[0];
    auto& in2 = parents[1];

    auto& in1_grad = in1.m_saved_context->grad();
    auto& in2_grad = in2.m_saved_context->grad();
    auto& out_grad = out.m_saved_context->grad();

    if (!in1_grad) {
        in1_grad = std::make_shared<Tensor>(in1.m_shape);
        *(in1_grad) = 0;
    }

    if (!in2_grad) {
        in2_grad = std::make_shared<Tensor>(in2.m_shape);
        *(in2_grad) = 0;
    }

    *(in1_grad) = *(in1_grad) + *(out_grad);
    *(in2_grad) = *(in2_grad) - *(out_grad);

    align_gradient_with_tensor(in1, *(in1_grad));
    align_gradient_with_tensor(in2, *(in2_grad));

    with_grad();
}

void Tensor::mul_backward_impl(Tensor& out) {
    if (!out.m_requires_grad) return;

    LOG_IF(FATAL, !out.m_saved_context->grad()) << "Grad tensor is not initialized";

    auto parents = out.m_saved_context->get_saved_variables();

    LOG_IF(FATAL, parents.size() != 2) << "Subtract backward function expected 2 parents only";

    with_no_grad();

    auto& in1 = parents[0];
    auto& in2 = parents[1];

    auto& in1_grad = in1.m_saved_context->grad();
    auto& in2_grad = in2.m_saved_context->grad();
    auto& out_grad = out.m_saved_context->grad();

    if (!in1_grad) {
        in1_grad = std::make_shared<Tensor>(in1.m_shape);
        *(in1_grad) = 0;
    }

    if (!in2_grad) {
        in2_grad = std::make_shared<Tensor>(in2.m_shape);
        *(in2_grad) = 0;
    }

    if (out.m_saved_context != in1.m_saved_context) {
        *(in1_grad) = *(in1_grad) + (*(out_grad)*in2);
    }

    if (out.m_saved_context != in2.m_saved_context) {
        *(in2_grad) = *(in2_grad) + (*(out_grad)*in1);
    }

    align_gradient_with_tensor(in1, *(in1_grad));
    align_gradient_with_tensor(in2, *(in2_grad));

    with_grad();
}

void Tensor::div_backward_impl(Tensor& out) {
    (void)out;
    LOG(FATAL) << "Not yet implemented";
}

void Tensor::matmul_backward_impl(Tensor& out) {
    if (!out.m_requires_grad) return;

    LOG_IF(FATAL, !out.m_saved_context->grad()) << "Grad tensor is not initialized";

    auto parents = out.m_saved_context->get_saved_variables();

    LOG_IF(FATAL, parents.size() != 2) << "Subtract backward function expected 2 parents only";

    with_no_grad();

    auto& in1 = parents[0];
    auto& in2 = parents[1];

    auto& in1_grad = in1.m_saved_context->grad();
    auto& in2_grad = in2.m_saved_context->grad();
    auto& out_grad = out.m_saved_context->grad();

    if (!in1_grad) {
        in1_grad = std::make_shared<Tensor>(in1.m_shape);
        *(in1_grad) = 0;
    }

    if (!in2_grad) {
        in2_grad = std::make_shared<Tensor>(in2.m_shape);
        *(in2_grad) = 0;
    }

    if (out.m_saved_context != in1.m_saved_context) {
        *(in1_grad) = *(in1_grad) + out_grad->mm(in2.transpose());
    }

    if (out.m_saved_context != in2.m_saved_context) {
        *(in2_grad) = *(in2_grad) + in1.transpose().mm(*(out_grad));
    }

    // align_gradient_with_tensor(in1, *(in1_grad));
    // align_gradient_with_tensor(in2, *(in2_grad));

    with_grad();
}

void Tensor::sum_backward_impl(Tensor& out) {
    if (!out.m_requires_grad) return;

    LOG_IF(FATAL, !out.m_saved_context->grad()) << "Grad tensor is not initialized";

    auto parents = out.m_saved_context->get_saved_variables();

    LOG_IF(FATAL, parents.size() != 1) << "Sum backward function expected  only 1 parent";

    with_no_grad();

    auto& in = parents[0];

    auto& in_grad = in.m_saved_context->grad();
    auto& out_grad = out.m_saved_context->grad();

    if (!in_grad) {
        in_grad = std::make_shared<Tensor>(in.m_shape);
        *(in_grad) = 0;
    }

    *(in_grad) = *(in_grad) + (*out_grad);

    with_grad();
}

};  // namespace micro
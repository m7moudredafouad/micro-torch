
#include "utilities.hpp"

#include "tensor.hpp"

std::ostream& operator<<(std::ostream& os, Tensor& t) {
    os << "Tensor(";

    int32_t ndims = t.m_shape.size();
    os << "ndims=" << ndims << ", offset=" << t.m_offset;

    os << ", shape=[";
    for (uint8_t i = 0; i < ndims - 1; i++) {
        os << t.m_shape[i] << ", ";
    }
    os << t.m_shape[ndims - 1] << "]";

    os << ", stride=[";
    for (uint8_t i = 0; i < ndims - 1; i++) {
        os << t.m_stride[i] << ", ";
    }
    os << t.m_stride[ndims - 1] << "]";

    os << "\nvalues=[";
    for (uint32_t i = 0; ndims >= 1 && i < t.m_shape[0]; i++) {
        if (ndims == 1) {
            os << t[{i}];
            if (i < t.m_shape[0] - 1) {
                os << ", ";
            }
            continue;
        }
        os << "[";
        for (uint32_t j = 0; ndims >= 2 && j < t.m_shape[1]; j++) {
            if (ndims == 2) {
                os << t[{i, j}];
                if (j < t.m_shape[1] - 1) {
                    os << ", ";
                }
                continue;
            }
            os << "[";
            for (uint32_t k = 0; ndims >= 3 && k < t.m_shape[2]; k++) {
                if (ndims == 3) {
                    os << t[{i, j, k}];
                    if (k < t.m_shape[2] - 1) {
                        os << ", ";
                    }
                    continue;
                }
                os << "[";
                for (uint32_t l = 0; ndims >= 4 && l < t.m_shape[3]; l++) {
                    os << t[{i, j, k, l}];
                    if (l < t.m_shape[3] - 1) {
                        os << ", ";
                    }
                }

                os << "]";
                if (k < t.m_shape[2] - 1) {
                    os << ", ";
                }
            }
            os << "]";
            if (j < t.m_shape[1] - 1) {
                os << ", ";
            }
        }
        os << "]";
        if (i < t.m_shape[0] - 1) {
            os << ", ";
        }
    }
    os << "]";

    os << ")";
    return os;
}

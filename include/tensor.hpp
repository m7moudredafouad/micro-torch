#pragma once
#include "includes.hpp"
#include "storage.hpp"

class Tensor {
   public:
    Tensor() = default;

    Tensor(const std::vector<uint32_t>& shape) : m_ndims(shape.size()) {
        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];
        std::copy(&shape[0], &shape[m_ndims], m_shape);
        SetDefaultStrides();

        m_storage = Storage(Size());
    }

    Tensor(const std::vector<uint32_t>& shape, const std::vector<uint32_t>& stride) : m_ndims(shape.size()) {
        LOG_IF(FATAL, shape.size() != stride.size());

        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];

        std::copy(&shape[0], &shape[m_ndims], m_shape);
        std::copy(&stride[0], &stride[m_ndims], m_shape);
        m_storage = Storage(Size());
    }

    uint32_t Size() {
        LOG_IF(FATAL, !m_shape);
        uint32_t size = 1;
        for (uint8_t i = 0; i < m_ndims; i++) {
            size *= m_shape[i];
        }

        return size;
    }

    void SetDefaultStrides() {
        LOG_IF(FATAL, !m_stride);
        LOG_IF(FATAL, !m_shape);

        m_stride[m_ndims - 1] = 1;

        for (int8_t i = (int8_t)m_ndims - 2; i >= 0; i--) {
            m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
        }
    }

    uint32_t operator[](uint32_t idx) const { return m_storage.read<uint32_t>(m_stride[m_ndims - 1] * idx); }

    uint32_t& operator[](uint32_t idx) { return m_storage.read<uint32_t>(m_stride[m_ndims - 1] * idx); }

    ~Tensor() {
        delete[] m_shape;
        delete[] m_stride;
    }

   private:
    int64_t m_offset = 0;
    uint8_t m_ndims = 0;

    uint32_t* m_shape = nullptr;
    uint32_t* m_stride = nullptr;
    Storage m_storage;
};
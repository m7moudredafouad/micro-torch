#pragma once
#include <cassert>

#include "includes.hpp"
#include "kernels.hpp"
#include "storage.hpp"
#include "utilities.hpp"

class Tensor {
   public:
    struct Element {
        enum class Type : uint8_t { UINT32 = 0, INT32, FLOAT32, RESERVED_MAX_VALUE };

        union Data {
            int32_t i32 = 0;
            float f32;
        };

        Type dtype{Type::UINT32};
        Data data;

        Element() = default;

        template <typename T>
        Element(T value) {
            if constexpr (std::is_same_v<T, int32_t>) {
                data.i32 = value;
                dtype = Type::INT32;
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                data.i32 = value;
                dtype = Type::UINT32;
            } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                data.f32 = float(value);
                dtype = Type::FLOAT32;
            } else {
                LOG(FATAL) << "Element Type is not supported";
            }
        }

        template <typename T>
        Element(T value, Type type) {
            dtype = type;
            switch (dtype) {
                case Type::UINT32:
                    data.i32 = uint32_t(value);
                    break;
                case Type::INT32:
                    data.i32 = int32_t(value);
                    break;
                case Type::FLOAT32:
                    data.f32 = float(value);
                    break;
                default:
                    LOG(FATAL) << "Element Type is not supported";
            }
        }

#define READ_DATA(element) ((element).dtype == Type::FLOAT32 ? (element).data.f32 : (element).data.i32)

        operator float() const { return (float)READ_DATA(*this); }

        operator int32_t() const { return (int32_t)READ_DATA(*this); }

        operator uint32_t() const { return (uint32_t)READ_DATA(*this); }

        friend std::ostream& operator<<(std::ostream& os, Element& element) {
            os << READ_DATA(element);
            return os;
        }

#undef READ_DATA

#define EXECUTE_OPERATION(out, first, second, operation)                     \
    {                                                                        \
        out.dtype = std::max(dtype, other.dtype);                            \
        if (out.dtype >= Type::RESERVED_MAX_VALUE) {                         \
            out.dtype = std::min(dtype, other.dtype);                        \
        }                                                                    \
        switch ((out).dtype) {                                               \
            case Type::UINT32:                                               \
                (out).data.i32 = uint32_t(first) operation uint32_t(second); \
                break;                                                       \
            case Type::INT32:                                                \
                (out).data.i32 = int32_t(first) operation int32_t(second);   \
                break;                                                       \
            case Type::FLOAT32:                                              \
                (out).data.f32 = float(first) operation float(second);       \
                break;                                                       \
            default:                                                         \
                (out).dtype = Type::UINT32;                                  \
                (out).data.i32 = 0;                                          \
                LOG(WARNING) << "Element Type is not supported";             \
        }                                                                    \
    }

        Element operator+(const Element& other) {
            Element out;
            EXECUTE_OPERATION(out, (*this), other, +);
            return out;
        }

        Element operator-(const Element& other) {
            Element out;
            EXECUTE_OPERATION(out, (*this), other, -);
            return out;
        }

        Element operator*(const Element& other) {
            Element out;
            EXECUTE_OPERATION(out, (*this), other, *);
            return out;
        }

        Element operator/(const Element& other) {
            Element out;
            EXECUTE_OPERATION(out, (*this), other, /);
            return out;
        }

#undef EXECUTE_OPERATION
    };

   public:
    Tensor(uint32_t ndims, uint32_t* const shape) : m_ndims(ndims) {
        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];
        std::copy(&shape[0], &shape[m_ndims], m_shape);
        SetDefaultStrides();

        m_storage = Storage(NumberOfBytes());
    }

    Tensor(const std::vector<uint32_t>& shape) : m_ndims(shape.size()) {
        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];
        std::copy(&shape[0], &shape[m_ndims], m_shape);
        SetDefaultStrides();

        m_storage = Storage(NumberOfBytes());
    }

    Tensor(const std::vector<uint32_t>& shape, const std::vector<uint32_t>& stride) : m_ndims(shape.size()) {
        LOG_IF(FATAL, shape.size() != stride.size());

        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];

        std::copy(&shape[0], &shape[m_ndims], m_shape);
        std::copy(&stride[0], &stride[m_ndims], m_shape);
        m_storage = Storage(NumberOfBytes());
    }

    Tensor(const Tensor& other) {
        m_offset = other.m_offset;
        m_ndims = other.m_ndims;
        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];
        std::copy(&other.m_shape[0], &other.m_shape[m_ndims], m_shape);
        std::copy(&other.m_stride[0], &other.m_stride[m_ndims], m_stride);
        m_storage = other.m_storage;
        dtype = other.dtype;
    }

    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;

        m_offset = other.m_offset;
        m_ndims = other.m_ndims;
        m_shape = new uint32_t[m_ndims];
        m_stride = new uint32_t[m_ndims];
        std::copy(&other.m_shape[0], &other.m_shape[m_ndims], m_shape);
        std::copy(&other.m_stride[0], &other.m_stride[m_ndims], m_stride);
        m_storage = other.m_storage;
        dtype = other.dtype;

        return *this;
    }

    ~Tensor() {
        if (m_shape) {
            delete[] m_shape;
        }

        if (m_stride) {
            delete[] m_stride;
        }
    }

    uint32_t Size() const {
        LOG_IF(FATAL, !m_shape);
        uint32_t size = 1;
        for (int8_t i = 0; i < m_ndims; i++) {
            size *= m_shape[i];
        }

        return size;
    }

    uint32_t NumberOfBytes() const { return Size() * sizeof(Element); }

    void SetDefaultStrides() {
        LOG_IF(FATAL, !m_stride);
        LOG_IF(FATAL, !m_shape);

        m_stride[m_ndims - 1] = 1;

        for (int8_t i = (int8_t)m_ndims - 2; i >= 0; i--) {
            m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
        }
    }

    Element operator[](std::initializer_list<uint32_t> indices) const {
        return const_cast<Tensor*>(this)->operator[](indices);
    }

    Element& operator[](std::initializer_list<uint32_t> indices) {
        LOG_IF(FATAL, (int8_t)indices.size() != m_ndims)
            << "Indices size=" << indices.size() << " don't match the full_shape=" << int(m_ndims);
        uint32_t offset = 0, i = 0;

        for (auto idx : indices) {
            LOG_IF(FATAL, idx >= m_shape[i])
                << "index is out of range, full_shape= " << m_shape[i] << " and index=" << idx;
            offset += idx * m_stride[i];
            i++;
        }

        return this->operator[](m_offset + offset);
    }

    Element& at(std::initializer_list<uint32_t> indices) { return this->operator[](indices); }

    Tensor operator+(const Tensor& other) { return add(*this, other); }

    Tensor operator*(const Tensor& other) { return mul(*this, other); }

    template <typename T>
    Tensor operator+(T value) {
        Tensor tensor(1, (uint32_t[1]){1});
        tensor[0] = Element(value, dtype);
        return this->operator+(tensor);
    }

    template <typename T>
    Tensor operator*(T value) {
        Tensor tensor(1, (uint32_t[1]){1});
        tensor[0] = Element(value, dtype);
        return this->operator*(tensor);
    }

    template <typename T>
    void operator=(T value) {
        for (uint32_t i = 0; i < Size(); i++) {
            this->operator[](i) = Element(value, dtype);
        }
    }

    friend std::ostream& operator<<(std::ostream& os, Tensor& t);
    friend Tensor get_element_wise_empty_output(const Tensor& in1, const Tensor& in2);
    friend Tensor add(const Tensor& in1, const Tensor& in2);
    friend Tensor mul(const Tensor& in1, const Tensor& in2);

   private:
    Element operator[](uint32_t offset) const { return const_cast<Tensor*>(this)->operator[](offset); }

    Element& operator[](uint32_t offset) {
        LOG_IF(FATAL, offset >= Size()) << "index out of range";
        return *reinterpret_cast<Element*>(m_storage.at(offset * sizeof(Element)));
    }

    Element broadcasted_read(std::initializer_list<uint32_t> indices) const {
        int8_t ndims = indices.size();
        uint32_t offset = 0, i = 0, j = 0;

        for (auto idx : indices) {
            if ((ndims - i) > m_ndims) {
                i++;
                continue;
            }

            if (idx < m_shape[j]) {
                offset += idx * m_stride[j];
            } else {
                LOG_IF(FATAL, m_shape[j] != 1) << "Broadcasting failed";
            }

            j++;
        }

        return this->operator[](m_offset + offset);
    }

   private:
    Tensor::Element::Type dtype = Tensor::Element::Type::FLOAT32;
    int64_t m_offset = 0;
    int8_t m_ndims = 0;

    uint32_t* m_shape = nullptr;
    uint32_t* m_stride = nullptr;
    Storage m_storage;
};
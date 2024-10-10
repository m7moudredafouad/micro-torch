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
    Tensor(std::vector<uint32_t> shape) : m_shape(std::move(shape)) {
        SetDefaultStrides();
        m_storage = Storage(NumberOfBytes());
    }

    Tensor(std::vector<uint32_t> shape, std::vector<uint32_t>& stride)
        : m_shape(std::move(shape)), m_stride(std::move(stride)) {
        LOG_IF(FATAL, shape.size() != stride.size());

        m_storage = Storage(NumberOfBytes());
    }

    uint32_t Size() const {
        LOG_IF(FATAL, m_shape.size() == 0);
        uint32_t size = 1;
        for (size_t i = 0; i < m_shape.size(); i++) {
            size *= m_shape[i];
        }

        return size;
    }

    uint32_t NumberOfBytes() const { return Size() * sizeof(Element); }

    void SetDefaultStrides() {
        LOG_IF(FATAL, m_shape.size() == 0);

        m_stride.resize(m_shape.size());
        m_stride[m_shape.size() - 1] = 1;

        for (int8_t i = (int8_t)m_shape.size() - 2; i >= 0; i--) {
            m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
        }
    }

    Element operator[](std::initializer_list<uint32_t> indices) const {
        return const_cast<Tensor*>(this)->operator[](indices);
    }

    Element& operator[](std::initializer_list<uint32_t> indices) {
        LOG_IF(FATAL, indices.size() != m_shape.size())
            << "Indices size=" << indices.size() << " don't match the full_shape=" << m_shape.size();
        uint32_t offset = 0, i = 0;

        for (auto idx : indices) {
            LOG_IF(FATAL, idx >= m_shape[i])
                << "index is out of range, full_shape= " << m_shape[i] << " and index=" << idx;
            offset += idx * m_stride[i];
            i++;
        }

        return this->operator[](offset);
    }

    Element& at(std::initializer_list<uint32_t> indices) { return this->operator[](indices); }

    Tensor operator+(const Tensor& other) {
        Tensor out = get_element_wise_empty_output(*this, other);
        add(*this, other, out);
        return out;
    }

    Tensor operator-(const Tensor& other) {
        Tensor out = get_element_wise_empty_output(*this, other);
        sub(*this, other, out);
        return out;
    }

    Tensor operator*(const Tensor& other) {
        Tensor out = get_element_wise_empty_output(*this, other);
        mul(*this, other, out);
        return out;
    }

    Tensor operator/(const Tensor& other) {
        Tensor out = get_element_wise_empty_output(*this, other);
        div(*this, other, out);
        return out;
    }

    template <typename T>
    Tensor operator+(T value) {
        Tensor tensor({1});
        tensor[0] = Element(value, dtype);
        return this->operator+(tensor);
    }

    template <typename T>
    void operator+=(T value) {
        Tensor tensor({1});
        tensor[0] = Element(value, dtype);
        add(*this, tensor, *this);
    }

    template <typename T>
    Tensor operator-(T value) {
        Tensor tensor({1});
        tensor[0] = Element(value, dtype);
        return this->operator-(tensor);
    }

    template <typename T>
    void operator-=(T value) {
        Tensor tensor({1});
        tensor[0] = Element(value, dtype);
        sub(*this, tensor, *this);
    }

    template <typename T>
    Tensor operator*(T value) {
        Tensor tensor({1});
        tensor[0] = Element(value, dtype);
        return this->operator*(tensor);
    }

    template <typename T>
    void operator*=(T value) {
        Tensor tensor({1});
        tensor[0] = Element(value, dtype);
        mul(*this, tensor, *this);
    }

    template <typename T>
    Tensor operator/(T value) {
        Tensor tensor({1});
        tensor[0] = Element(value, dtype);
        return this->operator/(tensor);
    }

    template <typename T>
    void operator/=(T value) {
        Tensor tensor({1});
        tensor[0] = Element(value, dtype);
        div(*this, tensor, *this);
    }

    template <typename T>
    void operator=(T value) {
        for (uint32_t i = 0; i < Size(); i++) {
            this->operator[](i) = Element(value, dtype);
        }
    }

    friend std::ostream& operator<<(std::ostream& os, Tensor& t);
    friend Tensor get_element_wise_empty_output(const Tensor& in1, const Tensor& in2);
    friend void add(const Tensor& in1, const Tensor& in2, Tensor& out);
    friend void sub(const Tensor& in1, const Tensor& in2, Tensor& out);
    friend void mul(const Tensor& in1, const Tensor& in2, Tensor& out);
    friend void div(const Tensor& in1, const Tensor& in2, Tensor& out);

   private:
    Element operator[](uint32_t offset) const { return const_cast<Tensor*>(this)->operator[](offset); }

    Element& operator[](uint32_t offset) {
        LOG_IF(FATAL, offset >= Size()) << "index out of range";
        return *reinterpret_cast<Element*>(m_storage.at((m_offset + offset) * sizeof(Element)));
    }

    Element broadcasted_read(std::initializer_list<uint32_t> indices) const {
        int32_t nindecies = indices.size();
        int32_t ndims = m_shape.size();
        int32_t i = 0, j = 0;
        uint32_t offset = 0;

        for (auto idx : indices) {
            if ((nindecies - i) > ndims) {
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

        return this->operator[](offset);
    }

   private:
    Tensor::Element::Type dtype = Tensor::Element::Type::FLOAT32;
    int64_t m_offset = 0;

    std::vector<uint32_t> m_shape, m_stride;
    Storage m_storage;
};
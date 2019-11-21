﻿#include "testPch.h"
// LotusRT
#include "core/framework/allocatormgr.h"
// #include "core/session/inference_session.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"

#include "protobufHelpers.h"
#include "onnx/onnx-ml.pb.h"
#include <gtest/gtest.h>
#include <fstream>

#include "winrt/Windows.Storage.Streams.h"

#pragma warning(disable : 4244)

using namespace winrt::Windows::Storage::Streams;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Foundation::Collections;

// Copy and pasted from LOTUS as is.    temporary code to load tensors from protobufs
int FdOpen(const std::string& name)
{
    int fd = -1;
#ifdef _WIN32
    _sopen_s(&fd, name.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
    fd = open(name.c_str(), O_RDONLY);
#endif
    return fd;
};

// Copy and pasted from LOTUS as is.    temporary code to load tensors from protobufs
void FdClose(int fd)
{
    if (fd >= 0)
    {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
    }
}

// Copy and pasted from LOTUS as is.    temporary code to load tensors from protobufs
bool LoadTensorFromPb(onnx::TensorProto& tensor, std::wstring filePath)
{
    // setup a string converter
    using convert_type = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_type, wchar_t> converter;

    // use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
    std::string file = converter.to_bytes(filePath.c_str());

    std::ifstream stream(file, std::ios::binary | std::ios::ate);
    std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (stream.read(buffer.data(), size))
    {
        return tensor.ParseFromArray(buffer.data(), static_cast<int>(size));
    }
    else
    {
        return false;
    }

}

template <typename DataType>
std::vector<DataType> GetTensorDataFromTensorProto(onnx::TensorProto tensorProto, int elementCount)
{
    if (tensorProto.has_raw_data())
    {
        std::vector<DataType> tensorData;
        auto& values = tensorProto.raw_data();
        EXPECT_EQ(elementCount, values.size() / sizeof(DataType)) << L"TensorProto elementcount should match raw data buffer size in elements.";

        tensorData = std::vector<DataType>(elementCount);
        memcpy(tensorData.data(), values.data(), values.size());
        return tensorData;
    }
    else
    {
        return std::vector<DataType>(std::begin(tensorProto.float_data()), std::end(tensorProto.float_data()));
    }
}

static
std::vector<winrt::hstring> GetTensorStringDataFromTensorProto(
    onnx::TensorProto tensorProto,
    int elementCount)
{
    EXPECT_EQ(tensorProto.string_data_size(), elementCount);
    auto& values = tensorProto.string_data();
    auto returnVector = std::vector<winrt::hstring>(elementCount);
    std::transform(std::begin(values), std::end(values), std::begin(returnVector),
        [](auto& value) { return winrt::to_hstring(value); });
    return returnVector;
}

ITensor ProtobufHelpers::LoadTensorFromProtobufFile(
    const std::wstring& filePath,
    bool isFp16)
{
    // load from the file path into the onnx format
    onnx::TensorProto tensorProto;
    if (LoadTensorFromPb(tensorProto, filePath))
    {
        std::vector<int64_t> tensorShape = std::vector<int64_t>(tensorProto.dims().begin(), tensorProto.dims().end());
        int64_t initialValue = 1;
        auto elementCount = std::accumulate(tensorShape.begin(), tensorShape.end(), initialValue, std::multiplies<int64_t>());

        if (!tensorProto.has_data_type())
        {
            std::cerr << "WARNING: Loading unknown TensorProto datatype.\n";
        }
        if (isFp16)
        {
            return TensorFloat16Bit::CreateFromIterable(tensorShape, GetTensorDataFromTensorProto<float>(tensorProto, elementCount));
        }
        switch (tensorProto.data_type())
        {
        case(onnx::TensorProto::DataType::TensorProto_DataType_FLOAT):
            return TensorFloat::CreateFromIterable(tensorShape, GetTensorDataFromTensorProto<float>(tensorProto, elementCount));
        case(onnx::TensorProto::DataType::TensorProto_DataType_INT32):
            return TensorInt32Bit::CreateFromIterable(tensorShape, GetTensorDataFromTensorProto<int32_t>(tensorProto, elementCount));
        case(onnx::TensorProto::DataType::TensorProto_DataType_INT64):
            return TensorInt64Bit::CreateFromIterable(tensorShape, GetTensorDataFromTensorProto<int64_t>(tensorProto, elementCount));
        case(onnx::TensorProto::DataType::TensorProto_DataType_STRING):
            return TensorString::CreateFromIterable(tensorShape, GetTensorStringDataFromTensorProto(tensorProto, elementCount));
        default:
            ADD_FAILURE() << L"Tensor type for creating tensor from protobuf file not supported.";
            break;
        }
    }
    return nullptr;
}

TensorFloat16Bit ProtobufHelpers::LoadTensorFloat16FromProtobufFile(
    const std::wstring& filePath)
{
    // load from the file path into the onnx format
    onnx::TensorProto tensorProto;
    if (LoadTensorFromPb(tensorProto, filePath))
    {
        if (tensorProto.has_data_type())
        {
            EXPECT_EQ(onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16, tensorProto.data_type());
        }
        else
        {
            std::cerr << "Loading unknown TensorProto datatype as TensorFloat16Bit.\n";
        }

        auto shape = winrt::single_threaded_vector<int64_t>(std::vector<int64_t>(tensorProto.dims().begin(), tensorProto.dims().end()));
        TensorFloat16Bit singleTensorValue = TensorFloat16Bit::Create(shape.GetView());

        uint16_t* data;
        winrt::com_ptr<ITensorNative> spTensorValueNative;
        singleTensorValue.as(spTensorValueNative);
        uint32_t sizeInBytes;
        spTensorValueNative->GetBuffer(reinterpret_cast<BYTE**>(&data), &sizeInBytes);

        EXPECT_TRUE(tensorProto.has_raw_data()) << L"Float16 tensor proto buffers are expected to contain raw data.";

        auto& raw_data = tensorProto.raw_data();
        auto buff = raw_data.c_str();
        const size_t type_size = sizeof(uint16_t);

        memcpy((void*)data, (void*)buff, raw_data.size() * sizeof(char));

        return singleTensorValue;
    }
    return nullptr;
}

winrt::Windows::AI::MachineLearning::LearningModel ProtobufHelpers::CreateModel(
    winrt::Windows::AI::MachineLearning::TensorKind kind,
    const std::vector<int64_t>& shape,
    uint32_t num_elements)
{
    onnx::ModelProto model;

    // Set opset import
    auto opsetimportproto = model.add_opset_import();
    opsetimportproto->set_version(7);

    onnx::GraphProto& graph = *model.mutable_graph();

    uint32_t begin = 0;
    uint32_t end = num_elements - 1;
    for (uint32_t i = begin; i <= end; i++)
    {
        onnx::NodeProto& node = *graph.add_node();
        node.set_op_type("Identity");
        if (i == begin && i == end)
        {
            node.add_input("input");
            node.add_output("output");
        }
        else if (i == begin)
        {
            node.add_input("input");
            node.add_output("output" + std::to_string(i));

        }
        else if (i == end)
        {
            node.add_input("output" + std::to_string(i-1));
            node.add_output("output");
        }
        else
        {
            node.add_input("output" + std::to_string(i-1));
            node.add_output("output" + std::to_string(i));
        }
    }

    onnx::TensorProto_DataType dataType;
    switch (kind)
    {
    case TensorKind::Float: dataType = onnx::TensorProto_DataType_FLOAT; break;
    case TensorKind::UInt8: dataType = onnx::TensorProto_DataType_UINT8; break;
    case TensorKind::Int8: dataType = onnx::TensorProto_DataType_INT8; break;
    case TensorKind::UInt16: dataType = onnx::TensorProto_DataType_UINT16; break;
    case TensorKind::Int16: dataType = onnx::TensorProto_DataType_INT16; break;
    case TensorKind::Int32: dataType = onnx::TensorProto_DataType_INT32; break;
    case TensorKind::Int64: dataType = onnx::TensorProto_DataType_INT64; break;
    case TensorKind::String: dataType = onnx::TensorProto_DataType_STRING; break;
    case TensorKind::Boolean: dataType = onnx::TensorProto_DataType_BOOL; break;
    case TensorKind::Float16: dataType = onnx::TensorProto_DataType_FLOAT16; break;
    case TensorKind::Double: dataType = onnx::TensorProto_DataType_DOUBLE; break;
    case TensorKind::UInt32: dataType = onnx::TensorProto_DataType_UINT32; break;
    case TensorKind::UInt64: dataType = onnx::TensorProto_DataType_UINT64; break;
    default:
        return nullptr;
    }

    char dim_param = 'a';
    // input
    {
        onnx::ValueInfoProto& variable = *graph.add_input();
        variable.set_name("input");
        //onnx::TypeProto_Tensor* pTensor = variable.mutable_type()->mutable_tensor_type();
        variable.mutable_type()->mutable_tensor_type()->set_elem_type(dataType);
        for (auto dim : shape)
        {
            if (dim == -1)
            {
                variable.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(&dim_param, 1);
                dim_param++;
            }
            else
            {
                variable.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
            }
        }

        if (shape.size() > 0)
        {
            variable.mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_denotation("DATA_BATCH");
        }
    }

    // output
    {
        onnx::ValueInfoProto& variable = *graph.add_output();
        variable.set_name("output");
        //onnx::TypeProto_Tensor* pTensor = variable.mutable_type()->mutable_tensor_type();
        variable.mutable_type()->mutable_tensor_type()->set_elem_type(dataType);
        for (auto dim : shape)
        {
            if (dim == -1)
            {
                variable.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(&dim_param, 1);
                dim_param++;
            }
            else
            {
                variable.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
            }
        }
    }

    struct BufferStreamAdapter : public std::streambuf
    {
        RandomAccessStreamReference BufferAsRandomAccessStreamReference()
        {
            auto buffer = m_dataWriter.DetachBuffer();
            m_dataWriter = DataWriter();

            InMemoryRandomAccessStream stream;
            stream.WriteAsync(buffer).get();
            return RandomAccessStreamReference::CreateFromStream(stream);
        }

    protected:
        virtual int_type overflow(int_type c) {
            if (c != EOF) {
                // convert lowercase to uppercase
                auto temp = static_cast<char>(c);

                m_dataWriter.WriteByte(temp);
            }
            return c;
        }

    private:
        DataWriter m_dataWriter;
    };

    auto size = model.ByteSize();
    auto raw_array = std::unique_ptr<char[]>(new char[size]);
    model.SerializeToArray(raw_array.get(), size);

    BufferStreamAdapter buffer;
    std::ostream os(&buffer);

    os.write(raw_array.get(), size);

    return LearningModel::LoadFromStream(buffer.BufferAsRandomAccessStreamReference());
}

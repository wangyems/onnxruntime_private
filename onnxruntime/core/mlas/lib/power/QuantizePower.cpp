#include <type_traits>
#include "mlasi.h"
#include <altivec.h>

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinearKernel(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::lowest();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    auto ScaleVector = vec_splats(Scale);
    auto MinimumValueVector = vec_splats(float(MinimumValue));
    auto MaximumValueVector = vec_splats(float(MaximumValue));
    auto ZeroPointVector = vec_splats(float(ZeroPoint));

    while (N >= 16) {
        auto FloatVector0 = vec_xl(0, Input);
        auto FloatVector1 = vec_xl(0, Input + 4);
        auto FloatVector2 = vec_xl(0, Input + 8);
        auto FloatVector3 = vec_xl(0, Input + 12);

        FloatVector0 = vec_div(FloatVector0, ScaleVector);
        FloatVector1 = vec_div(FloatVector1, ScaleVector);
        FloatVector2 = vec_div(FloatVector2, ScaleVector);
        FloatVector3 = vec_div(FloatVector3, ScaleVector);

        FloatVector0 = vec_round(FloatVector0);
        FloatVector1 = vec_round(FloatVector1);
        FloatVector2 = vec_round(FloatVector2);
        FloatVector3 = vec_round(FloatVector3);

        FloatVector0 = vec_add(FloatVector0, ZeroPointVector);
        FloatVector1 = vec_add(FloatVector1, ZeroPointVector);
        FloatVector2 = vec_add(FloatVector2, ZeroPointVector);
        FloatVector3 = vec_add(FloatVector3, ZeroPointVector);

        FloatVector0 = vec_max(FloatVector0, MinimumValueVector);
        FloatVector1 = vec_max(FloatVector1, MinimumValueVector);
        FloatVector2 = vec_max(FloatVector2, MinimumValueVector);
        FloatVector3 = vec_max(FloatVector3, MinimumValueVector);

        FloatVector0 = vec_min(FloatVector0, MaximumValueVector);
        FloatVector1 = vec_min(FloatVector1, MaximumValueVector);
        FloatVector2 = vec_min(FloatVector2, MaximumValueVector);
        FloatVector3 = vec_min(FloatVector3, MaximumValueVector);

        auto IntegerVector0 = vec_signed(FloatVector0);
        auto IntegerVector1 = vec_signed(FloatVector1);
        auto IntegerVector2 = vec_signed(FloatVector2);
        auto IntegerVector3 = vec_signed(FloatVector3);

        auto ShortVector0 = vec_pack(IntegerVector0, IntegerVector1);
        auto ShortVector1 = vec_pack(IntegerVector2, IntegerVector3);

        if constexpr (std::is_same_v<OutputType, uint8_t> || std::is_same_v<OutputType, int8_t>) {
            auto CharVector = vec_pack(ShortVector0, ShortVector1);
            vec_xst(CharVector, 0, (int8_t *)Output);
        } else {
            static_assert(std::is_same_v<OutputType, uint16_t> || std::is_same_v<OutputType, int16_t>);
            vec_xst(ShortVector0, 0, (int16_t *)Output);
            vec_xst(ShortVector1, 0, (int16_t *)&Output[8]);
        }

        Output += 16;
        Input += 16;
        N -= 16;
    }

    for (size_t n = 0; n < N; n++) {

        float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
        FloatValue = std::max(FloatValue, float(MinimumValue));
        FloatValue = std::min(FloatValue, float(MaximumValue));
        Output[n] = (OutputType)(int32_t)FloatValue;
    }
}

template<typename UnpackedType>
void
MLASCALL
MlasQuantizeLinearInt4Kernel(
    const float* Input,
    UnpackedType* Output,
    size_t N,
    float Scale,
    UnpackedType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer as int4 using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer. Contains packed 4-bit elements.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    static_assert(std::is_same_v<UnpackedType, uint8_t> || std::is_same_v<UnpackedType, int8_t>);
    constexpr int32_t MinimumValue = Int4Range<UnpackedType>::Min;
    constexpr int32_t MaximumValue = Int4Range<UnpackedType>::Max;

    auto ScaleVector = vec_splats(Scale);
    auto MinimumValueVector = vec_splats(float(MinimumValue));
    auto MaximumValueVector = vec_splats(float(MaximumValue));
    auto ZeroPointVector = vec_splats(float(ZeroPoint));

    // Holds 16 quantized 8-bit values that will be packed into the output as packed 4-bit values.
    UnpackedType TmpOutput[16] = {};

    while (N >= 16) {
        auto FloatVector0 = vec_xl(0, Input);
        auto FloatVector1 = vec_xl(0, Input + 4);
        auto FloatVector2 = vec_xl(0, Input + 8);
        auto FloatVector3 = vec_xl(0, Input + 12);

        FloatVector0 = vec_div(FloatVector0, ScaleVector);
        FloatVector1 = vec_div(FloatVector1, ScaleVector);
        FloatVector2 = vec_div(FloatVector2, ScaleVector);
        FloatVector3 = vec_div(FloatVector3, ScaleVector);

        FloatVector0 = vec_round(FloatVector0);
        FloatVector1 = vec_round(FloatVector1);
        FloatVector2 = vec_round(FloatVector2);
        FloatVector3 = vec_round(FloatVector3);

        FloatVector0 = vec_add(FloatVector0, ZeroPointVector);
        FloatVector1 = vec_add(FloatVector1, ZeroPointVector);
        FloatVector2 = vec_add(FloatVector2, ZeroPointVector);
        FloatVector3 = vec_add(FloatVector3, ZeroPointVector);

        FloatVector0 = vec_max(FloatVector0, MinimumValueVector);
        FloatVector1 = vec_max(FloatVector1, MinimumValueVector);
        FloatVector2 = vec_max(FloatVector2, MinimumValueVector);
        FloatVector3 = vec_max(FloatVector3, MinimumValueVector);

        FloatVector0 = vec_min(FloatVector0, MaximumValueVector);
        FloatVector1 = vec_min(FloatVector1, MaximumValueVector);
        FloatVector2 = vec_min(FloatVector2, MaximumValueVector);
        FloatVector3 = vec_min(FloatVector3, MaximumValueVector);

        auto IntegerVector0 = vec_signed(FloatVector0);
        auto IntegerVector1 = vec_signed(FloatVector1);
        auto IntegerVector2 = vec_signed(FloatVector2);
        auto IntegerVector3 = vec_signed(FloatVector3);

        auto ShortVector0 = vec_pack(IntegerVector0, IntegerVector1);
        auto ShortVector1 = vec_pack(IntegerVector2, IntegerVector3);

        auto CharVector = vec_pack(ShortVector0, ShortVector1);
        vec_xst(CharVector, 0, static_cast<int8_t *>(&TmpOutput[0]));

        Output[0] = static_cast<UnpackedType>(((TmpOutput[1] & 0xF) << 4) | (TmpOutput[0] & 0xF));
        Output[1] = static_cast<UnpackedType>(((TmpOutput[3] & 0xF) << 4) | (TmpOutput[2] & 0xF));
        Output[2] = static_cast<UnpackedType>(((TmpOutput[5] & 0xF) << 4) | (TmpOutput[4] & 0xF));
        Output[3] = static_cast<UnpackedType>(((TmpOutput[7] & 0xF) << 4) | (TmpOutput[6] & 0xF));
        Output[4] = static_cast<UnpackedType>(((TmpOutput[9] & 0xF) << 4) | (TmpOutput[8] & 0xF));
        Output[5] = static_cast<UnpackedType>(((TmpOutput[11] & 0xF) << 4) | (TmpOutput[10] & 0xF));
        Output[6] = static_cast<UnpackedType>(((TmpOutput[13] & 0xF) << 4) | (TmpOutput[12] & 0xF));
        Output[7] = static_cast<UnpackedType>(((TmpOutput[15] & 0xF) << 4) | (TmpOutput[14] & 0xF));

        Output += 8;
        Input += 16;
        N -= 16;
    }

    for (size_t n = 0; n < N; n++) {

        float FloatValue = std::nearbyintf(Input[n] / Scale) + static_cast<float>(ZeroPoint);
        FloatValue = std::max(FloatValue, static_cast<float>(MinimumValue));
        FloatValue = std::min(FloatValue, static_cast<float>(MaximumValue));
        UnpackedType IntValue = static_cast<UnpackedType>(FloatValue);

        MlasSetInt4Element(Output, n, IntValue);
    }
}

void
MLASCALL
MlasQuantizeLinearU8Kernel(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
{
    MlasQuantizeLinearKernel<uint8_t>(Input, Output, N, Scale, ZeroPoint);
}

void
MLASCALL
MlasQuantizeLinearS8Kernel(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    )
{
    MlasQuantizeLinearKernel<int8_t>(Input, Output, N, Scale, ZeroPoint);
}

void
MLASCALL
MlasQuantizeLinearU16Kernel(
    const float* Input,
    uint16_t* Output,
    size_t N,
    float Scale,
    uint16_t ZeroPoint
    )
{
    MlasQuantizeLinearKernel<uint16_t>(Input, Output, N, Scale, ZeroPoint);
}

void
MLASCALL
MlasQuantizeLinearS16Kernel(
    const float* Input,
    int16_t* Output,
    size_t N,
    float Scale,
    int16_t ZeroPoint
    )
{
    MlasQuantizeLinearKernel<int16_t>(Input, Output, N, Scale, ZeroPoint);
}

void
MLASCALL
MlasQuantizeLinearU4Kernel(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
{
    MlasQuantizeLinearInt4Kernel<uint8_t>(Input, Output, N, Scale, ZeroPoint);
}

void
MLASCALL
MlasQuantizeLinearS4Kernel(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    )
{
    MlasQuantizeLinearInt4Kernel<int8_t>(Input, Output, N, Scale, ZeroPoint);
}


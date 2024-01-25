// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUTLASS

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunMoETest(
    const std::vector<float>& input,
    const std::vector<float>& router_probs,
    const std::vector<float>& fc1_experts_weights,
    const std::vector<float>& fc2_experts_weights,
    const std::vector<float>& fc1_experts_bias,
    const std::vector<float>& fc2_experts_bias,
    const std::vector<float>& output_data,
    int num_rows,
    int num_experts,
    int hidden_size,
    int inter_size,
    std::string activation_type,
    bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  if (enable_cuda) {
    OpTester tester("MoE", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("k", static_cast<int64_t>(1));
    tester.AddAttribute<std::string>("activation_type", activation_type);

    std::vector<int64_t> input_dims = {num_rows, hidden_size};
    std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
    std::vector<int64_t> fc1_experts_weights_dims = {num_experts, hidden_size, inter_size};
    std::vector<int64_t> fc2_experts_weights_dims = {num_experts, inter_size, hidden_size};
    std::vector<int64_t> fc1_experts_bias_dims = {num_experts, inter_size};
    std::vector<int64_t> fc2_experts_bias_dims = {num_experts, hidden_size};
    std::vector<int64_t> output_dims = {num_rows, hidden_size};

    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
      tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
      tester.AddInput<MLFloat16>("fc1_experts_weights", fc1_experts_weights_dims, ToFloat16(fc1_experts_weights));
      tester.AddOptionalInputEdge<MLFloat16>(/*fc1_scales*/);
      tester.AddInput<MLFloat16>("fc1_experts_bias", fc1_experts_bias_dims, ToFloat16(fc1_experts_bias));
      tester.AddInput<MLFloat16>("fc2_experts_weights", fc2_experts_weights_dims, ToFloat16(fc2_experts_weights));
      tester.AddOptionalInputEdge<MLFloat16>(/*fc2_scales*/);
      tester.AddInput<MLFloat16>("fc2_experts_bias", fc2_experts_bias_dims, ToFloat16(fc2_experts_bias));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input);
      tester.AddInput<float>("router_probs", router_probs_dims, router_probs);
      tester.AddInput<float>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
      tester.AddOptionalInputEdge<float>(/*fc1_scales*/);
      tester.AddInput<float>("fc1_experts_bias", fc1_experts_bias_dims, fc1_experts_bias);
      tester.AddInput<float>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
      tester.AddOptionalInputEdge<float>(/*fc2_scales*/);
      tester.AddInput<float>("fc2_experts_bias", fc2_experts_bias_dims, fc2_experts_bias);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MoETest, MoETest_Gelu) {
  int num_rows = 4;
  int num_experts = 4;
  int hidden_size = 8;
  int inter_size = 16;

  const std::vector<float> input = {
      -1.1200173f, -0.45884353f, -1.2929888f, 1.0784022f, 0.116372705f, 0.26902613f, -1.8818876f, -0.5457026f,
      0.22222236f, -0.28868636f, 0.6692926f, 1.4944887f, 0.02431708f, -0.49781424f, 0.7378293f, 1.276276f,
      -0.15469065f, -0.28456813f, -0.6296439f, -0.24855971f, 0.80565417f, -1.1018785f, -0.74082595f, 0.82407707f,
      -0.95033455f, 0.659333f, -0.68629056f, -0.2916592f, 1.869919f, -1.1053563f, -0.14417848f, -0.34625578f};
  const std::vector<float> router_probs = {
      -0.84837115f, 0.100507565f, -0.10548311f, 0.40957215f, 1.0159845f, 0.26919764f, 0.021741152f, -0.34184334f,
      -0.71324956f, 0.29018253f, -0.18227568f, 0.31496462f, -0.48426327f, -1.006643f, -0.100081146f, -0.07692295f};
  const std::vector<float> fc1_experts_weights = {
      0.14731085f, 0.52229995f, 0.14753294f, 0.22475791f, 0.20864725f, 0.6708725f, 0.20204341f, 0.4890914f,
      0.52103406f, 0.8223115f, 0.122039974f, 0.15674388f, 0.20966923f, 0.8499667f, 0.3202675f, 0.92174435f,
      0.6808038f, 0.563313f, 0.496278f, 0.40115923f, 0.5627332f, 0.38582766f, 0.49648678f, 0.5637965f,
      0.10889745f, 0.23793429f, 0.90374637f, 0.09422666f, 0.4640969f, 0.99461937f, 0.6806185f, 0.5141565f,
      0.066695035f, 0.74768895f, 0.14385962f, 0.35806787f, 0.33224183f, 0.4259563f, 0.50546914f, 0.91240376f,
      0.5624194f, 0.9478464f, 0.8058562f, 0.18389302f, 0.72425205f, 0.14655197f, 0.28808743f, 0.64706135f,
      0.66509604f, 0.875114f, 0.33904207f, 0.50080043f, 0.7574118f, 0.016453922f, 0.8614903f, 0.08653879f,
      0.50689125f, 0.41499162f, 0.23666352f, 0.5660855f, 0.91345936f, 0.35384023f, 0.20315295f, 0.31508058f,
      0.0044258237f, 0.725697f, 0.25986814f, 0.16632986f, 0.21194929f, 0.787478f, 0.76478684f, 0.8837609f,
      0.68136156f, 0.33302015f, 0.36027592f, 0.647715f, 0.91101736f, 0.6359461f, 0.26342732f, 0.2649613f,
      0.02726549f, 0.608024f, 0.21940875f, 0.054212093f, 0.93843824f, 0.1752944f, 0.44311923f, 0.64324677f,
      0.51592916f, 0.16355914f, 0.09583914f, 0.8985412f, 0.58141935f, 0.91481227f, 0.3323797f, 0.6472777f,
      0.3856619f, 0.47776443f, 0.1954779f, 0.66910046f, 0.65808296f, 0.4896857f, 0.38754892f, 0.1917851f,
      0.8457724f, 0.12778795f, 0.70483273f, 0.33187324f, 0.258766f, 0.58982253f, 0.24027151f, 0.6152024f,
      0.5981904f, 0.12875527f, 0.5832493f, 0.7129646f, 0.6979155f, 0.43706065f, 0.09010619f, 0.42292297f,
      0.67365384f, 0.31756145f, 0.68979055f, 0.8329813f, 0.2389242f, 0.5049309f, 0.7067495f, 0.5391889f,
      0.54176575f, 0.5624327f, 0.10692614f, 0.5392941f, 0.8462349f, 0.9505569f, 0.79387546f, 0.5670015f,
      0.7335071f, 0.25676018f, 0.08565581f, 0.07003945f, 0.99880487f, 0.8173947f, 0.15438312f, 0.6956213f,
      0.8775838f, 0.9998074f, 0.93719745f, 0.8873769f, 0.38537037f, 0.32452917f, 0.9105244f, 0.7801898f,
      0.19911051f, 0.9495086f, 0.7415793f, 0.77256775f, 0.18661183f, 0.6434499f, 0.32471877f, 0.8906783f,
      0.4100297f, 0.69465625f, 0.5888109f, 0.7127341f, 0.33008623f, 0.7437857f, 0.15076452f, 0.6129275f,
      0.16170406f, 0.006731212f, 0.09847212f, 0.89473504f, 0.7705178f, 0.96910787f, 0.9005606f, 0.053477287f,
      0.15878445f, 0.4192087f, 0.17528385f, 0.84719825f, 0.121996105f, 0.25604928f, 0.016954303f, 0.21612722f,
      0.91123873f, 0.90938f, 0.85791886f, 0.88606364f, 0.94459325f, 0.3719685f, 0.72000104f, 0.9454652f,
      0.6654094f, 0.9998382f, 0.75933146f, 0.81082416f, 0.32500392f, 0.73991376f, 0.5574533f, 0.38059133f,
      0.21814507f, 0.21944171f, 0.11525959f, 0.83566517f, 0.8554656f, 0.44309366f, 0.210657f, 0.88645273f,
      0.81974447f, 0.537167f, 0.26393235f, 0.9595239f, 0.70447034f, 0.12042731f, 0.97854143f, 0.8796869f,
      0.31775457f, 0.78107727f, 0.21590549f, 0.42164284f, 0.9245506f, 0.52065957f, 0.14639091f, 0.33288354f,
      0.36427742f, 0.4035356f, 0.5478503f, 0.9624148f, 0.5267702f, 0.19128f, 0.52562714f, 0.7397436f,
      0.7480201f, 0.04303074f, 0.41052878f, 0.12842774f, 0.2866572f, 0.6801467f, 0.1449349f, 0.68586344f,
      0.92438906f, 0.5327942f, 0.16675615f, 0.32085752f, 0.60918206f, 0.11884099f, 0.74840516f, 0.04606521f,
      0.01935333f, 0.014169693f, 0.39856833f, 0.83621645f, 0.026760519f, 0.91559356f, 0.29998857f, 0.64644206f,
      0.52280146f, 0.049140453f, 0.9146645f, 0.7692217f, 0.99699783f, 0.7526061f, 0.1699655f, 0.9172919f,
      0.5268722f, 0.73710823f, 0.09908545f, 0.35618675f, 0.009061217f, 0.30525374f, 0.6078656f, 0.10741913f,
      0.6593821f, 0.7684034f, 0.56965464f, 0.16545832f, 0.11234015f, 0.3457417f, 0.7194791f, 0.9931982f,
      0.7875145f, 0.44369537f, 0.6753082f, 0.009468555f, 0.07294935f, 0.73330396f, 0.2167924f, 0.74054784f,
      0.14703393f, 0.25234455f, 0.08815551f, 0.76092035f, 0.44905245f, 0.88480055f, 0.8094361f, 0.7766713f,
      0.51607805f, 0.345411f, 0.39128417f, 0.5664503f, 0.74785477f, 0.14970505f, 0.91963893f, 0.44563496f,
      0.08102721f, 0.22947109f, 0.94240886f, 0.9572636f, 0.036860168f, 0.85264915f, 0.7505796f, 0.79595923f,
      0.9232646f, 0.23052484f, 0.6578879f, 0.7046166f, 0.35225332f, 0.66732657f, 0.3561433f, 0.80913067f,
      0.3612727f, 0.31360215f, 0.6258745f, 0.6773468f, 0.25571418f, 0.54419917f, 0.78976786f, 0.45025164f,
      0.65216696f, 0.3794065f, 0.6752498f, 0.1378029f, 0.2059856f, 0.24620473f, 0.95950544f, 0.36545795f,
      0.49863482f, 0.25775224f, 0.99914503f, 0.9883351f, 0.122906685f, 0.09466505f, 0.12100351f, 0.49758863f,
      0.37254804f, 0.17272717f, 0.32066393f, 0.59446543f, 0.23875463f, 0.61079127f, 0.38534206f, 0.25771832f,
      0.56869274f, 0.9111291f, 0.16196036f, 0.5232172f, 0.31561613f, 0.99065316f, 0.025618374f, 0.0206694f,
      0.9926925f, 0.18365502f, 0.5958617f, 0.45684695f, 0.3946715f, 0.3883261f, 0.8177203f, 0.5238985f,
      0.013192713f, 0.20481992f, 0.32954985f, 0.7516082f, 0.17643315f, 0.9714598f, 0.38863534f, 0.410219f,
      0.891779f, 0.75130385f, 0.92406017f, 0.7892222f, 0.34832305f, 0.1682638f, 0.46279848f, 0.9138188f,
      0.3321901f, 0.036315024f, 0.7049642f, 0.9867357f, 0.3576584f, 0.08598822f, 0.046470165f, 0.6252997f,
      0.46214014f, 0.24750638f, 0.60106593f, 0.6898794f, 0.8976595f, 0.8881911f, 0.42515814f, 0.059116423f,
      0.048188448f, 0.9668448f, 0.7210276f, 0.7179537f, 0.06738949f, 0.96300787f, 0.97367156f, 0.95143014f,
      0.07820749f, 0.3113383f, 0.1561181f, 0.9734828f, 0.28516f, 0.27172273f, 0.76195645f, 0.26870382f,
      0.25373894f, 0.45626426f, 0.45194024f, 0.11051077f, 0.91683406f, 0.27943915f, 0.67735744f, 0.9348918f,
      0.7521582f, 0.57078993f, 0.9254285f, 0.5672131f, 0.2686717f, 0.97299975f, 0.61834025f, 0.012159586f,
      0.3576542f, 0.15941626f, 0.9383765f, 0.41742706f, 0.044237554f, 0.46856833f, 0.81400645f, 0.6299002f,
      0.6581022f, 0.5464366f, 0.68640935f, 0.378174f, 0.3010999f, 0.032645762f, 0.12333155f, 0.71670127f,
      0.20394331f, 0.57173324f, 0.6595957f, 0.53540194f, 0.17582512f, 0.9781642f, 0.20925027f, 0.9112503f,
      0.10224587f, 0.37972575f, 0.7719844f, 0.29570967f, 0.9200215f, 0.15592176f, 0.080114245f, 0.27454042f,
      0.5808252f, 0.96037793f, 0.26129955f, 0.6788141f, 0.37464648f, 0.39156884f, 0.8676517f, 0.112507045f,
      0.55310667f, 0.9702046f, 0.4312939f, 0.88821906f, 0.3460216f, 0.9024811f, 0.016334832f, 0.42793816f,
      0.4121768f, 0.6620425f, 0.6961637f, 0.88390845f, 0.425507f, 0.48017246f, 0.8424056f, 0.36471343f,
      0.9383168f, 0.16709393f, 0.44589508f, 0.47314453f, 0.72310495f, 0.84183806f, 0.4207481f, 0.0857597f,
      0.7477461f, 0.6495659f, 0.70084965f, 0.19156617f, 0.8217978f, 0.9735775f, 0.5433857f, 0.032975793f,
      0.85099494f, 0.12927437f, 0.61493605f, 0.5726589f, 0.26598173f, 0.6740978f, 0.052783668f, 0.61387974f};
  const std::vector<float> fc2_experts_weights = {
      0.18302453f, 0.44593316f, 0.5643144f, 0.9259722f, 0.26143986f, 0.82031804f, 0.4364831f, 0.2625361f,
      0.06460017f, 0.04124081f, 0.98830533f, 0.37530023f, 0.5249744f, 0.63555616f, 0.8398661f, 0.92673707f,
      0.9055086f, 0.12955844f, 0.4198916f, 0.20413119f, 0.21432412f, 0.6186035f, 0.969324f, 0.099448025f,
      0.80260223f, 0.24076664f, 0.40261286f, 0.89688545f, 0.38691485f, 0.5455279f, 0.15048373f, 0.92562044f,
      0.43536508f, 0.13430476f, 0.64640516f, 0.14449131f, 0.10324633f, 0.5304596f, 0.8964218f, 0.358508f,
      0.73533344f, 0.9296606f, 0.83163047f, 0.23771948f, 0.44519007f, 0.34265757f, 0.09793854f, 0.5002066f,
      0.87621754f, 0.9212578f, 0.54665035f, 0.6135615f, 0.28353918f, 0.8774212f, 0.29194576f, 0.1526736f,
      0.57699674f, 0.7996927f, 0.04920423f, 0.95198375f, 0.67986554f, 0.14969361f, 0.39229625f, 0.93378997f,
      0.11638266f, 0.3538614f, 0.66399014f, 0.06195748f, 0.7740991f, 0.7602738f, 0.81010276f, 0.18122643f,
      0.9980005f, 0.20361924f, 0.99917024f, 0.020154774f, 0.054515004f, 0.80709815f, 0.55225646f, 0.52884465f,
      0.22312081f, 0.29026228f, 0.35380626f, 0.012922287f, 0.52598435f, 0.58842945f, 0.4995767f, 0.66146517f,
      0.9744255f, 0.632942f, 0.3169638f, 0.29422665f, 0.18009722f, 0.15339059f, 0.41947508f, 0.4115672f,
      0.72243124f, 0.2862816f, 0.89860183f, 0.14915991f, 0.5014211f, 0.94945997f, 0.99719256f, 0.21036887f,
      0.5890645f, 0.55906135f, 0.26557416f, 0.32725257f, 0.635427f, 0.1523174f, 0.58249784f, 0.71636236f,
      0.30296493f, 0.9153206f, 0.46709478f, 0.72685635f, 0.9951532f, 0.34716582f, 0.7717041f, 0.3569854f,
      0.4269635f, 0.41526443f, 0.4968937f, 0.3111158f, 0.61719346f, 0.5188402f, 0.8169449f, 0.39879733f,
      0.5501401f, 0.31400484f, 0.08127314f, 0.7023336f, 0.56397897f, 0.29975814f, 0.33094752f, 0.63076067f,
      0.40959156f, 0.82673794f, 0.52832156f, 0.68886834f, 0.7178481f, 0.37731683f, 0.71633244f, 0.86896664f,
      0.5230092f, 0.59784645f, 0.5181678f, 0.8461837f, 0.28890234f, 0.23421508f, 0.7178768f, 0.06484294f,
      0.5080162f, 0.27005446f, 0.8300168f, 0.034480453f, 0.8031663f, 0.9946784f, 0.60117006f, 0.46668667f,
      0.9921749f, 0.28632385f, 0.45993322f, 0.28104752f, 0.43097937f, 0.60866946f, 0.5667807f, 0.40556252f,
      7.969141e-05f, 0.52560204f, 0.48518902f, 0.5752184f, 0.8831251f, 0.9860047f, 0.20335877f, 0.46882278f,
      0.2996632f, 0.03917718f, 0.13617045f, 0.96928054f, 0.79153055f, 0.76857555f, 0.7778716f, 0.102760494f,
      0.5525096f, 0.9653573f, 0.22095704f, 0.94479716f, 0.63141924f, 0.8517718f, 0.28580618f, 0.73050886f,
      0.05675614f, 0.46825224f, 0.6667756f, 0.6499472f, 0.91840404f, 0.99132854f, 0.9548785f, 0.8356961f,
      0.851531f, 0.43548512f, 0.111976564f, 0.31438643f, 0.44386774f, 0.22980672f, 0.75558543f, 0.6755136f,
      0.58067596f, 0.62078035f, 0.93922615f, 0.6821157f, 0.061530292f, 0.13705963f, 0.7203748f, 0.5681396f,
      0.7438458f, 0.0006400347f, 0.038565338f, 0.8066132f, 0.81982285f, 0.047644496f, 0.68979263f, 0.109577894f,
      0.8786539f, 0.6568952f, 0.99439347f, 0.0070040226f, 0.018661916f, 0.838051f, 0.94391155f, 0.80634f,
      0.8324149f, 0.078864336f, 0.8619068f, 0.027926445f, 0.61170083f, 0.17248261f, 0.30140227f, 0.5885344f,
      0.30341f, 0.42088854f, 0.02608782f, 0.02856338f, 0.69368154f, 0.28836077f, 0.19580519f, 0.30270886f,
      0.09121573f, 0.100299895f, 0.79918617f, 0.75412107f, 0.56660175f, 0.22687018f, 0.6663505f, 0.5224626f,
      0.1426636f, 0.6075949f, 0.95527196f, 0.008196831f, 0.0028039217f, 0.5640625f, 0.87651116f, 0.19575512f,
      0.61006856f, 0.85149264f, 0.6541582f, 0.6082054f, 0.998863f, 0.82573634f, 0.21878648f, 0.54321826f,
      0.7554362f, 0.94095474f, 0.002533555f, 0.77075267f, 0.35483408f, 0.010389388f, 0.610987f, 0.22779316f,
      0.5708561f, 0.17537653f, 0.12373549f, 0.4575745f, 0.33203715f, 0.79243237f, 0.54310906f, 0.8902793f,
      0.5937015f, 0.33921933f, 0.8386668f, 0.52732253f, 0.59384584f, 0.3391887f, 0.5017944f, 0.40386343f,
      0.45749134f, 0.110060334f, 0.49692506f, 0.084977865f, 0.3924346f, 0.7897731f, 0.15232486f, 0.16297412f,
      0.37791175f, 0.36293298f, 0.5846437f, 0.5830078f, 0.75354826f, 0.15555972f, 0.4647144f, 0.7796456f,
      0.93248576f, 0.46352726f, 0.2106899f, 0.6437313f, 0.78473866f, 0.18762505f, 0.20985329f, 0.7209991f,
      0.464967f, 0.02775067f, 0.21170747f, 0.7027664f, 0.33041215f, 0.8451145f, 0.89526993f, 0.57273495f,
      0.46046263f, 0.34128642f, 0.47471708f, 0.59101045f, 0.11807448f, 0.38050216f, 0.08409953f, 0.80687743f,
      0.18158185f, 0.9567719f, 0.3711096f, 0.21356237f, 0.74022657f, 0.57453954f, 0.846228f, 0.70873487f,
      0.018330276f, 0.8162452f, 0.40584308f, 0.27901447f, 0.81752694f, 0.86466515f, 0.060534656f, 0.45478833f,
      0.9106033f, 0.6936434f, 0.92123467f, 0.32865065f, 0.22417879f, 0.9299548f, 0.70841146f, 0.97999126f,
      0.2911517f, 0.17896658f, 0.44139355f, 0.029210031f, 0.6959876f, 0.8687942f, 0.62002844f, 0.45059657f,
      0.74790317f, 0.18262434f, 0.98912156f, 0.0028281808f, 0.021027386f, 0.38184917f, 0.90842223f, 0.5500629f,
      0.69202286f, 0.13349658f, 0.6823429f, 0.44412827f, 0.7004118f, 0.8531213f, 0.7173401f, 0.4574679f,
      0.46920043f, 0.18640989f, 0.31914896f, 0.82491904f, 0.29950172f, 0.8105199f, 0.30173403f, 0.38355058f,
      0.5106411f, 0.04116726f, 0.49500751f, 0.44960213f, 0.45508182f, 0.4000479f, 0.89418864f, 0.8689936f,
      0.16112137f, 0.7322634f, 0.10780871f, 0.07433933f, 0.652841f, 0.50734824f, 0.26674682f, 0.017748117f,
      0.30643195f, 0.66699976f, 0.03719926f, 0.014267266f, 0.56343627f, 0.13979793f, 0.061959863f, 0.3073569f,
      0.41949958f, 0.045647383f, 0.16613615f, 0.5327839f, 0.028514147f, 0.4297228f, 0.17714864f, 0.15338135f,
      0.6965155f, 0.11515516f, 0.1210829f, 0.78514075f, 0.59348315f, 0.9553564f, 0.36635226f, 0.25849247f,
      0.45372677f, 0.5025297f, 0.88132215f, 0.0019600391f, 0.46439964f, 0.7211761f, 0.22465849f, 0.2459296f,
      0.7416339f, 0.020907402f, 0.6184779f, 0.112906754f, 0.7485309f, 0.072479784f, 0.8074024f, 0.026683688f,
      0.07971662f, 0.50736845f, 0.8939942f, 0.0718022f, 0.27697015f, 0.9391413f, 0.4161513f, 0.7071423f,
      0.019000888f, 0.34275955f, 0.24608392f, 0.9215306f, 0.70751995f, 0.13516217f, 0.5806135f, 0.49425328f,
      0.29456508f, 0.21446168f, 0.3340807f, 0.89411324f, 0.14157385f, 0.14382833f, 0.34574044f, 0.50869817f,
      0.63610595f, 0.51500404f, 0.37963718f, 0.19682491f, 0.41028368f, 0.29872334f, 0.9039644f, 0.013295233f,
      0.1810705f, 0.093204916f, 0.4086216f, 0.8896367f, 0.9382696f, 0.06472236f, 0.47833657f, 0.7934831f,
      0.7203987f, 0.9095519f, 0.4861309f, 0.16405362f, 0.83076525f, 0.3285427f, 0.7588931f, 0.37678176f,
      0.71254706f, 0.949713f, 0.96492773f, 0.044967473f, 0.16925985f, 0.2932666f, 0.18114948f, 0.97975004f,
      0.4558406f, 0.16832972f, 0.27750528f, 0.2238177f, 0.7039947f, 0.06387442f, 0.033798456f, 0.007119417f};
  const std::vector<float> fc1_experts_bias = {
      0.71526206f, 0.7472273f, 0.18946046f, 0.6239893f, 0.86909235f, 0.5726507f, 0.3942092f, 0.5369412f,
      0.44638616f, 0.7517496f, 0.16049433f, 0.75355124f, 0.7818118f, 0.19706267f, 0.9082818f, 0.9910924f,
      0.30288565f, 0.3599528f, 0.74917775f, 0.10828978f, 0.697729f, 0.61665237f, 0.81516486f, 0.0656966f,
      0.0846076f, 0.72456455f, 0.6801054f, 0.034616888f, 0.22117025f, 0.042510748f, 0.14178854f, 0.27440017f,
      0.91376925f, 0.40047455f, 0.7871756f, 0.97484046f, 0.7278661f, 0.052394807f, 0.75161135f, 0.6907173f,
      0.8875328f, 0.0067828894f, 0.807508f, 0.9092707f, 0.034817636f, 0.55231315f, 0.92683655f, 0.13634592f,
      0.66405964f, 0.7209387f, 0.63104504f, 0.9971379f, 0.9093898f, 0.9289774f, 0.4376766f, 0.9193563f,
      0.03404367f, 0.23018533f, 0.39305943f, 0.3514716f, 0.96184736f, 0.73583263f, 0.8219065f, 0.8401047f};
  const std::vector<float> fc2_experts_bias = {
      0.12649822f, 0.4420895f, 0.5730123f, 0.63004625f, 0.7571163f, 0.3010466f, 0.3492328f, 0.91837066f,
      0.36580783f, 0.15267932f, 0.8390199f, 0.83857775f, 0.34321654f, 0.40003997f, 0.13106f, 0.08245313f,
      0.68802476f, 0.28640372f, 0.89804775f, 0.09964341f, 0.43088746f, 0.5107959f, 0.75697356f, 0.90466535f,
      0.83860224f, 0.720098f, 0.2705031f, 0.14292616f, 0.052693605f, 0.5248023f, 0.9849401f, 0.40502876f};
  const std::vector<float> output = {
      0.2552814f, 0.17651685f, 0.0034551744f, -0.123282805f, 0.0073816925f, 0.004265253f, 0.16927283f, -0.05276826f,
      9.555821f, 7.6907287f, 10.626425f, 7.0543795f, 8.10093f, 10.3664465f, 10.925815f, 8.737018f,
      0.565234f, 0.17098689f, 0.10810414f, 0.43916586f, 0.3535297f, 0.45673048f, 0.3853893f, 0.18613164f,
      1.3354061f, 0.5049282f, 0.72775036f, 0.90331376f, 1.2945517f, 0.9123066f, 1.1995136f, 0.7708638f};

  RunMoETest(input,
             router_probs,
             fc1_experts_weights,
             fc2_experts_weights,
             fc1_experts_bias,
             fc2_experts_bias,
             output,
             num_rows,
             num_experts,
             hidden_size,
             inter_size,
             "gelu");
}

TEST(MoETest, MoETest_Relu) {
  int num_rows = 4;
  int num_experts = 4;
  int hidden_size = 8;
  int inter_size = 16;

  const std::vector<float> input = {
      0.7670296f, -0.93721074f, -2.330477f, -0.78088343f, 0.8250065f, 1.2206652f, -0.06297584f, 1.1463639f,
      1.2215378f, -0.31372663f, -0.7234253f, -0.3627346f, 0.44249064f, 0.19418247f, -0.49998695f, -0.55005103f,
      0.023851749f, -1.5203826f, 0.52939993f, -0.39082858f, -1.9291036f, 0.034976702f, -0.48336256f, -1.226073f,
      -0.33963847f, 0.0073261578f, -0.0521804f, 1.16749f, 1.7302082f, 2.0561688f, -0.2347232f, -1.3456243f};
  const std::vector<float> router_probs = {
      -0.08146476f, -0.40439552f, 1.0100367f, -0.7724162f, -0.08113786f, -0.36328858f, 0.3688482f, -0.013465762f,
      -0.32420647f, -0.3815508f, 0.79585606f, 0.14430691f, -0.21869831f, 0.11483674f, -0.11992836f, 0.35216537f};
  const std::vector<float> fc1_experts_weights = {
      0.81960344f, 0.9296998f, 0.45050132f, 0.38805157f, 0.50729614f, 0.47014588f, 0.62020564f, 0.6401168f,
      0.045871615f, 0.31548113f, 0.92106473f, 0.6947775f, 0.4751312f, 0.19854712f, 0.19409746f, 0.052116573f,
      0.3370188f, 0.6688521f, 0.8188108f, 0.73084867f, 0.058027983f, 0.19931877f, 0.42109168f, 0.98367476f,
      0.57232875f, 0.37051463f, 0.7068576f, 0.30955923f, 0.17637217f, 0.8649436f, 0.2726491f, 0.39976662f,
      0.0025978684f, 0.8346353f, 0.8788173f, 0.6822241f, 0.1513629f, 0.0065300465f, 0.093910515f, 0.8728501f,
      0.7400529f, 0.9207522f, 0.76193494f, 0.6265461f, 0.49510366f, 0.11974698f, 0.07161391f, 0.032325685f,
      0.704681f, 0.254516f, 0.3993737f, 0.21224737f, 0.40888822f, 0.14808255f, 0.17329216f, 0.6658554f,
      0.3514018f, 0.8086716f, 0.33959562f, 0.13321638f, 0.41178054f, 0.2576263f, 0.3470292f, 0.024002194f,
      0.77974546f, 0.15189773f, 0.75130886f, 0.7268921f, 0.85721636f, 0.11647397f, 0.8595984f, 0.2636242f,
      0.6855346f, 0.96955734f, 0.42948407f, 0.49613327f, 0.38488472f, 0.08250773f, 0.73995143f, 0.003641069f,
      0.81039995f, 0.87411255f, 0.9728532f, 0.38206023f, 0.08917904f, 0.61241513f, 0.77621365f, 0.0023456216f,
      0.38650817f, 0.20027226f, 0.45626813f, 0.25389326f, 0.2956162f, 0.34127057f, 0.024847984f, 0.91025376f,
      0.9191656f, 0.42156547f, 0.44305897f, 0.29594004f, 0.04846859f, 0.013427794f, 0.6858292f, 0.22547692f,
      0.17856151f, 0.4609884f, 0.33349442f, 0.3382396f, 0.5160656f, 0.3939438f, 0.3278438f, 0.26059705f,
      0.0930863f, 0.9192536f, 0.29990643f, 0.63248974f, 0.32651705f, 0.54063064f, 0.9661502f, 0.73036134f,
      0.06670016f, 0.6984514f, 0.9746214f, 0.63154167f, 0.83521235f, 0.99294376f, 0.4233855f, 0.6037772f,
      0.15248245f, 0.39696145f, 0.8702919f, 0.7563229f, 0.18360549f, 0.099057496f, 0.15831816f, 0.00656116f,
      0.114180505f, 0.3763513f, 0.8374386f, 0.5836911f, 0.11969727f, 0.09888804f, 0.74873763f, 0.12807935f,
      0.43843627f, 0.739853f, 0.26859397f, 0.44548005f, 0.45647776f, 0.38170832f, 0.24648392f, 0.054280818f,
      0.0958215f, 0.23226917f, 0.98291886f, 0.25849265f, 0.16423601f, 0.6211971f, 0.63780516f, 0.77395487f,
      0.8800602f, 0.7784371f, 0.004249513f, 0.5443443f, 0.80287653f, 0.45378727f, 0.20536041f, 0.9766699f,
      0.31298608f, 0.21532774f, 0.04922247f, 0.52233416f, 0.72156656f, 0.6106814f, 0.59887487f, 0.12080628f,
      0.03305638f, 0.5088047f, 0.95591706f, 0.7884607f, 0.20888287f, 0.43509573f, 0.13140821f, 0.2587883f,
      0.5905492f, 0.77226925f, 0.91418463f, 0.04094696f, 0.8343076f, 0.14735395f, 0.6872336f, 0.92312264f,
      0.5070212f, 0.9549045f, 0.07397425f, 0.3090204f, 0.79162645f, 0.39106607f, 0.39764988f, 0.29160416f,
      0.84465307f, 0.7452516f, 0.66022503f, 0.21901816f, 0.09412521f, 0.5540803f, 0.6481394f, 0.26914406f,
      0.36010116f, 0.83768386f, 0.53982985f, 0.52255917f, 0.37694973f, 0.04720515f, 0.029871285f, 0.26099247f,
      0.2458393f, 0.6557768f, 0.35444462f, 0.30438894f, 0.9767149f, 0.67416143f, 0.85645115f, 0.25794363f,
      0.2957666f, 0.68377024f, 0.16686243f, 0.17314798f, 0.47585016f, 0.31711966f, 0.125171f, 0.7965795f,
      0.90208143f, 0.58111167f, 0.41294336f, 0.036863506f, 0.31788063f, 0.6272928f, 0.73576546f, 0.43679124f,
      0.30232358f, 0.77861303f, 0.10180014f, 0.816009f, 0.30602258f, 0.5076527f, 0.40119207f, 0.5606195f,
      0.3489008f, 0.8635635f, 0.48700142f, 0.89029974f, 0.98074025f, 0.25640452f, 0.13524544f, 0.901151f,
      0.89180696f, 0.11822635f, 0.46134835f, 0.006936848f, 0.09070045f, 0.59657127f, 0.6330173f, 0.6059905f,
      0.36391765f, 0.96128887f, 0.571489f, 0.2049576f, 0.4716931f, 0.6200726f, 0.67509633f, 0.14645958f,
      0.6873948f, 0.24455917f, 0.08452982f, 0.22689629f, 0.9822047f, 0.9274289f, 0.9477422f, 0.7935056f,
      0.87772477f, 0.43307513f, 0.22488606f, 0.7498283f, 0.24090862f, 0.16256708f, 0.34033298f, 0.6049296f,
      0.7573983f, 0.3057955f, 0.20571685f, 0.56744653f, 0.2052834f, 0.17446929f, 0.76062596f, 0.4160077f,
      0.9568925f, 0.9863913f, 0.64955276f, 0.67207885f, 0.61514187f, 0.50783044f, 0.46363378f, 0.50687206f,
      0.6867124f, 0.9648854f, 0.37042046f, 0.2886421f, 0.37891757f, 0.25843787f, 0.58501935f, 0.8732242f,
      0.8909887f, 0.72956276f, 0.13203424f, 0.23164761f, 0.3901443f, 0.40783793f, 0.54112387f, 0.041014254f,
      0.65562236f, 0.11856395f, 0.18362767f, 0.08430874f, 0.9356598f, 0.026530087f, 0.8771834f, 0.48319155f,
      0.4418506f, 0.81273925f, 0.4537862f, 0.81357706f, 0.8615075f, 0.06589496f, 0.692392f, 0.5943895f,
      0.60750586f, 0.5729957f, 0.6367655f, 0.2594666f, 0.43602943f, 0.97506f, 0.83592474f, 0.48121578f,
      0.029734552f, 0.5219139f, 0.15951324f, 0.90659577f, 0.19645631f, 0.4638992f, 0.38902867f, 0.5889769f,
      0.9705138f, 0.5475096f, 0.789582f, 0.8881108f, 0.9036556f, 0.32732427f, 0.38817167f, 0.7409689f,
      0.36356616f, 0.734132f, 0.39076614f, 0.16087383f, 0.70352167f, 0.576659f, 0.7229242f, 0.996743f,
      0.84136647f, 0.97399056f, 0.5267614f, 0.06989372f, 0.14923638f, 0.18941313f, 0.059375823f, 0.24937624f,
      0.039716125f, 0.038692355f, 0.20122272f, 0.0070830584f, 0.19309378f, 0.69065434f, 0.9170264f, 0.3512686f,
      0.3545606f, 0.76697665f, 0.25331455f, 0.26358372f, 0.80806476f, 0.064349174f, 0.5611374f, 0.941691f,
      0.58574325f, 0.6359719f, 0.20880443f, 0.49310172f, 0.5274922f, 0.62271714f, 0.694273f, 0.9344639f,
      0.11835027f, 0.51498765f, 0.25018185f, 0.10446805f, 0.45996118f, 0.059881568f, 0.8489496f, 0.5579074f,
      0.23052096f, 0.76128954f, 0.02678603f, 0.3066004f, 0.40259063f, 0.07512486f, 0.18205583f, 0.4183907f,
      0.8793823f, 0.9828271f, 0.8181312f, 0.20143801f, 0.17288941f, 0.9363466f, 0.6768587f, 0.51328385f,
      0.56766605f, 0.098151624f, 0.33305728f, 0.98130906f, 0.3766839f, 0.47491795f, 0.08483446f, 0.22029644f,
      0.4897902f, 0.18942028f, 0.4379952f, 0.7034796f, 0.0109113455f, 0.64850605f, 0.16939592f, 0.25597447f,
      0.69195485f, 0.8975601f, 0.36334568f, 0.29471546f, 0.04788208f, 0.24217117f, 0.062181532f, 0.38556474f,
      0.6020277f, 0.03156215f, 0.93655676f, 0.81369543f, 0.010527074f, 0.2611835f, 0.6630776f, 0.3972702f,
      0.44551176f, 0.27424216f, 0.9016098f, 0.22050089f, 0.9146384f, 0.53226113f, 0.6005109f, 0.8900659f,
      0.4176172f, 0.21532834f, 0.4191329f, 0.9055267f, 0.12900633f, 0.6134902f, 0.008604288f, 0.76215106f,
      0.68473387f, 0.5211961f, 0.71459657f, 0.50056237f, 0.7766764f, 0.10418975f, 0.42657375f, 0.7218073f,
      0.9979084f, 0.7546957f, 0.1364128f, 0.8845484f, 0.38850087f, 0.39324278f, 0.04554516f, 0.42129284f,
      0.8536634f, 0.5697224f, 0.20877302f, 0.65390605f, 0.3396778f, 0.956497f, 0.066022694f, 0.34206223f,
      0.017213225f, 0.3030849f, 0.6576238f, 0.9813073f, 0.58397317f, 0.99017924f, 0.59782606f, 0.788768f,
      0.9008311f, 0.91796166f, 0.22013813f, 0.959695f, 0.80288273f, 0.2662105f, 0.26139832f, 0.080626905f};
  const std::vector<float> fc2_experts_weights = {
      0.6255686f, 0.09472537f, 0.71121234f, 0.65789884f, 0.065598905f, 0.63625044f, 0.45933473f, 0.7284089f,
      0.7868948f, 0.0029274821f, 0.95854944f, 0.919321f, 0.6989418f, 0.043019474f, 0.32138962f, 0.35509557f,
      0.37150103f, 0.78196156f, 0.6817853f, 0.89608955f, 0.31273842f, 0.6682699f, 0.6778976f, 0.08370459f,
      0.014990091f, 0.24055547f, 0.84227383f, 0.029270172f, 0.0647831f, 0.7801003f, 0.7697645f, 0.91119635f,
      0.12253064f, 0.13405013f, 0.75649333f, 0.9348151f, 0.7991694f, 0.57832605f, 0.66478735f, 0.97456336f,
      0.17739785f, 0.2729941f, 0.8497335f, 0.15788019f, 0.22429371f, 0.86499554f, 0.65776104f, 0.661535f,
      0.2880798f, 0.49309975f, 0.9576164f, 0.19988996f, 0.5039311f, 0.73779976f, 0.15482187f, 0.98558843f,
      0.25019473f, 0.379932f, 0.36471486f, 0.17417055f, 0.009367704f, 0.7819258f, 0.63283706f, 0.031699598f,
      0.1781866f, 0.994184f, 0.6911175f, 0.7006223f, 0.20085096f, 0.28080195f, 0.42452294f, 0.40856004f,
      0.15737581f, 0.5411925f, 0.549694f, 0.4366895f, 0.5693159f, 0.3018247f, 0.63012594f, 0.6885702f,
      0.2366305f, 0.004210472f, 0.7617172f, 0.61926836f, 0.24570602f, 0.981851f, 0.273876f, 0.8378734f,
      0.75366426f, 0.080795944f, 0.82247066f, 0.040263534f, 0.22299266f, 0.41664255f, 0.16297674f, 0.98845494f,
      0.39971018f, 0.69859487f, 0.053544044f, 0.7878332f, 0.34460813f, 0.11966437f, 0.5731115f, 0.7422309f,
      0.93269855f, 0.19460368f, 0.25394785f, 0.59613144f, 0.6356306f, 0.6922361f, 0.7744376f, 0.38662314f,
      0.7777848f, 0.8686458f, 0.36938924f, 0.8557286f, 0.74428976f, 0.9410264f, 0.21586305f, 0.2530955f,
      0.35543054f, 0.52536315f, 0.8000995f, 0.21456867f, 0.750327f, 0.3208093f, 0.80205464f, 0.47626138f,
      0.061956525f, 0.22487706f, 0.13812399f, 0.74798125f, 0.1647259f, 0.45834088f, 0.6078779f, 0.22580266f,
      0.644235f, 0.011788309f, 0.14224577f, 0.0469383f, 0.34876132f, 0.3178513f, 0.5715967f, 0.40754277f,
      0.735041f, 0.9583977f, 0.67939556f, 0.30301625f, 0.031807184f, 0.68110096f, 0.25227106f, 0.75443816f,
      0.83424246f, 0.69286025f, 0.9691554f, 0.9748982f, 0.60586995f, 0.13568163f, 0.94672066f, 0.26275212f,
      0.2638232f, 0.9183893f, 0.88740516f, 0.65107566f, 0.5313419f, 0.07941705f, 0.44809794f, 0.9795632f,
      0.6273294f, 0.542809f, 0.3961745f, 0.32560885f, 0.79801136f, 0.53083426f, 0.8252871f, 0.4115007f,
      0.7184546f, 0.70638496f, 0.57973206f, 0.8141865f, 0.81332296f, 0.96346164f, 0.88438797f, 0.37215167f,
      0.0766899f, 0.5914087f, 0.49563587f, 0.3695873f, 0.41627264f, 0.5235164f, 0.86481494f, 0.6558706f,
      0.32245284f, 0.29438752f, 0.37618434f, 0.3067485f, 0.9496114f, 0.76482266f, 0.95148784f, 0.5015968f,
      0.60083544f, 0.67338234f, 0.026723444f, 0.5446483f, 0.466555f, 0.21967298f, 0.112026334f, 0.9426372f,
      0.906533f, 0.73173434f, 0.97712487f, 0.29709607f, 0.41363865f, 0.6893093f, 0.4173867f, 0.4018826f,
      0.086719275f, 0.63433063f, 0.1978364f, 0.5181831f, 0.9874878f, 0.34609234f, 0.34240413f, 0.8016564f,
      0.31617337f, 0.4570613f, 0.96686924f, 0.29501313f, 0.14229488f, 0.22017813f, 0.36137718f, 0.26275063f,
      0.24053413f, 0.70197225f, 0.58496886f, 0.33996922f, 0.11154431f, 0.34257007f, 0.28898042f, 0.33729053f,
      0.048938513f, 0.60771453f, 0.13263822f, 0.11060041f, 0.091483414f, 0.70869184f, 0.19898665f, 0.29362458f,
      0.8919203f, 0.7654821f, 0.7866956f, 0.02524674f, 0.1414501f, 0.3112445f, 0.9130488f, 0.5511502f,
      0.12605143f, 0.5031309f, 0.11166459f, 0.39045036f, 0.36251247f, 0.9328308f, 0.65486836f, 0.41281444f,
      0.5844644f, 0.35566723f, 0.6964502f, 0.6977819f, 0.63427305f, 0.30511153f, 0.92657536f, 0.42781502f,
      0.30534166f, 0.813157f, 0.90752834f, 0.9975799f, 0.64812917f, 0.32955307f, 0.753946f, 0.92897725f,
      0.009582937f, 0.43805653f, 0.15901726f, 0.5931799f, 0.7067924f, 0.39670604f, 0.45817143f, 0.7250554f,
      0.41596514f, 0.08011025f, 0.900068f, 0.24834275f, 0.44507074f, 0.5471632f, 0.46995157f, 0.029657006f,
      0.7294f, 0.27288425f, 0.2406702f, 0.6194577f, 0.23906898f, 0.26892018f, 0.33152503f, 0.3121612f,
      0.29118127f, 0.36515707f, 0.6299379f, 0.095391035f, 0.19735986f, 0.5072957f, 0.56953406f, 0.77614623f,
      0.14877802f, 0.65959847f, 0.7841949f, 0.7776301f, 0.03428924f, 0.3091979f, 0.07021719f, 0.18359429f,
      0.77849144f, 0.42534047f, 0.7123557f, 0.20649683f, 0.57597995f, 0.19757104f, 0.749946f, 0.2813105f,
      0.37462044f, 0.06618434f, 0.50165176f, 0.9747401f, 0.7426891f, 0.23322952f, 0.50672436f, 0.44517577f,
      0.09746289f, 0.89204556f, 0.50806034f, 0.6052985f, 0.2980855f, 0.26604044f, 0.5824448f, 0.68485546f,
      0.612149f, 0.25902748f, 0.9854489f, 0.4263978f, 0.19379246f, 0.26614368f, 0.9922104f, 0.5000241f,
      0.4321279f, 0.2919191f, 0.3689273f, 0.078885734f, 0.10265827f, 0.79264474f, 0.9277247f, 0.9771502f,
      0.13902885f, 0.77043164f, 0.19051671f, 0.7982801f, 0.86077714f, 0.8869355f, 0.86002564f, 0.81278664f,
      0.5097318f, 0.7297412f, 0.32111454f, 0.7177174f, 0.33929902f, 0.49160433f, 0.064810574f, 0.3692627f,
      0.23706353f, 0.3313396f, 0.18070674f, 0.05027789f, 0.53255826f, 0.8244896f, 0.9553747f, 0.7917771f,
      0.24083132f, 0.005495131f, 0.6896569f, 0.78015697f, 0.07074398f, 0.67929304f, 0.9227386f, 0.5302883f,
      0.19877058f, 0.90993816f, 0.71350795f, 0.8311006f, 0.16185725f, 0.79097277f, 0.15846318f, 0.99474716f,
      0.28815013f, 0.80128354f, 0.6001208f, 0.63250524f, 0.4233225f, 0.7053677f, 0.29161406f, 0.028710365f,
      0.30789846f, 0.8917693f, 0.36836517f, 0.6571592f, 0.3151368f, 0.8750746f, 0.7992451f, 0.6765068f,
      0.24441916f, 0.091435075f, 0.5188247f, 0.20667112f, 0.9110969f, 0.019512117f, 0.72343415f, 0.998457f,
      0.7504142f, 0.6704894f, 0.01892668f, 0.9809466f, 0.41447622f, 0.032795787f, 0.9935814f, 0.29653466f,
      0.4646262f, 0.95763975f, 0.15339965f, 0.14625502f, 0.58130866f, 0.43307304f, 0.6151709f, 0.08064735f,
      0.5149533f, 0.27762014f, 0.25419557f, 0.04218155f, 0.7651092f, 0.59631824f, 0.077278376f, 0.89677596f,
      0.6508104f, 0.5927816f, 0.2064318f, 0.57540226f, 0.9817701f, 0.84294224f, 0.11056489f, 0.9564106f,
      0.5387549f, 0.74048257f, 0.88833815f, 0.9262546f, 0.11023259f, 0.93783194f, 0.16041255f, 0.53748304f,
      0.1506182f, 0.39038336f, 0.47727865f, 0.44018233f, 0.42101204f, 0.53943527f, 0.99320936f, 0.79050577f,
      0.77973497f, 0.7001237f, 0.88709056f, 0.4769255f, 0.5397561f, 0.60289854f, 0.06393474f, 0.09722155f,
      0.5613007f, 0.30437487f, 0.49082512f, 0.3852706f, 0.5778314f, 0.8253078f, 0.33417904f, 0.9004303f,
      0.8947809f, 0.11625093f, 0.11388689f, 0.09546256f, 0.22598988f, 0.30536187f, 0.46236527f, 0.3784039f,
      0.24737573f, 0.3411532f, 0.31912774f, 0.9905191f, 0.31468558f, 0.14199954f, 0.7078488f, 0.47111923f,
      0.882782f, 0.8124163f, 0.9593644f, 0.13382024f, 0.8214317f, 0.9196194f, 0.25308424f, 0.95958996f};
  const std::vector<float> fc1_experts_bias = {
      0.8748215f, 0.5054756f, 0.74107623f, 0.32518923f, 0.0639081f, 0.62639004f, 0.64906263f, 0.17322052f,
      0.7424998f, 0.07288867f, 0.93031204f, 0.9841952f, 0.6361292f, 0.18628561f, 0.7433356f, 0.5852079f,
      0.6359594f, 0.66432667f, 0.88067776f, 0.28508204f, 0.38752747f, 0.63635296f, 0.55448055f, 0.9031888f,
      0.23738074f, 0.48179168f, 0.5934266f, 0.3672055f, 0.84085834f, 0.5546908f, 0.03788501f, 0.44583207f,
      0.27322155f, 0.5485856f, 0.44189203f, 0.00403291f, 0.40888733f, 0.45211035f, 0.35256076f, 0.9593902f,
      0.39090043f, 0.8212086f, 0.62385887f, 0.07793343f, 0.61749303f, 0.9143678f, 0.17294967f, 0.17681253f,
      0.9894245f, 0.901755f, 0.221053f, 0.8008725f, 0.43603396f, 0.007035315f, 0.5375667f, 0.661547f,
      0.35001957f, 0.67394173f, 0.072449565f, 0.84650797f, 0.92626715f, 0.77573335f, 0.58474565f, 0.66467446f};
  const std::vector<float> fc2_experts_bias = {
      0.13822609f, 0.3750633f, 0.45226622f, 0.22175694f, 0.13068998f, 0.8363088f, 0.8393226f, 0.045905888f,
      0.65910596f, 0.7034011f, 0.97498417f, 0.78927684f, 0.95966834f, 0.33630514f, 0.8501932f, 0.9067007f,
      0.027835965f, 0.09864664f, 0.6012027f, 0.7730189f, 0.25159347f, 0.55506724f, 0.49927413f, 0.62655383f,
      0.23132521f, 0.7820195f, 0.8325047f, 0.15307087f, 0.5048437f, 0.5013873f, 0.66055787f, 0.96579224f};
  const std::vector<float> output = {
      1.3775184f, 2.0985768f, 2.091839f, 2.9706357f, 1.9404914f, 1.9915576f, 2.3302228f, 2.3702593f,
      0.51896286f, 0.7936432f, 0.9944805f, 1.3225251f, 0.73894113f, 0.87975955f, 1.0468717f, 1.1585085f,
      0.012911659f, 0.045757107f, 0.27884653f, 0.3585817f, 0.116771236f, 0.25755364f, 0.23161705f, 0.2906256f,
      4.8571277f, 5.649453f, 5.485141f, 5.306299f, 4.767025f, 6.9010167f, 5.3520975f, 6.711155f};

  RunMoETest(input,
             router_probs,
             fc1_experts_weights,
             fc2_experts_weights,
             fc1_experts_bias,
             fc2_experts_bias,
             output,
             num_rows,
             num_experts,
             hidden_size,
             inter_size,
             "relu");
}

}  // namespace test
}  // namespace onnxruntime

#endif

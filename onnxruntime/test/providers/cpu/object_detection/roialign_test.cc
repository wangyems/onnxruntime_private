// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(RoiAlignTest, AvgModePositive) {
  OpTester test("RoiAlign", 10);
  test.AddAttribute<int64_t>("output_height", 3);
  test.AddAttribute<int64_t>("output_width", 4);
  test.AddAttribute<int64_t>("sampling_ratio", 2);
  test.AddAttribute<float>("spatial_scale", 1.0f / 16.0f);

  constexpr int N = 1;
  constexpr int C = 3;
  constexpr int H = 5;
  constexpr int W = 5;

 std::vector<float> rois{0.,7.,5.,7.,5.,0.,-15.,-15.,-15.,-15.,0.,-10.,21.,-10.,21.,0.,13.,8.,13.,8.,0.,-14.,19.,-14.,19.};
  test.AddInput<float>("X", {N, C, H, W}, {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,
                                           25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.,
                                           47.,48.,49.,50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,
                                           69.,70.,71.,72.,73.,74.});
  test.AddInput<float>("rois", {5, 4}, {7.,5.,7.,5.,-15.,-15.,-15.,-15.,-10.,21.,-10.,21.,13.,8.,13.,8.,-14.,19.,-14.,19.});
  test.AddInput<int64_t>("batch_indices", {5}, {0, 0, 0, 0, 0});
  test.AddOutput<float>("Y", {5,3,3,4}, {2.95833f,3.20833f,3.45833f,3.70833f,4.625f,4.875f,5.125f,5.375f,
                                         6.29167f,6.54167f,6.79167f,7.04167f,27.9583f,28.2083f,28.4583f,
                                         28.7083f,29.625f,29.875f,30.125f,30.375f,31.2917f,31.5417f,31.7917f,
                                         32.0417f,52.9583f,53.2083f,53.4583f,53.7083f,54.625f,54.875f,55.125f,
                                         55.375f,56.2917f,56.5417f,56.7917f,57.0417f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                                         25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,
                                         50.f,50.f,50.f,7.39583f,7.39583f,7.42708f,7.64583f,9.0625f,9.0625f,9.09375f,
                                         9.3125f,10.7292f,10.7292f,10.7604f,10.9792f,32.3958f,32.3958f,32.4271f,
                                         32.6458f,34.0625f,34.0625f,34.0938f,34.3125f,35.7292f,35.7292f,35.7604f,
                                         35.9792f,57.3958f,57.3958f,57.4271f,57.6458f,59.0625f,59.0625f,59.0938f,
                                         59.3125f,60.7292f,60.7292f,60.7604f,60.9792f,4.27083f,4.52083f,4.77083f,
                                         5.02083f,5.9375f,6.1875f,6.4375f,6.6875f,7.60417f,7.85417f,8.10417f,8.35417f,
                                         29.2708f,29.5208f,29.7708f,30.0208f,30.9375f,31.1875f,31.4375f,31.6875f,32.6042f,
                                         32.8542f,33.1042f,33.3542f,54.2708f,54.5208f,54.7708f,55.0208f,55.9375f,56.1875f,
                                         56.4375f,56.6875f,57.6042f,57.8542f,58.1042f,58.3542f,6.77083f,6.77083f,6.77083f,
                                         6.80208f,8.4375f,8.4375f,8.4375f,8.46875f,10.1042f,10.1042f,10.1042f,10.1354f,31.7708f,
                                         31.7708f,31.7708f,31.8021f,33.4375f,33.4375f,33.4375f,33.4688f,35.1042f,35.1042f,35.1042f,
                                         35.1354f,56.7708f,56.7708f,56.7708f,56.8021f,58.4375f,58.4375f,58.4375f,58.4688f,60.1042f,
                                         60.1042f,60.1042f,60.1354f});

  test.Run();
}

TEST(RoiAlignTest, OnnxTest) {
  OpTester test("RoiAlign", 10);
  test.AddAttribute<int64_t>("output_height", 5);
  test.AddAttribute<int64_t>("output_width", 5);
  test.AddAttribute<int64_t>("sampling_ratio", 2);
  test.AddAttribute<float>("spatial_scale", 1.0f);

  constexpr int N = 1;
  constexpr int C = 1;
  constexpr int H = 10;
  constexpr int W = 10;

  test.AddInput<float>("X", {N, C, H, W}, {
                    0.2764f, 0.7150f, 0.1958f, 0.3416f, 0.4638f, 0.0259f, 0.2963f, 0.6518f, 0.4856f, 0.7250f,
                    0.9637f, 0.0895f, 0.2919f, 0.6753f, 0.0234f, 0.6132f, 0.8085f, 0.5324f, 0.8992f, 0.4467f,
                    0.3265f, 0.8479f, 0.9698f, 0.2471f, 0.9336f, 0.1878f, 0.4766f, 0.4308f, 0.3400f, 0.2162f,
                    0.0206f, 0.1720f, 0.2155f, 0.4394f, 0.0653f, 0.3406f, 0.7724f, 0.3921f, 0.2541f, 0.5799f,
                    0.4062f, 0.2194f, 0.4473f, 0.4687f, 0.7109f, 0.9327f, 0.9815f, 0.6320f, 0.1728f, 0.6119f,
                    0.3097f, 0.1283f, 0.4984f, 0.5068f, 0.4279f, 0.0173f, 0.4388f, 0.0430f, 0.4671f, 0.7119f,
                    0.1011f, 0.8477f, 0.4726f, 0.1777f, 0.9923f, 0.4042f, 0.1869f, 0.7795f, 0.9946f, 0.9689f,
                    0.1366f, 0.3671f, 0.7011f, 0.6234f, 0.9867f, 0.5585f, 0.6985f, 0.5609f, 0.8788f, 0.9928f,
                    0.5697f, 0.8511f, 0.6711f, 0.9406f, 0.8751f, 0.7496f, 0.1650f, 0.1049f, 0.1559f, 0.2514f,
                    0.7012f, 0.4056f, 0.7879f, 0.3461f, 0.0415f, 0.2998f, 0.5094f, 0.3727f, 0.5482f, 0.0502f,});
  test.AddInput<float>("rois", {3, 4}, {0., 0., 9., 9., 0., 5., 4., 9., 5., 5., 9., 9.});
  test.AddInput<int64_t>("batch_indices", {3}, {0, 0, 0});
  test.AddOutput<float>("Y", {3,1,5,5}, {
                0.4664f, 0.4466f, 0.3405f, 0.5688f, 0.6068f,
                0.3714f, 0.4296f, 0.3835f, 0.5562f, 0.3510f,
                0.2768f, 0.4883f, 0.5222f, 0.5528f, 0.4171f,
                0.4713f, 0.4844f, 0.6904f, 0.4920f, 0.8774f,
                0.6239f, 0.7125f, 0.6289f, 0.3355f, 0.3495f,

                0.3022f, 0.4305f, 0.4696f, 0.3978f, 0.5423f,
                0.3656f, 0.7050f, 0.5165f, 0.3172f, 0.7015f,
                0.2912f, 0.5059f, 0.6476f, 0.6235f, 0.8299f,
                0.5916f, 0.7389f, 0.7048f, 0.8372f, 0.8893f,
                0.6227f, 0.6153f, 0.7097f, 0.6154f, 0.4585f,

                0.2384f, 0.3379f, 0.3717f, 0.6100f, 0.7601f,
                0.3767f, 0.3785f, 0.7147f, 0.9243f, 0.9727f,
                0.5749f, 0.5826f, 0.5709f, 0.7619f, 0.8770f,
                0.5355f, 0.2566f, 0.2141f, 0.2796f, 0.3600f,
                0.4365f, 0.3504f, 0.2887f, 0.3661f, 0.2349f,
    });

  test.Run();
}

TEST(RoiAlignTest, MaxModePositive) {
  OpTester test("RoiAlign", 10);
  test.AddAttribute<std::string>("mode", "max");
  test.AddAttribute<int64_t>("output_height", 3);
  test.AddAttribute<int64_t>("output_width", 4);
  test.AddAttribute<int64_t>("sampling_ratio", 2);
  test.AddAttribute<float>("spatial_scale", 1.0f / 16.0f);

  constexpr int N = 1;
  constexpr int C = 3;
  constexpr int H = 5;
  constexpr int W = 5;

 std::vector<float> rois{0.,7.,5.,7.,5.,0.,-15.,-15.,-15.,-15.,0.,-10.,21.,-10.,21.,0.,13.,8.,13.,8.,0.,-14.,19.,-14.,19.};
  test.AddInput<float>("X", {N, C, H, W}, {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,
                                           25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.,
                                           47.,48.,49.,50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,
                                           69.,70.,71.,72.,73.,74.});
  test.AddInput<float>("rois", {5, 4}, {7.,5.,7.,5.,-15.,-15.,-15.,-15.,-10.,21.,-10.,21.,13.,8.,13.,8.,-14.,19.,-14.,19.});
  test.AddInput<int64_t>("batch_indices", {5}, {0, 0, 0, 0, 0});
  test.AddOutput<float>("Y", {5,3,3,4}, {2.10938f,2.95313f,3.375f,2.53125f,3.35938f,4.70313f,5.375f,4.03125f,3.51563f,4.92188f,5.625f,
                                         4.21875f,10.8984f,15.2578f,17.4375f,13.0781f,17.3568f,24.2995f,27.7708f,20.8281f,18.1641f,
                                         25.4297f,29.0625f,21.7969f,19.6875f,27.5625f,31.5f,23.625f,31.3542f,43.8958f,50.1667f,37.625f,
                                         32.8125f,45.9375f,52.5f,39.375f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,
                                         25.f,25.f,25.f,25.f,25.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,5.625f,5.625f,5.625f,4.57031f,
                                         8.95833f,8.95833f,8.95833f,7.27865f,9.375f,9.375f,9.375f,7.61719f,19.6875f,19.6875f,19.6875f,15.9961f,
                                         31.3542f,31.3542f,31.3542f,25.4753f,32.8125f,32.8125f,32.8125f,26.6602f,33.75f,33.75f,33.75f,27.4219f,
                                         53.75f,53.75f,53.75f,43.6719f,56.25f,56.25f,56.25f,45.7031f,4.5f,3.9375f,2.8125f,3.9375f,5.5f,4.8125f,
                                         3.4375f,4.8125f,4.58333f,4.01042f,2.86458f,3.9375f,23.25f,20.3438f,14.5313f,18.f,28.4167f,24.86458f,
                                         17.76042f,22.f,23.25f,20.3437f,14.5312f,18.f,42.f,36.75f,26.25f,32.0625f,51.3333f,44.9167f,32.08333f,39.1875f,
                                         42.f,36.75f,26.25f,32.0625f,4.375f,4.375f,4.375f,4.375f,7.70833f,7.70833f,7.70833f,7.70833f,9.375f,
                                         9.375f,9.375f,9.375f,21.875f,21.875f,21.875f,21.875f,26.9792f,26.9792f,26.9792f,26.9792f,32.8125f,
                                         32.8125f,32.8125f,32.8125f,40.1042f,40.1042f,40.1042f,40.1042f,46.25f,46.25f,46.25f,46.25f,56.25f,56.25f,
                                         56.25f,56.25f});

  test.Run();
}

TEST(RoiAlignTest, AvgModeNegativeInvalidMode) {
  OpTester test("RoiAlign", 10);
  test.AddAttribute<std::string>("mode", "foobar"); // <-- failure condition
  test.AddAttribute<int64_t>("output_height", 3);
  test.AddAttribute<int64_t>("output_width", 4);
  test.AddAttribute<int64_t>("sampling_ratio", -2);
  test.AddAttribute<float>("spatial_scale", 1.0f / 16.0f);

  constexpr int N = 1;
  constexpr int C = 3;
  constexpr int H = 5;
  constexpr int W = 5;

 std::vector<float> rois{0.,7.,5.,7.,5.,0.,-15.,-15.,-15.,-15.,0.,-10.,21.,-10.,21.,0.,13.,8.,13.,8.,0.,-14.,19.,-14.,19.};
  test.AddInput<float>("X", {N, C, H, W}, {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,
                                           25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.,
                                           47.,48.,49.,50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,
                                           69.,70.,71.,72.,73.,74.});
  test.AddInput<float>("rois", {5, 4}, {7.,5.,7.,5.,-15.,-15.,-15.,-15.,-10.,21.,-10.,21.,13.,8.,13.,8.,-14.,19.,-14.,19.});
  test.AddInput<int64_t>("batch_indices", {5}, {0, 0, 0, 0, 0});
  test.AddOutput<float>("Y", {5,3,3,4}, {2.95833f,3.20833f,3.45833f,3.70833f,4.625f,4.875f,5.125f,5.375f,
                                         6.29167f,6.54167f,6.79167f,7.04167f,27.9583f,28.2083f,28.4583f,
                                         28.7083f,29.625f,29.875f,30.125f,30.375f,31.2917f,31.5417f,31.7917f,
                                         32.0417f,52.9583f,53.2083f,53.4583f,53.7083f,54.625f,54.875f,55.125f,
                                         55.375f,56.2917f,56.5417f,56.7917f,57.0417f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                                         25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,
                                         50.f,50.f,50.f,7.39583f,7.39583f,7.42708f,7.64583f,9.0625f,9.0625f,9.09375f,
                                         9.3125f,10.7292f,10.7292f,10.7604f,10.9792f,32.3958f,32.3958f,32.4271f,
                                         32.6458f,34.0625f,34.0625f,34.0938f,34.3125f,35.7292f,35.7292f,35.7604f,
                                         35.9792f,57.3958f,57.3958f,57.4271f,57.6458f,59.0625f,59.0625f,59.0938f,
                                         59.3125f,60.7292f,60.7292f,60.7604f,60.9792f,4.27083f,4.52083f,4.77083f,
                                         5.02083f,5.9375f,6.1875f,6.4375f,6.6875f,7.60417f,7.85417f,8.10417f,8.35417f,
                                         29.2708f,29.5208f,29.7708f,30.0208f,30.9375f,31.1875f,31.4375f,31.6875f,32.6042f,
                                         32.8542f,33.1042f,33.3542f,54.2708f,54.5208f,54.7708f,55.0208f,55.9375f,56.1875f,
                                         56.4375f,56.6875f,57.6042f,57.8542f,58.1042f,58.3542f,6.77083f,6.77083f,6.77083f,
                                         6.80208f,8.4375f,8.4375f,8.4375f,8.46875f,10.1042f,10.1042f,10.1042f,10.1354f,31.7708f,
                                         31.7708f,31.7708f,31.8021f,33.4375f,33.4375f,33.4375f,33.4688f,35.1042f,35.1042f,35.1042f,
                                         35.1354f,56.7708f,56.7708f,56.7708f,56.8021f,58.4375f,58.4375f,58.4375f,58.4688f,60.1042f,
                                         60.1042f,60.1042f,60.1354f});

  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid mode");
}

TEST(RoiAlignTest, AvgModeNegativeSamplingRatio) {
  OpTester test("RoiAlign", 10);
  test.AddAttribute<int64_t>("output_height", 3);
  test.AddAttribute<int64_t>("output_width", 4);
  test.AddAttribute<int64_t>("sampling_ratio", -2); // <-- failure condition
  test.AddAttribute<float>("spatial_scale", 1.0f / 16.0f);

  constexpr int N = 1;
  constexpr int C = 3;
  constexpr int H = 5;
  constexpr int W = 5;

  test.AddInput<float>("X", {N, C, H, W}, {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,
                                           25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.,
                                           47.,48.,49.,50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,
                                           69.,70.,71.,72.,73.,74.});
  test.AddInput<float>("rois", {5, 4}, {7.,5.,7.,5.,-15.,-15.,-15.,-15.,-10.,21.,-10.,21.,13.,8.,13.,8.,-14.,19.,-14.,19.});
  test.AddInput<int64_t>("batch_indices", {5}, {0, 0, 0, 0, 0});
  test.AddOutput<float>("Y", {5,3,3,4}, {2.95833f,3.20833f,3.45833f,3.70833f,4.625f,4.875f,5.125f,5.375f,
                                         6.29167f,6.54167f,6.79167f,7.04167f,27.9583f,28.2083f,28.4583f,
                                         28.7083f,29.625f,29.875f,30.125f,30.375f,31.2917f,31.5417f,31.7917f,
                                         32.0417f,52.9583f,53.2083f,53.4583f,53.7083f,54.625f,54.875f,55.125f,
                                         55.375f,56.2917f,56.5417f,56.7917f,57.0417f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                                         25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,
                                         50.f,50.f,50.f,7.39583f,7.39583f,7.42708f,7.64583f,9.0625f,9.0625f,9.09375f,
                                         9.3125f,10.7292f,10.7292f,10.7604f,10.9792f,32.3958f,32.3958f,32.4271f,
                                         32.6458f,34.0625f,34.0625f,34.0938f,34.3125f,35.7292f,35.7292f,35.7604f,
                                         35.9792f,57.3958f,57.3958f,57.4271f,57.6458f,59.0625f,59.0625f,59.0938f,
                                         59.3125f,60.7292f,60.7292f,60.7604f,60.9792f,4.27083f,4.52083f,4.77083f,
                                         5.02083f,5.9375f,6.1875f,6.4375f,6.6875f,7.60417f,7.85417f,8.10417f,8.35417f,
                                         29.2708f,29.5208f,29.7708f,30.0208f,30.9375f,31.1875f,31.4375f,31.6875f,32.6042f,
                                         32.8542f,33.1042f,33.3542f,54.2708f,54.5208f,54.7708f,55.0208f,55.9375f,56.1875f,
                                         56.4375f,56.6875f,57.6042f,57.8542f,58.1042f,58.3542f,6.77083f,6.77083f,6.77083f,
                                         6.80208f,8.4375f,8.4375f,8.4375f,8.46875f,10.1042f,10.1042f,10.1042f,10.1354f,31.7708f,
                                         31.7708f,31.7708f,31.8021f,33.4375f,33.4375f,33.4375f,33.4688f,35.1042f,35.1042f,35.1042f,
                                         35.1354f,56.7708f,56.7708f,56.7708f,56.8021f,58.4375f,58.4375f,58.4375f,58.4688f,60.1042f,
                                         60.1042f,60.1042f,60.1354f});

  test.Run(OpTester::ExpectResult::kExpectFailure, "Sampling ratio should be >=0");
}

TEST(RoiAlignTest, AvgModeNegativeInvalidNumRoiDims) {
  OpTester test("RoiAlign", 10);
  test.AddAttribute<int64_t>("output_height", 3);
  test.AddAttribute<int64_t>("output_width", 4);
  test.AddAttribute<int64_t>("sampling_ratio", 2);
  test.AddAttribute<float>("spatial_scale", 1.0f / 16.0f);

  constexpr int N = 1;
  constexpr int C = 3;
  constexpr int H = 5;
  constexpr int W = 5;

  std::vector<float> rois{0.,7.,5.,7.,5.,0.,-15.,-15.,-15.,-15.,0.,-10.,21.,-10.,21.,0.,13.,8.,13.,8.,0.,-14.,19.,-14.,19.};
  test.AddInput<float>("X", {N, C, H, W}, {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,
                                           25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.,
                                           47.,48.,49.,50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,
                                           69.,70.,71.,72.,73.,74.});
  test.AddInput<float>("rois", {5, 4, 1}, {7.,5.,7.,5.,-15.,-15.,-15.,-15.,-10.,21.,-10.,21.,13.,8.,13.,8.,-14.,19.,-14.,19.}); // <-- failure condition
  test.AddInput<int64_t>("batch_indices", {5}, {0, 0, 0, 0, 0});
  test.AddOutput<float>("Y", {5,3,3,4}, {2.95833f,3.20833f,3.45833f,3.70833f,4.625f,4.875f,5.125f,5.375f,
                                         6.29167f,6.54167f,6.79167f,7.04167f,27.9583f,28.2083f,28.4583f,
                                         28.7083f,29.625f,29.875f,30.125f,30.375f,31.2917f,31.5417f,31.7917f,
                                         32.0417f,52.9583f,53.2083f,53.4583f,53.7083f,54.625f,54.875f,55.125f,
                                         55.375f,56.2917f,56.5417f,56.7917f,57.0417f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                                         25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,
                                         50.f,50.f,50.f,7.39583f,7.39583f,7.42708f,7.64583f,9.0625f,9.0625f,9.09375f,
                                         9.3125f,10.7292f,10.7292f,10.7604f,10.9792f,32.3958f,32.3958f,32.4271f,
                                         32.6458f,34.0625f,34.0625f,34.0938f,34.3125f,35.7292f,35.7292f,35.7604f,
                                         35.9792f,57.3958f,57.3958f,57.4271f,57.6458f,59.0625f,59.0625f,59.0938f,
                                         59.3125f,60.7292f,60.7292f,60.7604f,60.9792f,4.27083f,4.52083f,4.77083f,
                                         5.02083f,5.9375f,6.1875f,6.4375f,6.6875f,7.60417f,7.85417f,8.10417f,8.35417f,
                                         29.2708f,29.5208f,29.7708f,30.0208f,30.9375f,31.1875f,31.4375f,31.6875f,32.6042f,
                                         32.8542f,33.1042f,33.3542f,54.2708f,54.5208f,54.7708f,55.0208f,55.9375f,56.1875f,
                                         56.4375f,56.6875f,57.6042f,57.8542f,58.1042f,58.3542f,6.77083f,6.77083f,6.77083f,
                                         6.80208f,8.4375f,8.4375f,8.4375f,8.46875f,10.1042f,10.1042f,10.1042f,10.1354f,31.7708f,
                                         31.7708f,31.7708f,31.8021f,33.4375f,33.4375f,33.4375f,33.4688f,35.1042f,35.1042f,35.1042f,
                                         35.1354f,56.7708f,56.7708f,56.7708f,56.8021f,58.4375f,58.4375f,58.4375f,58.4688f,60.1042f,
                                         60.1042f,60.1042f,60.1354f});

  test.Run(OpTester::ExpectResult::kExpectFailure, "[ShapeInferenceError] Input 1 expected to have rank 2 but has rank 3");
}

TEST(RoiAlignTest, AvgModeNegativeInvalidSecondRoiDims) {
  OpTester test("RoiAlign", 10);
  test.AddAttribute<int64_t>("output_height", 3);
  test.AddAttribute<int64_t>("output_width", 4);
  test.AddAttribute<int64_t>("sampling_ratio", 2);
  test.AddAttribute<float>("spatial_scale", 1.0f / 16.0f);

  constexpr int N = 1;
  constexpr int C = 3;
  constexpr int H = 5;
  constexpr int W = 5;

 test.AddInput<float>("X", {N, C, H, W}, {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,
                                          25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.,
                                          47.,48.,49.,50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,
                                          69.,70.,71.,72.,73.,74.});
 test.AddInput<float>("rois", {5, 3}, {7.,5.,7.,5.,-15.,-15.,-15.,-15.,-10.,21.,-10.,21.,13.,8.,13.}); // <-- failure condition
 test.AddInput<int64_t>("batch_indices", {5}, {0, 0, 0, 0, 0});
 test.AddOutput<float>("Y", {5,3,3,4}, {2.95833f,3.20833f,3.45833f,3.70833f,4.625f,4.875f,5.125f,5.375f,
                                        6.29167f,6.54167f,6.79167f,7.04167f,27.9583f,28.2083f,28.4583f,
                                        28.7083f,29.625f,29.875f,30.125f,30.375f,31.2917f,31.5417f,31.7917f,
                                        32.0417f,52.9583f,53.2083f,53.4583f,53.7083f,54.625f,54.875f,55.125f,
                                        55.375f,56.2917f,56.5417f,56.7917f,57.0417f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                                        25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,
                                        50.f,50.f,50.f,7.39583f,7.39583f,7.42708f,7.64583f,9.0625f,9.0625f,9.09375f,
                                        9.3125f,10.7292f,10.7292f,10.7604f,10.9792f,32.3958f,32.3958f,32.4271f,
                                        32.6458f,34.0625f,34.0625f,34.0938f,34.3125f,35.7292f,35.7292f,35.7604f,
                                        35.9792f,57.3958f,57.3958f,57.4271f,57.6458f,59.0625f,59.0625f,59.0938f,
                                        59.3125f,60.7292f,60.7292f,60.7604f,60.9792f,4.27083f,4.52083f,4.77083f,
                                        5.02083f,5.9375f,6.1875f,6.4375f,6.6875f,7.60417f,7.85417f,8.10417f,8.35417f,
                                        29.2708f,29.5208f,29.7708f,30.0208f,30.9375f,31.1875f,31.4375f,31.6875f,32.6042f,
                                        32.8542f,33.1042f,33.3542f,54.2708f,54.5208f,54.7708f,55.0208f,55.9375f,56.1875f,
                                        56.4375f,56.6875f,57.6042f,57.8542f,58.1042f,58.3542f,6.77083f,6.77083f,6.77083f,
                                        6.80208f,8.4375f,8.4375f,8.4375f,8.46875f,10.1042f,10.1042f,10.1042f,10.1354f,31.7708f,
                                        31.7708f,31.7708f,31.8021f,33.4375f,33.4375f,33.4375f,33.4688f,35.1042f,35.1042f,35.1042f,
                                        35.1354f,56.7708f,56.7708f,56.7708f,56.8021f,58.4375f,58.4375f,58.4375f,58.4688f,60.1042f,
                                        60.1042f,60.1042f,60.1354f});

 test.Run(OpTester::ExpectResult::kExpectFailure, "Second dimension for rois should be exactly 4");
}

TEST(RoiAlignTest, MismatchNumRois) {
  OpTester test("RoiAlign", 10);
  test.AddAttribute<int64_t>("output_height", 3);
  test.AddAttribute<int64_t>("output_width", 4);
  test.AddAttribute<int64_t>("sampling_ratio", 2);
  test.AddAttribute<float>("spatial_scale", 1.0f / 16.0f);

  constexpr int N = 1;
  constexpr int C = 3;
  constexpr int H = 5;
  constexpr int W = 5;

 std::vector<float> rois{0.,7.,5.,7.,5.,0.,-15.,-15.,-15.,-15.,0.,-10.,21.,-10.,21.,0.,13.,8.,13.,8.,0.,-14.,19.,-14.,19.};
  test.AddInput<float>("X", {N, C, H, W}, {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,
                                           25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.,
                                           47.,48.,49.,50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,
                                           69.,70.,71.,72.,73.,74.});
  test.AddInput<float>("rois", {5, 4}, {7.,5.,7.,5.,-15.,-15.,-15.,-15.,-10.,21.,-10.,21.,13.,8.,13.,8.,-14.,19.,-14.,19.});
  test.AddInput<int64_t>("batch_indices", {4}, {0, 0, 0, 0}); // <-- failure condition
  test.AddOutput<float>("Y", {5,3,3,4}, {2.95833f,3.20833f,3.45833f,3.70833f,4.625f,4.875f,5.125f,5.375f,
                                         6.29167f,6.54167f,6.79167f,7.04167f,27.9583f,28.2083f,28.4583f,
                                         28.7083f,29.625f,29.875f,30.125f,30.375f,31.2917f,31.5417f,31.7917f,
                                         32.0417f,52.9583f,53.2083f,53.4583f,53.7083f,54.625f,54.875f,55.125f,
                                         55.375f,56.2917f,56.5417f,56.7917f,57.0417f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
                                         25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,25.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,50.f,
                                         50.f,50.f,50.f,7.39583f,7.39583f,7.42708f,7.64583f,9.0625f,9.0625f,9.09375f,
                                         9.3125f,10.7292f,10.7292f,10.7604f,10.9792f,32.3958f,32.3958f,32.4271f,
                                         32.6458f,34.0625f,34.0625f,34.0938f,34.3125f,35.7292f,35.7292f,35.7604f,
                                         35.9792f,57.3958f,57.3958f,57.4271f,57.6458f,59.0625f,59.0625f,59.0938f,
                                         59.3125f,60.7292f,60.7292f,60.7604f,60.9792f,4.27083f,4.52083f,4.77083f,
                                         5.02083f,5.9375f,6.1875f,6.4375f,6.6875f,7.60417f,7.85417f,8.10417f,8.35417f,
                                         29.2708f,29.5208f,29.7708f,30.0208f,30.9375f,31.1875f,31.4375f,31.6875f,32.6042f,
                                         32.8542f,33.1042f,33.3542f,54.2708f,54.5208f,54.7708f,55.0208f,55.9375f,56.1875f,
                                         56.4375f,56.6875f,57.6042f,57.8542f,58.1042f,58.3542f,6.77083f,6.77083f,6.77083f,
                                         6.80208f,8.4375f,8.4375f,8.4375f,8.46875f,10.1042f,10.1042f,10.1042f,10.1354f,31.7708f,
                                         31.7708f,31.7708f,31.8021f,33.4375f,33.4375f,33.4375f,33.4688f,35.1042f,35.1042f,35.1042f,
                                         35.1354f,56.7708f,56.7708f,56.7708f,56.8021f,58.4375f,58.4375f,58.4375f,58.4688f,60.1042f,
                                         60.1042f,60.1042f,60.1354f});

  test.Run(OpTester::ExpectResult::kExpectFailure, "[ShapeInferenceError] Dimension mismatch in unification between 4 and 5");
}
}  // namespace test
}  // namespace onnxruntime

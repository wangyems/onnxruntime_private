#if 0
//
// Generated by Microsoft (R) HLSL Shader Compiler 10.1
//
//
// Buffer Definitions: 
//
// cbuffer cbCS
// {
//
//   uint height;                       // Offset:    0 Size:     4
//   uint width;                        // Offset:    4 Size:     4
//
// }
//
// Resource bind info for input
// {
//
//   float $Element;                    // Offset:    0 Size:     4
//
// }
//
//
// Resource Bindings:
//
// Name                                 Type  Format         Dim      HLSL Bind  Count
// ------------------------------ ---------- ------- ----------- -------------- ------
// input                             texture  struct         r/o             t0      1 
// output                                UAV  float4          2d             u0      1 
// cbCS                              cbuffer      NA          NA            cb0      1 
//
//
//
// Input signature:
//
// Name                 Index   Mask Register SysValue  Format   Used
// -------------------- ----- ------ -------- -------- ------- ------
// no Input
//
// Output signature:
//
// Name                 Index   Mask Register SysValue  Format   Used
// -------------------- ----- ------ -------- -------- ------- ------
// no Output
cs_5_0
dcl_globalFlags refactoringAllowed
dcl_constantbuffer CB0[1], immediateIndexed
dcl_resource_structured t0, 4
dcl_uav_typed_texture2d (float,float,float,float) u0
dcl_input vThreadID.xy
dcl_temps 1
dcl_thread_group 16, 4, 1
ult r0.xy, vThreadID.xyxx, cb0[0].yxyy
and r0.x, r0.y, r0.x
if_nz r0.x
  imul null, r0.x, cb0[0].y, cb0[0].x
  imad r0.y, cb0[0].y, vThreadID.y, vThreadID.x
  ld_structured_indexable(structured_buffer, stride=4)(mixed,mixed,mixed,mixed) r0.z, r0.y, l(0), t0.xxxx
  imad r0.w, cb0[0].x, cb0[0].y, r0.y
  ld_structured_indexable(structured_buffer, stride=4)(mixed,mixed,mixed,mixed) r0.w, r0.w, l(0), t0.xxxx
  ishl r0.x, r0.x, l(1)
  iadd r0.x, r0.x, r0.y
  ld_structured_indexable(structured_buffer, stride=4)(mixed,mixed,mixed,mixed) r0.x, r0.x, l(0), t0.xxxx
  mul r0.y, r0.w, l(0.002805)
  mad r0.y, r0.z, l(0.000834), r0.y
  mad r0.x, r0.x, l(0.000283), r0.y
  mov r0.yzw, l(0,0,0,0)
  store_uav_typed u0.xyzw, vThreadID.xyyy, r0.xyzw
endif 
ret 
// Approximately 18 instruction slots used
#endif

const BYTE g_csTensorRGB8ToSurfaceGRAY8[] =
{
     68,  88,  66,  67, 203,  21, 
     43, 103, 143, 172, 251, 111, 
     63,  74, 141, 225,  46, 231, 
    143,  14,   1,   0,   0,   0, 
    100,   5,   0,   0,   5,   0, 
      0,   0,  52,   0,   0,   0, 
     44,   2,   0,   0,  60,   2, 
      0,   0,  76,   2,   0,   0, 
    200,   4,   0,   0,  82,  68, 
     69,  70, 240,   1,   0,   0, 
      2,   0,   0,   0, 176,   0, 
      0,   0,   3,   0,   0,   0, 
     60,   0,   0,   0,   0,   5, 
     83,  67,   0,   1,   0,   0, 
    200,   1,   0,   0,  82,  68, 
     49,  49,  60,   0,   0,   0, 
     24,   0,   0,   0,  32,   0, 
      0,   0,  40,   0,   0,   0, 
     36,   0,   0,   0,  12,   0, 
      0,   0,   0,   0,   0,   0, 
    156,   0,   0,   0,   5,   0, 
      0,   0,   6,   0,   0,   0, 
      1,   0,   0,   0,   4,   0, 
      0,   0,   0,   0,   0,   0, 
      1,   0,   0,   0,   1,   0, 
      0,   0, 162,   0,   0,   0, 
      4,   0,   0,   0,   5,   0, 
      0,   0,   4,   0,   0,   0, 
    255, 255, 255, 255,   0,   0, 
      0,   0,   1,   0,   0,   0, 
     13,   0,   0,   0, 169,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   1,   0, 
      0,   0,   1,   0,   0,   0, 
    105, 110, 112, 117, 116,   0, 
    111, 117, 116, 112, 117, 116, 
      0,  99,  98,  67,  83,   0, 
    171, 171, 169,   0,   0,   0, 
      2,   0,   0,   0, 224,   0, 
      0,   0,  16,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0, 156,   0,   0,   0, 
      1,   0,   0,   0, 108,   1, 
      0,   0,   4,   0,   0,   0, 
      0,   0,   0,   0,   3,   0, 
      0,   0,  48,   1,   0,   0, 
      0,   0,   0,   0,   4,   0, 
      0,   0,   2,   0,   0,   0, 
     64,   1,   0,   0,   0,   0, 
      0,   0, 255, 255, 255, 255, 
      0,   0,   0,   0, 255, 255, 
    255, 255,   0,   0,   0,   0, 
    100,   1,   0,   0,   4,   0, 
      0,   0,   4,   0,   0,   0, 
      2,   0,   0,   0,  64,   1, 
      0,   0,   0,   0,   0,   0, 
    255, 255, 255, 255,   0,   0, 
      0,   0, 255, 255, 255, 255, 
      0,   0,   0,   0, 104, 101, 
    105, 103, 104, 116,   0, 100, 
    119, 111, 114, 100,   0, 171, 
    171, 171,   0,   0,  19,   0, 
      1,   0,   1,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,  55,   1, 
      0,   0, 119, 105, 100, 116, 
    104,   0, 171, 171, 148,   1, 
      0,   0,   0,   0,   0,   0, 
      4,   0,   0,   0,   2,   0, 
      0,   0, 164,   1,   0,   0, 
      0,   0,   0,   0, 255, 255, 
    255, 255,   0,   0,   0,   0, 
    255, 255, 255, 255,   0,   0, 
      0,   0,  36,  69, 108, 101, 
    109, 101, 110, 116,   0, 102, 
    108, 111,  97, 116,   0, 171, 
      0,   0,   3,   0,   1,   0, 
      1,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0, 157,   1,   0,   0, 
     77, 105,  99, 114, 111, 115, 
    111, 102, 116,  32,  40,  82, 
     41,  32,  72,  76,  83,  76, 
     32,  83, 104,  97, 100, 101, 
    114,  32,  67, 111, 109, 112, 
    105, 108, 101, 114,  32,  49, 
     48,  46,  49,   0,  73,  83, 
     71,  78,   8,   0,   0,   0, 
      0,   0,   0,   0,   8,   0, 
      0,   0,  79,  83,  71,  78, 
      8,   0,   0,   0,   0,   0, 
      0,   0,   8,   0,   0,   0, 
     83,  72,  69,  88, 116,   2, 
      0,   0,  80,   0,   5,   0, 
    157,   0,   0,   0, 106,   8, 
      0,   1,  89,   0,   0,   4, 
     70, 142,  32,   0,   0,   0, 
      0,   0,   1,   0,   0,   0, 
    162,   0,   0,   4,   0, 112, 
     16,   0,   0,   0,   0,   0, 
      4,   0,   0,   0, 156,  24, 
      0,   4,   0, 224,  17,   0, 
      0,   0,   0,   0,  85,  85, 
      0,   0,  95,   0,   0,   2, 
     50,   0,   2,   0, 104,   0, 
      0,   2,   1,   0,   0,   0, 
    155,   0,   0,   4,  16,   0, 
      0,   0,   4,   0,   0,   0, 
      1,   0,   0,   0,  79,   0, 
      0,   7,  50,   0,  16,   0, 
      0,   0,   0,   0,  70,   0, 
      2,   0,  22, 133,  32,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   1,   0,   0,   7, 
     18,   0,  16,   0,   0,   0, 
      0,   0,  26,   0,  16,   0, 
      0,   0,   0,   0,  10,   0, 
     16,   0,   0,   0,   0,   0, 
     31,   0,   4,   3,  10,   0, 
     16,   0,   0,   0,   0,   0, 
     38,   0,   0,  10,   0, 208, 
      0,   0,  18,   0,  16,   0, 
      0,   0,   0,   0,  26, 128, 
     32,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,  10, 128, 
     32,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,  35,   0, 
      0,   8,  34,   0,  16,   0, 
      0,   0,   0,   0,  26, 128, 
     32,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,  26,   0, 
      2,   0,  10,   0,   2,   0, 
    167,   0,   0, 139,   2,  35, 
      0, 128, 131, 153,  25,   0, 
     66,   0,  16,   0,   0,   0, 
      0,   0,  26,   0,  16,   0, 
      0,   0,   0,   0,   1,  64, 
      0,   0,   0,   0,   0,   0, 
      6, 112,  16,   0,   0,   0, 
      0,   0,  35,   0,   0,  11, 
    130,   0,  16,   0,   0,   0, 
      0,   0,  10, 128,  32,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,  26, 128,  32,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,  26,   0,  16,   0, 
      0,   0,   0,   0, 167,   0, 
      0, 139,   2,  35,   0, 128, 
    131, 153,  25,   0, 130,   0, 
     16,   0,   0,   0,   0,   0, 
     58,   0,  16,   0,   0,   0, 
      0,   0,   1,  64,   0,   0, 
      0,   0,   0,   0,   6, 112, 
     16,   0,   0,   0,   0,   0, 
     41,   0,   0,   7,  18,   0, 
     16,   0,   0,   0,   0,   0, 
     10,   0,  16,   0,   0,   0, 
      0,   0,   1,  64,   0,   0, 
      1,   0,   0,   0,  30,   0, 
      0,   7,  18,   0,  16,   0, 
      0,   0,   0,   0,  10,   0, 
     16,   0,   0,   0,   0,   0, 
     26,   0,  16,   0,   0,   0, 
      0,   0, 167,   0,   0, 139, 
      2,  35,   0, 128, 131, 153, 
     25,   0,  18,   0,  16,   0, 
      0,   0,   0,   0,  10,   0, 
     16,   0,   0,   0,   0,   0, 
      1,  64,   0,   0,   0,   0, 
      0,   0,   6, 112,  16,   0, 
      0,   0,   0,   0,  56,   0, 
      0,   7,  34,   0,  16,   0, 
      0,   0,   0,   0,  58,   0, 
     16,   0,   0,   0,   0,   0, 
      1,  64,   0,   0,  41, 207, 
     55,  59,  50,   0,   0,   9, 
     34,   0,  16,   0,   0,   0, 
      0,   0,  42,   0,  16,   0, 
      0,   0,   0,   0,   1,  64, 
      0,   0,  95, 142,  90,  58, 
     26,   0,  16,   0,   0,   0, 
      0,   0,  50,   0,   0,   9, 
     18,   0,  16,   0,   0,   0, 
      0,   0,  10,   0,  16,   0, 
      0,   0,   0,   0,   1,  64, 
      0,   0,  11, 114, 148,  57, 
     26,   0,  16,   0,   0,   0, 
      0,   0,  54,   0,   0,   8, 
    226,   0,  16,   0,   0,   0, 
      0,   0,   2,  64,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0, 164,   0, 
      0,   6, 242, 224,  17,   0, 
      0,   0,   0,   0,  70,   5, 
      2,   0,  70,  14,  16,   0, 
      0,   0,   0,   0,  21,   0, 
      0,   1,  62,   0,   0,   1, 
     83,  84,  65,  84, 148,   0, 
      0,   0,  18,   0,   0,   0, 
      1,   0,   0,   0,   0,   0, 
      0,   0,   1,   0,   0,   0, 
      3,   0,   0,   0,   5,   0, 
      0,   0,   2,   0,   0,   0, 
      1,   0,   0,   0,   1,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   3,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      1,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   1,   0,   0,   0
};

cbuffer SceneConstantBuffer : register( b0 )
{
    int M;
    int K;
    int N;
    int TILE_K;
}

static uint3 gl_LocalInvocationID = uint3(0, 0, 0);
static uint3 gl_GlobalInvocationID = uint3(0, 0, 0);

struct CS_INPUT
{
    uint3 dx_LocalInvocationID : SV_GroupThreadID;
    uint3 dx_GlobalInvocationID : SV_DispatchThreadID;
};

void initGLBuiltins(CS_INPUT input)
{
    gl_LocalInvocationID = input.dx_LocalInvocationID;
    gl_GlobalInvocationID = input.dx_GlobalInvocationID;
};

#ifdef USE_STRUCTURED_BUFFERS
#ifdef USE_VEC4
StructuredBuffer<float4> src0 : register(t0);
StructuredBuffer<float4> src1 : register(t1);
RWStructuredBuffer<float4> dst : register(u0);
  float4 mm_readA(int index) {
    return src0[index];
  }

  float4 mm_readB(int index) {
    return src1[index];
  }

  void mm_write(int index, float4 value) {
    dst[index] = value;
  }
#else
StructuredBuffer<float> src0 : register(t0);
StructuredBuffer<float> src1 : register(t1);
RWStructuredBuffer<float> dst : register(u0);

  float mm_readA(int index) {
    return src0[index];
  }

  float mm_readB(int index) {
    return src1[index];
  }

  void mm_write(int index, float value) {
    dst[index] = value;
  }
#endif  // USE_VEC4
#else
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
RWByteAddressBuffer dst : register(u0);

#ifdef USE_VEC4
float4 mm_readA(int index) {
    float4 result = asfloat(src0.Load4(4 * (index * 4)));
    return result;
}

float4 mm_readB(int index) {
    float4 result = asfloat(src1.Load4(4 * (index * 4)));
    return result;
}

void mm_write(int index, float4 value) {
    dst.Store4(4 * (index * 4), asuint(value));
}
#else
float mm_readA(int index) {
    float result = asfloat(src0.Load(4 * index));
    return result;
}

float mm_readB(int index) {
    float result = asfloat(src1.Load(4 * index));
    return result;
}

void mm_write(int index, float value) {
    dst.Store(4 * index, asuint(value));
}
#endif  // USE_VEC4
#endif  // USE_STRUCTURED_BUFFERS

[numthreads(128, 1, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int index = int(gl_GlobalInvocationID.x);
#ifdef USE_VEC4
	float4 result = mm_readA(index) + mm_readB(index);
#else
	float result = mm_readA(index) + mm_readB(index);
#endif  // USE_VEC4
	mm_write(index, result);
}

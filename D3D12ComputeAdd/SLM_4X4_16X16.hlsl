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
#ifdef USE_INT
int mm_readA(int index) {
    int result = asint(src0.Load(4 * index));
    return result;
}

int mm_readB(int index) {
    int result = asint(src1.Load(4 * index));
    return result;
}

void mm_write(int index, int value) {
    dst.Store(4 * index, asuint(value));
}

int tint_div(int lhs, int rhs) {
    return (lhs / (((rhs == 0) | ((lhs == -2147483648) & (rhs == -1))) ? 1 : rhs));
}
int tint_mod(int lhs, int rhs) {
    const int rhs_or_one = (((rhs == 0) | ((lhs == -2147483648) & (rhs == -1))) ? 1 : rhs);
    if (any(((uint((lhs | rhs_or_one)) & 2147483648u) != 0u))) {
        return (lhs - ((lhs / rhs_or_one) * rhs_or_one));
    }
    else {
        return (lhs % rhs_or_one);
    }
}
#else
uint mm_readA(int index) {
    uint result = asuint(src0.Load(4 * index));
    return result;
}

uint mm_readB(int index) {
    uint result = asuint(src1.Load(4 * index));
    return result;
}

void mm_write(int index, uint value) {
    dst.Store(4 * index, asuint(value));
}
uint tint_div(uint lhs, uint rhs) {
    return (lhs / ((rhs == 0u) ? 1u : rhs));
}

uint tint_mod(uint lhs, uint rhs) {
    return (lhs % ((rhs == 0u) ? 1u : rhs));
}
#endif  // USE_INT
#endif  // USE_VEC4
#endif  // USE_STRUCTURED_BUFFERS

[numthreads(64, 1, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int index = int(gl_GlobalInvocationID.x);
#ifdef USE_VEC4
	float4 result = mm_readA(index) + mm_readB(index);
#else
#ifdef USE_INT
    const int a = mm_readA(index);
    const int b = mm_readB(index);
    int c = 0;
    {
        for (int i = 1; (i < 200); i = (i + 1)) {
            c = (c + (tint_div(a, i) + tint_mod(a, i)));
            c = (c + (tint_div(b, i) + tint_mod(b, i)));
        }
    }
#else
    const uint a = mm_readA(index);
    const uint b = mm_readB(index);
    uint c = 0u;
    {
        for (uint i = 1u; (i < 200u); i = (i + 1u)) {
            c = (c + (tint_div(a, i) + tint_mod(a, i)));
            c = (c + (tint_div(b, i) + tint_mod(b, i)));
        }
    }
#endif  // USE_INT
#endif  // USE_VEC4
	mm_write(index, c);
}

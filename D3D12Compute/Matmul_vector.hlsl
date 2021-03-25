cbuffer SceneConstantBuffer : register( b0 )
{
    int M;
    int K;
    int N;
    int TILE_K;
}

static uint3 gl_WorkGroupID = uint3(0, 0, 0);
static uint3 gl_LocalInvocationID = uint3(0, 0, 0);
static uint3 gl_GlobalInvocationID = uint3(0, 0, 0);

struct CS_INPUT
{
    uint3 dx_WorkGroupID : SV_GroupID;
    uint3 dx_LocalInvocationID : SV_GroupThreadID;
    uint3 dx_GlobalInvocationID : SV_DispatchThreadID;
};

void initGLBuiltins(CS_INPUT input)
{
    gl_WorkGroupID = input.dx_WorkGroupID;
    gl_LocalInvocationID = input.dx_LocalInvocationID;
    gl_GlobalInvocationID = input.dx_GlobalInvocationID;
};

#ifdef USE_TEXTURE
Texture2D<float4> src0 : register(t0);
Texture2D<float4> src1 : register(t1);
RWTexture2D<float4> dst : register(u0);

float4 mm_readA(int row, int col) {
    if (row < M && col < K / 4)
    {
        return src0.Load(int3(col, row, 0));
    }
    else {
        return float4(0, 0, 0, 0);
    }
}

float4 mm_readB(int row, int col) {
    return src1.Load(int3(col, row, 0));
}

void mm_write(int row, int col, float4 value) {
    if (row < M && col < N / 4)
    {
        dst[uint2(col, row)] = value;
    }
}
#else
#ifdef USE_STRUCTURED_BUFFERS
StructuredBuffer<float> src0 : register(t0);
StructuredBuffer<float> src1 : register(t1);
RWStructuredBuffer<float> dst : register(u0);

float4 mm_readA(int row, int col) {
    if (row < M)
    {
        int index = row * K + col;
        float4 result = float4(src0[index],
            src0[index + 1],
            src0[index + 2],
            src0[index + 3]);
        return result;
    }
    else {
        return float4(0, 0, 0, 0);
    }
}

float4 mm_readB(int index) {
    float4 result = float4(src1[index],
        src1[index + 1],
        src1[index + 2],
        src1[index + 3]);
    return result;
}

void mm_write(int index, float value) {
    if (index < M * N)
    {
        dst[index] = value;
    }
}
#else
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
RWByteAddressBuffer dst : register(u0);

float4 mm_readA(int row, int col) {
    if (row < M)
    {
        int index = row * K + col;
        float4 result = float4(asfloat(src0.Load(4 * index)),
            asfloat(src0.Load(4 * (index + 1))),
            asfloat(src0.Load(4 * (index + 2))),
            asfloat(src0.Load(4 * (index + 3))));
        return result;
    }
    else {
        return float4(0, 0, 0, 0);
    }
}

float4 mm_readB(int index) {
    float4 result = float4(asfloat(src1.Load(4 * index)),
        asfloat(src1.Load(4 * (index + 1))),
        asfloat(src1.Load(4 * (index + 2))),
        asfloat(src1.Load(4 * (index + 3))));
    return result;
}

void mm_write(int index, float value) {
    if (index < M * N)
        {
            dst.Store(4 * (index), asuint(value));
        }
    }
}
#endif  // USE_STRUCTURED_BUFFERS
#endif  // USE_TEXTURE

[numthreads(LOCAL_GROUP_SIZE_X, LOCAL_GROUP_SIZE_Y, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);

    int globalRow = int(gl_GlobalInvocationID.x) * WORK_PER_THREAD_X;

    float acc[WORK_PER_THREAD_X];
    float4 ACached;
    float4 BCached;

    // Without this initialization strange values show up in acc.
    for (int innerRow = 0; innerRow < WORK_PER_THREAD_X; innerRow++) {
        acc[innerRow] = 0.0;
    }

    int sharedDimNearestVec4 = floor(K / 4);
    int sharedDimVec4Remainder = K % 4;
    for (int k = 0; k < sharedDimNearestVec4; k++) {
        BCached = mm_readB(k * 4);
        for (int i = 0; i < WORK_PER_THREAD_X; i++) {
            ACached = mm_readA(globalRow + i, k * 4);
            acc[i] = dot(ACached, BCached) + acc[i];
        }
    }

    if (sharedDimVec4Remainder == 1) {
        BCached = mm_readB(K - 1);
        for (int i = 0; i < WORK_PER_THREAD_X; i++) {
            ACached = mm_readA(globalRow + i, K - 1);
            acc[i] = dot(ACached.x, BCached.x) + acc[i];
        }
    }
    else if (sharedDimVec4Remainder == 2) {
        BCached = mm_readB(K - 2);
        for (int i = 0; i < WORK_PER_THREAD_X; i++) {
            ACached = mm_readA(globalRow + i, K - 2);
            acc[i] = dot(ACached.xy, BCached.xy) + acc[i];
        }
    }
    else if (sharedDimVec4Remainder == 3) {
        BCached = mm_readB(K - 3);
        for (int i = 0; i < WORK_PER_THREAD_X; i++) {
            ACached = mm_readA(globalRow + i, K - 3);
            acc[i] = dot(ACached.xyz, BCached.xyz) + acc[i];
        }
    }

    for (int innerRow = 0; innerRow < WORK_PER_THREAD_X; innerRow++) {
        mm_write(globalRow + innerRow, acc[innerRow]);
    }
}

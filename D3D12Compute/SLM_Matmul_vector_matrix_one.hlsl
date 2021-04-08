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

float mm_readA(int row, int col) {
    if (row < M)
    {
        int index = row * K + col;
        float result = src0[index];
        return result;
    }
    else {
        return float(0);
    }
}

float mm_readB(int row, int col) {
    int index = row * N + col;
    float result = src1[index];
    return result;
}

void mm_write(int row, int col, float value) {
    if (row < M && col < N)
    {
        int index = row * N + col;
        dst[index] = value;
    }
}
#else
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
RWByteAddressBuffer dst : register(u0);

float mm_readA(int row, int col) {
    if (row < M)
    {
        int index = row * K + col;
        float result = asfloat(src0.Load(4 * index));
        return result;
    }
    else {
        return float(0);
    }
}

float mm_readB(int row, int col) {
    int index = row * N + col;
    float result = asfloat(src1.Load(4 * index));
    return result;
}

void mm_write(int row, int col, float value) {
    if (row < M && col < N)
    {
        int index = row * N + col;
        dst.Store(4 * (index), asuint(value));
    }
}
#endif  // USE_STRUCTURED_BUFFERS
#endif  // USE_TEXTURE

groupshared float mm_Asub[1280];
[numthreads(LOCAL_GROUP_SIZE_X, LOCAL_GROUP_SIZE_Y, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);

    int globalRow = int(gl_GlobalInvocationID.y) * WORK_PER_THREAD_Y;
    int globalCol = int(gl_GlobalInvocationID.x) * WORK_PER_THREAD_X;
    float acc = 0.0;
    float ACached;
    float BCached;

    int localIndex = int(gl_LocalInvocationID.x);
    while (localIndex < 1280)
    {
        mm_Asub[localIndex] = mm_readA(globalRow, localIndex);
        localIndex += LOCAL_GROUP_SIZE_X;
    }
    GroupMemoryBarrierWithGroupSync();

    for (int k = 0; k < K; k++) {
        BCached = mm_readB(k, globalCol);
        ACached = mm_Asub[k];
        acc += BCached * ACached;
    }

    mm_write(globalRow, globalCol, acc);
}

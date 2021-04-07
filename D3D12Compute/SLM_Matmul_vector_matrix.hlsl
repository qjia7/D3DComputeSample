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

float4 mm_readB(int row, int col) {
    int index = row * N + col;
    float4 result = float4(src1[index],
        src1[index + 1],
        src1[index + 2],
        src1[index + 3]);
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

float4 mm_readB(int row, int col) {
    int index = row * N + col;
    float4 result = float4(asfloat(src1.Load(4 * index)),
        asfloat(src1.Load(4 * (index + 1))),
        asfloat(src1.Load(4 * (index + 2))),
        asfloat(src1.Load(4 * (index + 3))));
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

groupshared float4 mm_Asub[320];
[numthreads(LOCAL_GROUP_SIZE_X, LOCAL_GROUP_SIZE_Y, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);

    int globalRow = int(gl_GlobalInvocationID.y) * WORK_PER_THREAD_Y;
    int globalCol = int(gl_GlobalInvocationID.x) * WORK_PER_THREAD_X;
    int RowPerThread = WORK_PER_THREAD_Y;
    float4 acc[WORK_PER_THREAD_Y];
    float4 ACached;
    float4 BCached[4];

    // Without this initialization strange values show up in acc.
    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        acc[innerRow] = float4(0.0, 0.0, 0.0, 0.0);
    }

    int localIndex = int(gl_LocalInvocationID.x);
    while (localIndex < 320)
    {
        mm_Asub[localIndex] = mm_readA(globalRow, localIndex * 4);
        localIndex += LOCAL_GROUP_SIZE_X;
    }
    GroupMemoryBarrierWithGroupSync();

    int sharedDimNearestVec4 = floor(K / 4);
    int sharedDimVec4Remainder = K % 4;
    for (int k = 0; k < sharedDimNearestVec4; k++) {
        BCached[0] = mm_readB(k * 4, globalCol);
        BCached[1] = mm_readB(k * 4 + 1, globalCol);
        BCached[2] = mm_readB(k * 4 + 2, globalCol);
        BCached[3] = mm_readB(k * 4 + 3, globalCol);

        for (int i = 0; i < RowPerThread; i++) {
            ACached = mm_Asub[k];
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
            acc[i] = BCached[2] * ACached.z + acc[i];
            acc[i] = BCached[3] * ACached.w + acc[i];
        }
    }

    if (sharedDimVec4Remainder == 1) {
        BCached[0] = mm_readB(K - 1, globalCol);
        for (int i = 0; i < RowPerThread; i++) {
            ACached = mm_Asub[sharedDimNearestVec4];
            acc[i] = BCached[0] * ACached.x + acc[i];
        }
    }
    else if (sharedDimVec4Remainder == 2) {
        BCached[0] = mm_readB(K - 2, globalCol);
        BCached[1] = mm_readB(K - 1, globalCol);
        for (int i = 0; i < RowPerThread; i++) {
            ACached = mm_Asub[sharedDimNearestVec4];
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
        }
    }
    else if (sharedDimVec4Remainder == 3) {
        BCached[0] = mm_readB(K - 3, globalCol);
        BCached[1] = mm_readB(K - 2, globalCol);
        BCached[2] = mm_readB(K - 1, globalCol);
        for (int i = 0; i < RowPerThread; i++) {
            ACached = mm_Asub[sharedDimNearestVec4];
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
            acc[i] = BCached[2] * ACached.z + acc[i];
        }
    }

    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        mm_write(globalRow + innerRow, globalCol, acc[innerRow].x);
        mm_write(globalRow + innerRow, globalCol + 1, acc[innerRow].y);
        mm_write(globalRow + innerRow, globalCol + 2, acc[innerRow].z);
        mm_write(globalRow + innerRow, globalCol + 3, acc[innerRow].w);
    }
}

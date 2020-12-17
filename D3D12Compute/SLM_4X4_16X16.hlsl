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
StructuredBuffer<float> src0 : register(t0);
StructuredBuffer<float> src1 : register(t1);
RWStructuredBuffer<float> dst : register(u0);

  float mm_readA(int row, int col) {
      if (row < M && col < K)
      {
          float result = src0[row * K + col];
          return result;
      }
      else {
          return 0.0;
      }

  }

  float mm_readB(int row, int col) {
    float result = src1[row * N + col];
    return result;
  }

  void mm_write(int row, int col, float value) {
      if (row < M && col < N)
      {
          dst[row * N + col] = value;
      }
  }
#else
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
RWByteAddressBuffer dst : register(u0);

float mm_readA(int row, int col) {
    if (row < M && col < K)
    {
        float result = asfloat(src0.Load(4 * (row * K + col)));
        return result;
    }
    else {
        return 0.0;
    }
}

float mm_readB(int row, int col) {
    float result = asfloat(src1.Load(4 * (row * N + col)));
    return result;
}

void mm_write(int row, int col, float value) {
    if (row < M && col < N)
    {
        dst.Store(4 * (row * N + col), asuint(value));
    }
}
#endif  // USE_STRUCTURED_BUFFERS

static int RowPerThread = 4;
static int ColPerThread = 4;
static int TileInner = LOCAL_GROUP_SIZE_X * 4;

groupshared float mm_Asub[LOCAL_GROUP_SIZE_Y * 4][LOCAL_GROUP_SIZE_X * 4];
groupshared float mm_Bsub[LOCAL_GROUP_SIZE_Y * 4][LOCAL_GROUP_SIZE_X * 4];

[numthreads(LOCAL_GROUP_SIZE_X, LOCAL_GROUP_SIZE_Y, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int dimAOuter = M;
    int dimInner = K;
    int dimBOuter = N;
    int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
    int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;

    int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
    int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;

    int numTiles = (dimInner - 1) / TileInner + 1;

    float acc[4][4];
    float ACached;
    float BCached[4];

    // Without this initialization strange values show up in acc.
    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
      for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
        acc[innerRow][innerCol] = 0.0;
      }
    }

    int tileColA = int(gl_LocalInvocationID.x) * 4;
    int tileRowB = int(gl_LocalInvocationID.y) * 4;

    // Loop over shared dimension.
    for (int t = 0; t < numTiles; t++) {
      // Load one tile of A into local memory.
      for (int innerRow = 0; innerRow < 4; innerRow++) {
        for (int innerCol = 0; innerCol < 4; innerCol++) {
          int inputRow = tileRow + innerRow;
          int inputCol = tileColA + innerCol;

          mm_Asub[inputRow][inputCol] = mm_readA(
              globalRow + innerRow,
              t * TileInner + inputCol);
        }
      }
      // Load one tile of B into local memory.
      for (int innerRow = 0; innerRow < 4; innerRow++) {
        for (int innerCol = 0; innerCol < 4; innerCol++) {
          int inputRow = tileRowB + innerRow;
          int inputCol = tileCol + innerCol;

          mm_Bsub[inputRow][inputCol] = mm_readB(
            t * TileInner + inputRow,
            globalCol + innerCol);;
        }
      }

      GroupMemoryBarrierWithGroupSync();

      // Compute acc values for a single thread.
      for (int k = 0; k < TileInner; k++) {
        for (int inner = 0; inner < ColPerThread; inner++) {
          BCached[inner] = mm_Bsub[k][tileCol + inner];
        }

        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          ACached = mm_Asub[tileRow + innerRow][k];
          for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
            acc[innerRow][innerCol] += ACached * BCached[innerCol];
          }
        }
      }

      GroupMemoryBarrierWithGroupSync();
    }

    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
      for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {

        if ((globalCol + innerCol) < dimBOuter &&
            (globalRow + innerRow) < dimAOuter) {
          mm_write(globalRow + innerRow,
                   globalCol + innerCol,
                   acc[innerRow][innerCol]);
        }
      }
    }
}

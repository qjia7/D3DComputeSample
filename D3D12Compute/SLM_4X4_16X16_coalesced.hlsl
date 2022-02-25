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
static int TileAOuter = LOCAL_GROUP_SIZE_Y * 4;
static int TileBOuter = LOCAL_GROUP_SIZE_X * 4;

groupshared float mm_Asub[LOCAL_GROUP_SIZE_Y * 4][LOCAL_GROUP_SIZE_X * 4];
groupshared float mm_Bsub[LOCAL_GROUP_SIZE_X * 4][LOCAL_GROUP_SIZE_X * 4];

[numthreads(LOCAL_GROUP_SIZE_X, LOCAL_GROUP_SIZE_Y, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int dimAOuter = M;
    int dimInner = K;
    int dimBOuter = N;
      int localRow = int(gl_LocalInvocationID.y);
      int localCol = int(gl_LocalInvocationID.x);
    int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
    int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;

    int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
    int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;
      int newGlobalRow = int(gl_WorkGroupID.y) * TileAOuter;
      int newGlobalCol = int(gl_WorkGroupID.x) * TileBOuter;

    int numTiles = (dimInner - 1) / TileInner + 1;

  float acc[4][4] = (float[4][4]) 0;
  float ACached = 0.0f;
  float BCached[4] = (float[4]) 0;
  {
    [loop] for (int innerRow = 0; (innerRow < 4); innerRow = (innerRow + 1)) {
      {
        [loop] for (int innerCol = 0; (innerCol < 4);
                    innerCol = (innerCol + 1)) {
          acc[min(uint(innerRow), 3u)][min(uint(innerCol), 3u)] = 0.0f;
        }
      }
    }
  }
  {
    [loop] for (int t = 0; (t < numTiles); t = (t + 1)) {
      {
        [loop] for (int inputRow = localRow; (inputRow < 32);
                    inputRow = (inputRow + 8)) {
          {
            [loop] for (int inputCol = localCol; (inputCol < 32);
                        inputCol = (inputCol + 8)) {
              mm_Asub[min(uint(inputRow), 31u)][min(uint(inputCol), 31u)] =
                  mm_readA(
                      (newGlobalRow + inputRow), ((t * 32) + inputCol));
            }
          }
        }
      }
      {
        [loop] for (int inputRow = localRow; (inputRow < 32);
                    inputRow = (inputRow + 8)) {
          {
            [loop] for (int inputCol = localCol; (inputCol < 32);
                        inputCol = (inputCol + 8)) {
              mm_Bsub[min(uint(inputRow), 31u)][min(uint(inputCol), 31u)] =
                  mm_readB(
                      ((t * 32) + inputRow), (newGlobalCol + inputCol));
            }
          }
        }
      }
      GroupMemoryBarrierWithGroupSync();
      {
        [loop] for (int k = 0; (k < 32); k = (k + 1)) {
          {
            [loop] for (int inner = 0; (inner < 4); inner = (inner + 1)) {
              BCached[min(uint(inner), 3u)] =
                  mm_Bsub[min(uint(k), 31u)][min(uint((tileCol + inner)), 31u)];
            }
          }
          {
            [loop] for (int innerRow = 0; (innerRow < 4);
                        innerRow = (innerRow + 1)) {
              ACached = mm_Asub[min(uint((tileRow + innerRow)), 31u)][min(
                  uint(k), 31u)];
              {
                [loop] for (int innerCol = 0; (innerCol < 4);
                            innerCol = (innerCol + 1)) {
                  acc[min(uint(innerRow), 3u)][min(uint(innerCol), 3u)] =
                      (acc[min(uint(innerRow), 3u)][min(uint(innerCol), 3u)] +
                       (ACached * BCached[min(uint(innerCol), 3u)]));
                }
              }
            }
          }
        }
      }
      GroupMemoryBarrierWithGroupSync();
    }
  }
  {
    [loop] for (int innerRow = 0; (innerRow < 4); innerRow = (innerRow + 1)) {
      {
        [loop] for (int innerCol = 0; (innerCol < 4);
                    innerCol = (innerCol + 1)) {
          bool tint_tmp_1 = ((globalCol + innerCol) < dimBOuter);
          if (tint_tmp_1) {
            tint_tmp_1 = ((globalRow + innerRow) < dimAOuter);
          }
          if ((tint_tmp_1)) {
            mm_write(
                (globalRow + innerRow), (globalCol + innerCol),
                acc[min(uint(innerRow), 3u)][min(uint(innerCol), 3u)]);
          }
        }
      }
    }
  }
}

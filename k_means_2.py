import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt 

import numpy as np
from math import floor,sqrt
from typing import List

N = 512
P_MIN = 0
P_MAX = 10
MAX_ITER = 1000
EPOCH_1_THRESHOLD = 1
D = 2
NUM_POINTS = 500
def main():

    n = NUM_POINTS
    # we should generate the points so that they actually try to cluster around a certain point with standard deviation + some random points 
    P = np.random.uniform(P_MIN, P_MAX, (n,1)).astype(np.float32)     # data with d dimensions for n datapoints 
    P, L, C = k_means(P)

    # then plot these clusters 
    
    return None 

def plot_k_means(P, C, L):
    return None 

def k_means(P):
    """
    """
    n = len(P)
    d = D
    k = 10
    grid_size = 10
    block_size = 10

    mod = SourceModule("""                       
    __device__ float dist(float4 point, float4 centroid, int d) {
        int dist = 0;
        for(int i = 0; i < d; i++) {
            dist += (point[i] - centroid[i])**2;
        }
        dist = sqrt(dist);
        return dist;
    }
    __global__ void label(float4* P, float4* C, int* L, float** ICD, int** RID, int n, int k, int* M) {
        int t = n / (gridDim.x * blockDim.x);
        int s = blockIdx.x * (n / gridDim.x) + threadIdx.x * t;
        int oldCnt, newCnt, curCnt;
        float oldDist, newDist, curDist;
        for (int i = s + 1; i <= s + t; i++) {
            M[i] = 0;
            oldCnt = L[i];
            oldDist = dist(P[i], C[oldCnt], d);
            M[i]++;
            newCnt = oldCnt;
            newDist = oldDist;

            for (int j = 2; j <= k; j++) {
                curCnt = RID[oldCnt][j];
                if (ICD[oldCnt][curCnt] > 2 * oldDist) {
                    break;
                }

                curDist = dist(P[i], C[curCnt], d);
                M[i]++;
                if (curDist < newDist) {
                    newDist = curDist;
                    newCnt = curCnt;
                }
            }

            L[i] = newCnt;
        }
    }
    """)

    C = create_equally_spaced_points(k, d, P_MIN, P_MAX)              # k cluster-centroids (seeding)
    ICD =  np.zeros((k,k)).astype(np.int32)                           # inter-centroid distances   
    RID = np.zeros((k,k)).astype(np.int32)                            # ranked inter-centroid distancess 
    L= np.zeros((n,1)).astype(np.int32)                               # labels for n datapoints 
    M = np.full((n,1),k)                                              # initially we assume that all points take k distance calculations (maximum - worst case here)
    
    # allocate memory on the GPU
    C_gpu = cuda.mem_alloc(C.nbytes)
    ICD_gpu = cuda.mem_alloc(ICD.nbytes)
    RID_gpu = cuda.mem_alloc(RID.nbytes)
    L_gpu = cuda.mem_alloc(L.nbytes)
    P_gpu = cuda.mem_alloc(P.nbytes)
    M_gpu = cuda.mem_alloc(M.bytes)
  
    # copy over data from host -> device (GPU)
    cuda.memcpy_htod(P_gpu, P)  # copy P to GPU device 

    func = mod.get_function("label")
    grid = (1, 1)
    block = (4, 4, 1)
    func.prepare("PPPPPiPiP")
    

    #epoch stage 1 
    while sum(M)/n > EPOCH_1_THRESHOLD:
        #calculate inter-centroid distances (ICD) matrix
        ICD = calc_ICD(k, d, C)
        #sort each row of the ICD matrix to device ranked index (RID) matrix
        RID = calc_RID(ICD, k)
        #copy C, ICD, RID to GPU device 
        cuda.memcpy_htod(C_gpu, C)
        cuda.memcpy_htod(ICD_gpu, ICD)
        cuda.memcpy_htod(RID_gpu, RID)
        #launch GPU kernel to label P to nearest centroids 
        func.prepared_call(grid, block, P_gpu, C_gpu, L_gpu, ICD_gpu, RID_gpu, np.int32(n), np.int32(k), M_gpu)
        #copy L back to host 
        cuda.memcpy_dtoh(L, L_gpu)
        cuda.memcpy_dtoh(M, M_gpu)
        #calculate mean for each cluster and update C
        C = update_C(P, L, d, k)

    # EPOCH 2 
    #rearrange P to Pa1 s.t  [Ma1, ..., M-an] is in decreasing order
    Pa = [P[i] for i, _ in sorted(enumerate(M), key=lambda x: x[1], reverse = True)]
    cuda.memcpy_htod(P_gpu, Pa)

    convergence = False
    while not convergence:
        #calculate inter-centroid distances (ICD) matrix
        ICD = calc_ICD(k, d, C)
        #sort each row of the ICD matrix to device ranked index (RID) matrix
        RID = calc_RID(ICD, k)
        #copy C, ICD, RID to GPU device 
        cuda.memcpy_htod(C_gpu, C)
        cuda.memcpy_htod(ICD_gpu, ICD)
        cuda.memcpy_htod(RID_gpu, RID)
        #launch GPU kernel to label P to nearest centroids 
        func.prepared_call(grid, block, P_gpu, C_gpu, L_gpu, ICD_gpu, RID_gpu, np.int32(n), np.int32(k), M_gpu)
        #copy L back to host 
        cuda.memcpy_dtoh(L, L_gpu)
        cuda.memcpy_dtoh(M, M_gpu)
        #calculate mean for each cluster and update C
        C = update_C(P, L, d, k)

    # Cleanup
    C_gpu.free()
    ICD_gpu.free()
    RID_gpu.free()
    L_gpu.free()
    P_gpu.free()
    M_gpu.free()

    return P, L, C

def create_equally_spaced_points(k, d, P_MIN, P_MAX):
    """seeds the centroids of the k clusters so that they are equally spaced from each other 
    
    """
    spacing = np.linspace(P_MIN, P_MAX, num=int(round(k ** (1/d))) + 1) # +1 so we include the P_MAX as a possible endpoint 
    grid_points = np.array(np.meshgrid(*([spacing]*d))).T.reshape(-1, d)
    
    # Select n points from the grid points
    centroids = grid_points[:k]
    
    return centroids.astype(np.float32)


def update_C(P, L, d, k):
    """ updates the cluster centroid 
    """
    count_arrr = np.zeros((k,1))    # how many points in each cluster 
    output_arr = np.zeros((k,d))    # coordinates of the cluster centres 

    for i in range(P): # for each point 
        label = L[i]  # label of cluster for that point
        count_arrr[label] += 1
        output_arr[label] += P  # element-wise add all the values of the current point to the sum 
    
    # now divide through for each datapoint to find the mean cluster point for each dimension  
    for i in range(P): # for each point 
        label = L[i]  # label of cluster for that point
        output_arr[label] = output_arr[label]/count_arrr[label]
    
    return output_arr

def dist(point1: List[float], point2: List[float]):
    """
    """
    d = len(point1)
    dist = 0
    for i in range(d):
        dist += (point1[i] - point2[i])**2
    dist = sqrt(dist)
    return dist 

def calc_ICD(k, d, C):
    """Calculates the inter-centroid distances 
    """
    ICD = np.zeros((k,d))

    for c1 in range(k):
        for c2 in range(k):
            ICD[c2][c2] = dist(C[c1], C[c2])
    return ICD 


def calc_RID(ICD, k):
    RID = np.zeros((k,k))
    for row in range(k):
        RID[row] = [i for i, _ in sorted(enumerate(ICD[row]), key=lambda x: x[1])]
    return RID 



if __name__ == "__main__":
    main()
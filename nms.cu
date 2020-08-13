/* 
 * NMS Benchmarking Framework
 *
 * "Work-Efficient Parallel Non-Maximum Suppression Kernels"
 * Copyright (c) 2019 David Oro et al.
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include "config.h"
#include "nms.h"

#define MAX_DETECTIONS		  4096
#define N_PARTITIONS		  32



/* GPU array for storing the coordinates, dimensions and score of each detected object */
float4* points;

/* CPU array for storing the coordinates, dimensions and score of each detected object */
float4* cpu_points;

/* GPU array for storing the detection bitmap */
uint8* pointsbitmap;

/* CPU array for storing the detection bitmap */
uint8* cpu_pointsbitmap;

/* Number of detection windows */
int ndetections;

/* GPU array for storing the non-maximum supression bitmap */
uint8* nmsbitmap;

/* Kernel streams and events */
cudaEvent_t begin_map_event;
cudaEvent_t end_map_event;
cudaEvent_t begin_reduce_event;
cudaEvent_t end_reduce_event;



/* Reads detections from a comma separated input text file 
   encoded as follows:

     x0,y0,width0,score0\n
     x1,y1,width1,score1\n
     x2,y2,width2,score2\n
     ...
     xn,yn,widthn,scoren\n
*/

int read_detections(const char* filename)
{
  FILE* fp;
  int x, y, w, cnt;
  float score;


  ndetections = 0;

  fp = fopen(filename, "r");

  if (!fp)
  {
    printf("Error: Unable to open file %s.\n", filename);
    return -1;
  }

  /* Memory allocation in the host memory address space */
  cpu_points = (float4*) malloc(sizeof(float4) * MAX_DETECTIONS);

  if(!cpu_points)
  {
    printf("Error: Unable to allocate CPU memory.\n");
    return -1;
  }

  memset(cpu_points, 0, sizeof(float4) * MAX_DETECTIONS);

  while(!feof(fp))
  {
     cnt = fscanf(fp, "%d,%d,%d,%f\n", &x, &y, &w, &score);

     if (cnt !=4)
     {
	printf("Error: Invalid file format in line %d when reading %s\n", ndetections, filename);
        return -1;
     }
 
    cpu_points[ndetections].x = (float) x;       // x coordinate
    cpu_points[ndetections].y = (float) y;       // y coordinate
    cpu_points[ndetections].z = (float) w;       // window dimensions
    cpu_points[ndetections].w = score;           // score

    ndetections++;
  }

  printf("Detections read from input file (%s): %d\n", filename, ndetections);

  fclose(fp);
  return 0;
}


int dump_merged_detections(const char* filename)
{
  FILE* fp;
  cudaError_t err;
  int x, y, w, i, totaldets;
  float score;

  totaldets = 0;

  err = cudaMemcpy(cpu_pointsbitmap, pointsbitmap, sizeof(uint8) * MAX_DETECTIONS, cudaMemcpyDeviceToHost);

  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  fp = fopen(filename, "w");

  if (!fp)
  {
    printf("Error: Unable to open file %s for writing.\n", filename);
    return -1;
  }

  for(i = 0; i < ndetections; i++)
  {
    if(cpu_pointsbitmap[i])
    {
      x = (int) cpu_points[i].x;          // x coordinate
      y = (int) cpu_points[i].y;          // y coordinate
      w = (int) cpu_points[i].z;          // window dimensions
      score = cpu_points[i].w;            // score
      fprintf(fp, "%d,%d,%d,%f\n", x, y, w, score);
      totaldets++; 
    }
  }

  printf("Detections after NMS: %d\n", totaldets);

  fclose(fp);
  return 0;
}


/* Gets the optimal X or Y dimension for a given CUDA block */
int get_optimal_dim(int val)
{
  int div, neg, cntneg, cntpos;


  /* We start figuring out if 'val' is divisible by 16 
     (e.g. optimal 16x16 CUDA block of maximum GPU occupancy */

  neg = 1;
  div = 16;
  cntneg = div;
  cntpos = div;

  /* In order to guarantee the ending of this loop if 'val' is 
     a prime number, we limit the loop to 5 iterations */

  for(int i=0; i<5; i++)
  {
    if(val % div == 0)
      return div;

    if(neg)
    {
      cntneg--;
      div = cntneg;
      neg = 0;
    }

    else
    {
      cntpos++;
      div = cntpos;
      neg = 1;
    }
  }

  return 16;
}


/* Gets an upper limit for 'val' multiple of the 'mul' integer */
int get_upper_limit(int val, int mul)
{
  int cnt = mul;

  /* The upper limit must be lower than
     the maximum allowed number of detections */

  while(cnt < val)
    cnt += mul;

  if(cnt > MAX_DETECTIONS)
    cnt = MAX_DETECTIONS;

  return cnt;
}


int allocate_gpu_memory()
{
  cudaError_t err;


  cudaEventCreate(&begin_map_event);
  cudaEventCreate(&end_map_event);
  cudaEventCreate(&begin_reduce_event);
  cudaEventCreate(&end_reduce_event);

  /* Memory allocation for the data structure of detected objects */
  err = cudaMalloc((void**) &points, sizeof(float4) * MAX_DETECTIONS);

  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMemset(points, 0, sizeof(float4) * MAX_DETECTIONS);

  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  /* Memory allocation for the non-maximum supression bitmaps */
  err = cudaMalloc((void**) &nmsbitmap, sizeof(uint8) * MAX_DETECTIONS * MAX_DETECTIONS);

  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMalloc((void**) &pointsbitmap, sizeof(uint8) * MAX_DETECTIONS);

  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMemset(nmsbitmap, 1, sizeof(uint8) * MAX_DETECTIONS * MAX_DETECTIONS);

  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMemset(pointsbitmap, 0, sizeof(uint8) * MAX_DETECTIONS);

  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  cpu_pointsbitmap = (uint8*) malloc(sizeof(uint8) * MAX_DETECTIONS);
  memset(cpu_pointsbitmap, 0, sizeof(uint8) * MAX_DETECTIONS);

  return 0;   
}


void free_memory()
{
  cudaEventDestroy(begin_map_event);
  cudaEventDestroy(end_map_event);
  cudaEventDestroy(begin_reduce_event);
  cudaEventDestroy(end_reduce_event);
  cudaFree(points);
  cudaFree(nmsbitmap);
  cudaFree(pointsbitmap);
  free(cpu_points);
  free(cpu_pointsbitmap);
}


int transfer_detections_to_gpu()
{
  cudaError_t err;


  err = cudaMemcpy(points, cpu_points, sizeof(float4) * MAX_DETECTIONS, cudaMemcpyHostToDevice);

  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  return 0;
}


/* NMS Map kernel */
__global__ void generate_nms_bitmap(float4* rects, uint8* nmsbitmap, float othreshold)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;


  if(rects[i].w < rects[j].w)
  {
    float area = (rects[j].z + 1.0f) * (rects[j].z + 1.0f);
    float w = max(0.0f, min(rects[i].x + rects[i].z, rects[j].x + rects[j].z) - max(rects[i].x, rects[j].x) + 1.0f);
    float h = max(0.0f, min(rects[i].y + rects[i].z, rects[j].y + rects[j].z) - max(rects[i].y, rects[j].y) + 1.0f);
    nmsbitmap[i * MAX_DETECTIONS + j] = (((w * h) / area) < othreshold) && (rects[j].z != 0);
  } 
}


/* NMS Reduce kernel */
__device__ __inline__ void compute_nms_point_mask(uint8* pointsbitmap, int cond, int idx, int ndetections)
{
  *pointsbitmap = __syncthreads_and(cond);
}


__global__ void reduce_nms_bitmap(uint8* nmsbitmap, uint8* pointsbitmap, int ndetections)
{
  int idx = blockIdx.x * MAX_DETECTIONS + threadIdx.x;


  compute_nms_point_mask(&pointsbitmap[blockIdx.x], nmsbitmap[idx], idx, ndetections);

  for(int i=0; i<(N_PARTITIONS-1); i++)
  {
    idx += MAX_DETECTIONS / N_PARTITIONS;
    compute_nms_point_mask(&pointsbitmap[blockIdx.x], pointsbitmap[blockIdx.x] && nmsbitmap[idx], idx, ndetections);
  }
} 


void non_maximum_suppression()
{
  dim3 pkthreads(1, 1, 1);
  dim3 pkgrid(1, 1, 1);
  int limit;
  float nms_elapsed_time;

  
  limit = get_upper_limit(ndetections, 16);

  pkthreads.x = get_optimal_dim(limit);
  pkthreads.y = get_optimal_dim(limit);
  pkgrid.x = limit / pkthreads.x;
  pkgrid.y = limit / pkthreads.y;

  cudaEventRecord(begin_map_event, 0);
  
  /* We build up the non-maximum supression bitmap matrix by removing overlapping windows */
  generate_nms_bitmap<<<pkgrid, pkthreads>>>(points, nmsbitmap, 0.3f);
  
  cudaEventRecord(end_map_event, 0);
  cudaEventSynchronize(end_map_event);
  cudaEventElapsedTime(&nms_elapsed_time, begin_map_event, end_map_event);
  printf("NMS-MAP elapsed time: %.3f ms\n", nms_elapsed_time);

  pkthreads.x = MAX_DETECTIONS / N_PARTITIONS; 
  pkthreads.y = 1;
  pkgrid.x = ndetections;
  pkgrid.y = 1;

  cudaEventRecord(begin_reduce_event, 0);

  /* Then we perform a reduction for generating a point bitmap vector */
  reduce_nms_bitmap<<<pkgrid, pkthreads>>>(nmsbitmap, pointsbitmap, ndetections);

  cudaEventRecord(end_reduce_event, 0);
  cudaEventSynchronize(end_reduce_event);
  cudaEventElapsedTime(&nms_elapsed_time, begin_reduce_event, end_reduce_event);
  printf("NMS-REDUCE elapsed time: %.3f ms\n", nms_elapsed_time);
}



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

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nms.h"


void print_help()
{
  printf("\nUsage: nmstest  <detections.txt>  <output.txt>\n\n");
  printf("               detections.txt -> Input file containing the coordinates, width, and scores of detected objects\n");
  printf("               output.txt     -> Output file after performing NMS\n\n");
}



int init_cuda_runtime()
{
  cudaError_t err;
  int i, value, dev, seldev;
  int maxsm = 0, maxmajor = 0, maxminor = 0;
  struct cudaDeviceProp device;


  err = cudaRuntimeGetVersion(&value);
  
  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  printf("\nCUDA Runtime Version %d\n", value);
  err = cudaGetDeviceCount(&dev);

  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -2;
  }

  /* We require at least CUDA 6.0 */
  if(value < 6000)
  {
    printf("Error: This software requires CUDA Runtime version 6.0 or better\n");
    return -3;
  }

  seldev = 0;

  for(i=0; i<dev; i++)
  {
    err = cudaGetDeviceProperties(&device, i); 
    if(err != cudaSuccess)
    {
      printf("Error: %s\n", cudaGetErrorString(err));
      return -4;
    }

    printf("Device %d# %s\t [%2.2f GHz - %d Multiprocessors - Core sm_%d%d - %ld MB]\n", i, device.name, 
    (float) device.clockRate / 1e6, device.multiProcessorCount, device.major, device.minor,
    (long) device.totalGlobalMem / (1024*1024));

    /* We select the CUDA device with the 
       highest number of multiprocessors and 
       with the newest core revision */

    if((device.major >= maxmajor) && (device.minor >= maxminor))
    {
      maxmajor = device.major;
      maxminor = device.minor;

      if(device.multiProcessorCount > maxsm)
      {
	seldev = i;
	maxsm = device.multiProcessorCount;
      }
    }
   }

  err = cudaSetDevice(seldev);
  if(err != cudaSuccess)
  {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -5;
  }

  printf("Device %d# has been selected for CUDA computation\n\n", seldev);
  return 0;
}



int main(int argc, char *argv[])
{
  int res;


  if(argc != 3)
  {
    print_help();
    return 0;
  }

  /* CUDA runtime initialization */
  if (init_cuda_runtime() < 0)
    return -1;

  /* Read input detection coordinates from the text file */
  res = read_detections("detections.txt");

  if(res < 0)
    return -1;

  /* Allocate GPU memory */
  if (allocate_gpu_memory() < 0)
    return -1;

  /* Transfer detection coordinates read from the input text file to the GPU */
  transfer_detections_to_gpu(); 

  /* Execute NMS on the GPU */
  non_maximum_suppression();

  /* Dump detections after having performed the NMS */
  res = dump_merged_detections("output.txt");

  if(res < 0)
    return -1;

  free_memory();

  return 0;
}


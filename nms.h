/* 
 * NMS Benchmarking Framework
 *
 * "Work-efficient Parallel Non-Maximum Suppression Kernels"
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


#ifndef NMS_H
#define NMS_H

#ifdef __cplusplus
extern "C" {
#endif

int read_detections(const char* filename);
int dump_merged_detections(const char* filename);
int transfer_detections_to_gpu(); 
int allocate_gpu_memory();
void free_memory();
void non_maximum_suppression();

#ifdef __cplusplus
}
#endif

#endif


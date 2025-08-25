/*******************************************************************
 * Author: <Name1>, <Name2>
 * Date: <Date>
 * File: mat_mul.c
 * Description: This file contains implementations of matrix multiplication
 *			    algorithms using various optimization techniques.
 *******************************************************************/

// PA 1: Matrix Multiplication

// includes
#include <stdio.h>
#include <stdlib.h>         // for malloc, free, atoi
#include <time.h>           // for time()
#include <chrono>	        // for timing
#include <xmmintrin.h> 		// for SSE
#include <immintrin.h>		// for AVX

#include "helper.h"			// for helper functions

// defines
// NOTE: you can change this value as per your requirement
#define TILE_SIZE	100		// size of the tile for blocking

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void naive_mat_mul(double *A, double *B, double *C, int size) {

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				C[i * size + j] += A[i * size + k] * B[k * size + j];
			}
		}
	}
}

/**
 * @brief 		Task 1A: Performs matrix multiplication of two matrices using loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void loop_opt_mat_mul(double *A, double *B, double *C, int size){
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
// for(int i=0; i<size; i++){
//     for(int j=0; j<size; j++){
//         double r = A[i*size + j];
//         for(int k=0; k<size; k+=16){
//             C[i*size+k    ] = r * B[j*size+k    ];
//             C[i*size+k + 1] = r * B[j*size+k + 1];
//             C[i*size+k + 2] = r * B[j*size+k + 2];
//             C[i*size+k + 3] = r * B[j*size+k + 3];
//             C[i*size+k + 4] = r * B[j*size+k + 4];
//             C[i*size+k + 5] = r * B[j*size+k + 5];
//             C[i*size+k + 6] = r * B[j*size+k + 6];
//             C[i*size+k + 7] = r * B[j*size+k + 7];
//             C[i*size+k + 8] = r * B[j*size+k + 8];
//             C[i*size+k + 9] = r * B[j*size+k + 9];
//             C[i*size+k +10] = r * B[j*size+k +10];
//             C[i*size+k +11] = r * B[j*size+k +11];
//             C[i*size+k +12] = r * B[j*size+k +12];
//             C[i*size+k +13] = r * B[j*size+k +13];
//             C[i*size+k +14] = r * B[j*size+k +14];
//             C[i*size+k +15] = r * B[j*size+k +15];
//         }
//     }
// }

// for(int i=0; i<size; i++){
//     for(int j=0; j<size; j++){
//         double r = A[i*size + j];
//         for(int k=0; k<size; k+=32){
//             C[i*size + k    ] = r * B[j*size + k    ];
//             C[i*size + k + 1] = r * B[j*size + k + 1];
//             C[i*size + k + 2] = r * B[j*size + k + 2];
//             C[i*size + k + 3] = r * B[j*size + k + 3];
//             C[i*size + k + 4] = r * B[j*size + k + 4];
//             C[i*size + k + 5] = r * B[j*size + k + 5];
//             C[i*size + k + 6] = r * B[j*size + k + 6];
//             C[i*size + k + 7] = r * B[j*size + k + 7];
//             C[i*size + k + 8] = r * B[j*size + k + 8];
//             C[i*size + k + 9] = r * B[j*size + k + 9];
//             C[i*size + k +10] = r * B[j*size + k +10];
//             C[i*size + k +11] = r * B[j*size + k +11];
//             C[i*size + k +12] = r * B[j*size + k +12];
//             C[i*size + k +13] = r * B[j*size + k +13];
//             C[i*size + k +14] = r * B[j*size + k +14];
//             C[i*size + k +15] = r * B[j*size + k +15];
//             C[i*size + k +16] = r * B[j*size + k +16];
//             C[i*size + k +17] = r * B[j*size + k +17];
//             C[i*size + k +18] = r * B[j*size + k +18];
//             C[i*size + k +19] = r * B[j*size + k +19];
//             C[i*size + k +20] = r * B[j*size + k +20];
//             C[i*size + k +21] = r * B[j*size + k +21];
//             C[i*size + k +22] = r * B[j*size + k +22];
//             C[i*size + k +23] = r * B[j*size + k +23];
//             C[i*size + k +24] = r * B[j*size + k +24];
//             C[i*size + k +25] = r * B[j*size + k +25];
//             C[i*size + k +26] = r * B[j*size + k +26];
//             C[i*size + k +27] = r * B[j*size + k +27];
//             C[i*size + k +28] = r * B[j*size + k +28];
//             C[i*size + k +29] = r * B[j*size + k +29];
//             C[i*size + k +30] = r * B[j*size + k +30];
//             C[i*size + k +31] = r * B[j*size + k +31];
//         }
//     }
// }
// for(int i=0; i<size; i++){
//     for(int j=0; j<size; j++){
//         double r = A[i*size + j];
//         for(int k=0; k<size; k+=64){
//             C[i*size + k    ] = r * B[j*size + k    ];
//             C[i*size + k + 1] = r * B[j*size + k + 1];
//             C[i*size + k + 2] = r * B[j*size + k + 2];
//             C[i*size + k + 3] = r * B[j*size + k + 3];
//             C[i*size + k + 4] = r * B[j*size + k + 4];
//             C[i*size + k + 5] = r * B[j*size + k + 5];
//             C[i*size + k + 6] = r * B[j*size + k + 6];
//             C[i*size + k + 7] = r * B[j*size + k + 7];
//             C[i*size + k + 8] = r * B[j*size + k + 8];
//             C[i*size + k + 9] = r * B[j*size + k + 9];
//             C[i*size + k +10] = r * B[j*size + k +10];
//             C[i*size + k +11] = r * B[j*size + k +11];
//             C[i*size + k +12] = r * B[j*size + k +12];
//             C[i*size + k +13] = r * B[j*size + k +13];
//             C[i*size + k +14] = r * B[j*size + k +14];
//             C[i*size + k +15] = r * B[j*size + k +15];
//             C[i*size + k +16] = r * B[j*size + k +16];
//             C[i*size + k +17] = r * B[j*size + k +17];
//             C[i*size + k +18] = r * B[j*size + k +18];
//             C[i*size + k +19] = r * B[j*size + k +19];
//             C[i*size + k +20] = r * B[j*size + k +20];
//             C[i*size + k +21] = r * B[j*size + k +21];
//             C[i*size + k +22] = r * B[j*size + k +22];
//             C[i*size + k +23] = r * B[j*size + k +23];
//             C[i*size + k +24] = r * B[j*size + k +24];
//             C[i*size + k +25] = r * B[j*size + k +25];
//             C[i*size + k +26] = r * B[j*size + k +26];
//             C[i*size + k +27] = r * B[j*size + k +27];
//             C[i*size + k +28] = r * B[j*size + k +28];
//             C[i*size + k +29] = r * B[j*size + k +29];
//             C[i*size + k +30] = r * B[j*size + k +30];
//             C[i*size + k +31] = r * B[j*size + k +31];
//             C[i*size + k +32] = r * B[j*size + k +32];
//             C[i*size + k +33] = r * B[j*size + k +33];
//             C[i*size + k +34] = r * B[j*size + k +34];
//             C[i*size + k +35] = r * B[j*size + k +35];
//             C[i*size + k +36] = r * B[j*size + k +36];
//             C[i*size + k +37] = r * B[j*size + k +37];
//             C[i*size + k +38] = r * B[j*size + k +38];
//             C[i*size + k +39] = r * B[j*size + k +39];
//             C[i*size + k +40] = r * B[j*size + k +40];
//             C[i*size + k +41] = r * B[j*size + k +41];
//             C[i*size + k +42] = r * B[j*size + k +42];
//             C[i*size + k +43] = r * B[j*size + k +43];
//             C[i*size + k +44] = r * B[j*size + k +44];
//             C[i*size + k +45] = r * B[j*size + k +45];
//             C[i*size + k +46] = r * B[j*size + k +46];
//             C[i*size + k +47] = r * B[j*size + k +47];
//             C[i*size + k +48] = r * B[j*size + k +48];
//             C[i*size + k +49] = r * B[j*size + k +49];
//             C[i*size + k +50] = r * B[j*size + k +50];
//             C[i*size + k +51] = r * B[j*size + k +51];
//             C[i*size + k +52] = r * B[j*size + k +52];
//             C[i*size + k +53] = r * B[j*size + k +53];
//             C[i*size + k +54] = r * B[j*size + k +54];
//             C[i*size + k +55] = r * B[j*size + k +55];
//             C[i*size + k +56] = r * B[j*size + k +56];
//             C[i*size + k +57] = r * B[j*size + k +57];
//             C[i*size + k +58] = r * B[j*size + k +58];
//             C[i*size + k +59] = r * B[j*size + k +59];
//             C[i*size + k +60] = r * B[j*size + k +60];
//             C[i*size + k +61] = r * B[j*size + k +61];
//             C[i*size + k +62] = r * B[j*size + k +62];
//             C[i*size + k +63] = r * B[j*size + k +63];
//         }
//     }
// }
// for(int i=0; i<size; i++){
//     for(int j=0; j<size; j++){
//         double sum = 0;
//         for(int k=0; k<size; k++){
//             sum += A[i*size + k] * B[k*size + j];
//         }
//         C[i*size + j] = sum;
//     }
// }

// for(int i=0; i<size; i++){
//     for(int j=0; j<size; j+=4){
//         double sum0=0, sum1=0, sum2=0, sum3=0;
//         for(int k=0; k<size; k++){
//             double r = A[i*size + k];
//             sum0 += r * B[k*size + j    ];
//             sum1 += r * B[k*size + j + 1];
//             sum2 += r * B[k*size + j + 2];
//             sum3 += r * B[k*size + j + 3];
//         }
//         C[i*size + j    ] = sum0;
//         C[i*size + j + 1] = sum1;
//         C[i*size + j + 2] = sum2;
//         C[i*size + j + 3] = sum3;
//     }
// }

// for(int i=0; i<size; i++){
//     for(int j=0; j<size; j+=8){
//         double sum0=0,sum1=0,sum2=0,sum3=0,sum4=0,sum5=0,sum6=0,sum7=0;
//         for(int k=0; k<size; k++){
//             double r = A[i*size + k];
//             sum0 += r * B[k*size + j    ];
//             sum1 += r * B[k*size + j + 1];
//             sum2 += r * B[k*size + j + 2];
//             sum3 += r * B[k*size + j + 3];
//             sum4 += r * B[k*size + j + 4];
//             sum5 += r * B[k*size + j + 5];
//             sum6 += r * B[k*size + j + 6];
//             sum7 += r * B[k*size + j + 7];
//         }
//         C[i*size + j    ] = sum0;
//         C[i*size + j + 1] = sum1;
//         C[i*size + j + 2] = sum2;
//         C[i*size + j + 3] = sum3;
//         C[i*size + j + 4] = sum4;
//         C[i*size + j + 5] = sum5;
//         C[i*size + j + 6] = sum6;
//         C[i*size + j + 7] = sum7;
//     }
// }
for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j += 64) {
        double sum[64] = {0};
        for (int k = 0; k < size; k++) {
            double r = A[i*size + k];
            for (int x = 0; x < 64; x++) {
                sum[x] += r * B[k*size + j + x];
            }
        }
        for (int x = 0; x < 64; x++) {
            C[i*size + j + x] = sum[x];  
        }
    }
}



//-------------------------------------------------------------------------------------------------------------------------------------------

}


/**
 * @brief 		Task 1B: Performs matrix multiplication of two matrices using tiling.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the tile size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void tile_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    // int i,j,k;
	// for(i=0;i<size;i=+ tile_size){
	// 	for(j=0;j<size;j=+ tile_size){
	// 		for(k=0;k<size; k=+ tile_size){
	// 			for(int l=i;l<i+ tile_size;l++){
	// 				for(int m=j;m<j+ tile_size; m++){
	// 					for(int n=k;n=k+ tile_size; n++){
	// 						C[l*tile_size+m] += A[l*tile_size+n]*B[n*tile_size+m];
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}

	// }

//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

/**
 * @brief 		Task 1C: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *A, double *B, double *C, int size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    

//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

/**
 * @brief 		Task 1D: Performs matrix multiplication of two matrices using combination of tiling/SIMD/loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void combination_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
    
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {

	if ( argc <= 1 ) {
		printf("Usage: %s <matrix_dimension>\n", argv[0]);
		return 0;
	}

	else {
		int size = atoi(argv[1]);

		double *A = (double *)malloc(size * size * sizeof(double));
		double *B = (double *)malloc(size * size * sizeof(double));
		double *C = (double *)calloc(size * size, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, size, size);
		initialize_matrix(B, size, size);

		// perform normal matrix multiplication
		auto start = std::chrono::high_resolution_clock::now();
		naive_mat_mul(A, B, C, size);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_naive_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Normal matrix multiplication took %ld ms to execute \n\n", time_naive_mat_mul);

	#ifdef OPTIMIZE_LOOP_OPT
		// Task 1a: perform matrix multiplication with loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		loop_opt_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_loop_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Loop optimized matrix multiplication took %ld ms to execute \n", time_loop_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_loop_mat_mul);
	#endif

	#ifdef OPTIMIZE_TILING
		// Task 1b: perform matrix multiplication with tiling

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		tile_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_tiling_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Tiling matrix multiplication took %ld ms to execute \n", time_tiling_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_tiling_mat_mul);
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 1c: perform matrix multiplication with SIMD instructions 

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		simd_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_simd_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		printf("SIMD matrix multiplication took %ld ms to execute \n", time_simd_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_simd_mat_mul);
	#endif

	#ifdef OPTIMIZE_COMBINED
		// Task 1d: perform matrix multiplication with combination of tiling, SIMD and loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		combination_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_combination = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Combined optimization matrix multiplication took %ld ms to execute \n", time_combination);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_combination);
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>

int check_is_power_of_two(int n) {
    if (n == 0)
        return 0; 
    while (n != 1) { 
        if (n%2 != 0) return 0;
        n = n/2;
    }
    return 1;
}


int merge(int **parts, int *sizes, int n_array, int *sorted)
{
    int i = 0;           // index of output values
    int min_val;         // current minimum
    int min_i;           // minimium value position

    int *to_check = calloc(n_array,sizeof(int));

    while(1){
        min_val = INT_MAX;                                                          // declaring that the current maximum is the very high number
        min_i = -1;                                                   

        for(int j = 0; j < n_array; j++){
            if(to_check[j] < sizes[j] && parts[j][to_check[j]] < min_val){          // if there is still the number to compare and this number is less than current minimum
                min_val = parts[j][to_check[j]];                                    // change current minimum
                min_i = j;                                                          // change index of min position
            }
        }

        if(min_i == -1)                                                             // if no minima then 
            break;
           
        sorted[i++] = min_val;                                                      // add current minima to output array
        to_check[min_i]++;                                                          // note than this number was a minimum number
    }
    free(to_check);
    return 0;
}


void quickSort(int *arr, int first, int last) {
    int i, j, pivot, temp;
    
    if(first<last){                                 
        pivot=first;
        i=first;                                    
        j=last;

        while(i<j){
            while(arr[i]<=arr[pivot] && i<last)    
                i++;
            while(arr[j]>arr[pivot])                
                j--;                                
            if(i<j){                                
                temp=arr[i];                        
                arr[i]=arr[j];
                arr[j]=temp;
            }
        }

        temp=arr[pivot];                     
        arr[pivot]=arr[j];
        arr[j]=temp;
        quickSort(arr,first,j-1);
        quickSort(arr,j+1,last);
   }
}


int main(int argc, char** argv) {
	srand(time(NULL));
    MPI_Init(&argc, &argv);                         // initializing

    int rank, size;                                 // process rank and their quantity	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);           // getting rank
	MPI_Comm_size(MPI_COMM_WORLD, &size);           // getting size of ranks
	
    if (argc != 2 || argv[1] <= 0) {                // checking for the right inputs
        if (rank == 0)
            fprintf(stderr, "Usage: %s <quantity-of-numbers>!\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (!check_is_power_of_two(size) || atoi(argv[1])%size != 0) {        // checking if the number of processes is 2^n and the array size is divisible without remainder
        if (rank == 0)
            fprintf(stderr, "np should be a power of 2 and size of array should divide by number of processes without remainder!\n");
        MPI_Finalize();
        exit(1);
    }

	int number_qty = atoi(argv[1]);                                                     // getting quantity of numbers              
	int arr_size = number_qty/size;                                                     // size of each part of array
	int *array = malloc(number_qty * sizeof(int));                                      // array of our numbers
	for(int c = 0; c < number_qty; c++) array[c] = rand() % number_qty;                 // giving random numbers to array from 0 to quantity of numbers
	
	int *part = malloc(arr_size * sizeof(int));                                         // part of our array
	MPI_Scatter(array, arr_size, MPI_INT, part, arr_size, MPI_INT, 0, MPI_COMM_WORLD);  // send each part to each process	
	quickSort(part,0,arr_size-1);                                                       // sorting these parts of array

	int *sorted = NULL;                                                                 // our sorted array
	if(rank == 0) {	 
        printf("Unsorted array: ");
        for(int c = 0; c < number_qty; c++) printf("%d ", array[c]);                    // printing unsorted array
	    printf("\n\n");
		sorted = malloc(number_qty * sizeof(int));                                      // initializing sorted array
	}
	

	MPI_Gather(part, arr_size, MPI_INT, sorted, arr_size, MPI_INT, 0, MPI_COMM_WORLD);  // gathering our sorted parts not in sorted order
	
	if(rank == 0) {

        int **parts = (int **) malloc(size * sizeof(int *));                            // creating the array of arrays of sorted parts
        for (int i=0; i < size; i++) {
	        int j = 0;
	        parts[i] = (int *) malloc(arr_size * sizeof(int));
	        for (j = 0; j < arr_size; j++) {
		        parts[i][j] = sorted[j+i*arr_size];
	        }   
        }

        int *sizes = (int*)malloc(size*sizeof(int));                                     // sizes of each array
        for (int i =0; i < size; i++) sizes[i]=arr_size;

        merge(parts,sizes,size,sorted);                                                  // merging parts in sorted way

		printf("Sorted array: ");
		for(int c = 0; c < number_qty; c++) printf("%d ", sorted[c]);		
		printf("\n\n");

        free(sizes);
        free(parts);
		free(sorted);	
	}

	free(part);
	free(array);
	
	MPI_Finalize();
	
}
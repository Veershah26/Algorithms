#include <stdio.h>
#include <stdlib.h>

#define PIVOT_BEAD(row, col) beads[(row) * max_value + (col)]

void print_array(const int *array, int size) {
    for(int idx = 0; idx < size; idx++) {
        printf("%d ", array[idx]);
    }
    printf("\n");
}

void bead_sort(int *array, size_t size) {
    int max_value = array[0];
    for(int idx = 1; idx < size; idx++) {
        if(array[idx] > max_value) {
            max_value = array[idx];
        }
    }

    unsigned char *beads = calloc(max_value * size, sizeof(unsigned char));

    for(int row = 0; row < size; row++) {
        for(int col = 0; col < array[row]; col++) {
            PIVOT_BEAD(row, col) = 1;
        }
    }

    for(int col = 0; col < max_value; col++) {
        int count = 0;
        for(int row = 0; row < size; row++) {
            count += PIVOT_BEAD(row, col);
            PIVOT_BEAD(row, col) = 0;
        }
        for(int row = size - count; row < size; row++) {
            PIVOT_BEAD(row, col) = 1;
        }
    }

    for(int row = 0; row < size; row++) {
        int col = 0;
        while(col < max_value && PIVOT_BEAD(row, col)) {
            col++;
        }
        array[row] = col;
    }

    free(beads); 
 }

 int main() {
    int array_size;
    printf("Enter Array Size: ");
    scanf("%d", &array_size);

    int *array = malloc(array_size * sizeof(int));

    printf("Enter Array Elements:\n");
    for(int idx = 0; idx < array_size; idx++) {
        scanf("%d", &array[idx]);
    }

    printf("Original Array: ");
    print_array(array, array_size);

    bead_sort(array, array_size);

    printf("Sorted Array: ");
    print_array(array, array_size);

    free(array);
    return 0;
 }
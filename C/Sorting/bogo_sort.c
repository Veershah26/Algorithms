#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void print_array(const int *array, int size) {
    for(int i = 0; i <  size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

bool is_sorted(int *array, int size) {
    for(int i = 1; i < size; i++) {
        if(array[i] < array[i - 1]) {
            return false;
        }
    }
    return true;
}

void randomize(int *array, int size) {
    for(int i = 0; i < size; i++) {
        int random_index = rand() % size;
        int temp = array[i];
        array[i] = array[random_index];
        array[random_index] = temp;
    }
}

void bogo_sort(int *array, int size) {
    while (!is_sorted(array, size)) {
        randomize(array, size);
    }
}

int main() {
    int array_size;

    printf("Enter Array Size: ");
    scanf("%d", &array_size);

    int *data = malloc(array_size * sizeof(int));

    printf("Enter Array Elements: \n");
    for(int i = 0; i < array_size; i++) {
        scanf("%d", &data[i]);
    }

    printf("Original Array: \n");
    print_array(data, array_size);

    bogo_sort(data, array_size);

    printf("Sorted Array: \n");
    print_array(data, array_size);

    free(data);
    return 0;
}
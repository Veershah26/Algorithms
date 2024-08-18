#include <stdio.h>
#include <stdlib.h>

void print_array(int *array, int length) {
    for(int i = 0; i < length; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

// si = SearchInsert

int si_position(int *array, int value, int start, int end) {
    if(start >= end) {
        return(value > array[start]) ? (start + 1) : start;
    }

    int middle = start + (end - start) / 2;

    if(array[middle] == value) {
        return middle + 1;
    } else if(array[middle] > value) {
        return si_position(array, value, start, middle - 1);
    } else {
        return si_position(array, value, middle + 1, end);
    }
}

// bi = binaryInsertion

void bi_sort(int *array, int length) {
    for(int i =1; i < length; i++) {
        int value = array[i];
        int position = si_position(array, value, 0, i - 1);

        for(int j = i; j > position; j--) {
            array[j] = array[j - 1];
        }

        array[position] = value;
    }
}

int main() {
    int length;
    printf("Enter Array Size: ");
    scanf("%d", &length);

    int *array = (int *)malloc(length * sizeof(int));

    printf("Enter Array Elements: \n");
    for(int i = 0; i < length; i++) {
        scanf("%d", &array[i]);
    }

    printf("Original Array: ");
    print_array(array, length);

    bi_sort(array, length);

    printf("Sorted Array: ");
    print_array(array, length);

    free(array);
    return 0;
}